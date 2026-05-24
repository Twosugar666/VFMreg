"""
VFMReg端到端配准框架训练脚本
两阶段训练策略：
- 阶段一：合成数据预训练（80K样本，100 epoch，~20h A100）
- 阶段二：真实域微调（50组真实数据，10 epoch，~2h）

训练超参数：
- 优化器：Adam (β1=0.9, β2=0.999)
- 阶段一学习率：1e-4，余弦退火→1e-6
- 阶段二学习率：1e-5（固定）
- 批量大小：32（×4视图）
- 混合精度：bf16
- 可训练模块：注意力层 + 回归头（backbone冻结）
"""

import os
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vfmreg import VFMReg, VFMRegLoss, compute_rotation_error, compute_translation_error
from preprocess.dataset import SyntheticDataset, MEGHeadDataset


class CosineAnnealingWarmup:
    """余弦退火学习率调度（带Warmup）"""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # 线性Warmup
            scale = self.step_count / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_vfmreg(args):
    """VFMReg训练主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ===== 1. 模型初始化 =====
    model = VFMReg(
        backbone_name=args.backbone,
        feature_dim=args.feature_dim,
        num_views=args.num_views,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        freeze_backbone=True,
    ).to(device)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.1f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.1f}M")

    # 损失函数
    criterion = VFMRegLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_rot=args.lambda_rot,
        lambda_trans=args.lambda_trans,
    )

    # ===== 阶段一：合成数据预训练 =====
    if args.stage in ['all', 'pretrain']:
        print("\n" + "=" * 60)
        print("阶段一：合成数据预训练")
        print("=" * 60)

        # 加载合成数据集
        train_dataset = SyntheticDataset(
            data_root=args.synthetic_data_root,
            split='train',
            num_views=args.num_views,
            image_size=args.image_size,
            augment=True,
        )
        val_dataset = SyntheticDataset(
            data_root=args.synthetic_data_root,
            split='val',
            num_views=args.num_views,
            image_size=args.image_size,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

        # 优化器（仅优化注意力层和回归头）
        trainable_modules = nn.ModuleList([
            model.cross_view_attention,
            model.pose_head,
        ])
        optimizer = optim.Adam(
            trainable_modules.parameters(),
            lr=args.lr_pretrain,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )

        # 学习率调度
        total_steps = len(train_loader) * args.epochs_pretrain
        scheduler = CosineAnnealingWarmup(
            optimizer,
            warmup_steps=total_steps // 20,
            total_steps=total_steps,
            min_lr=1e-6,
        )

        # 混合精度训练
        scaler = GradScaler() if args.use_amp else None

        # 训练循环
        best_val_rot_error = float('inf')
        os.makedirs(args.output_dir, exist_ok=True)

        for epoch in range(args.epochs_pretrain):
            model.train()
            model.backbone.eval()  # backbone始终eval模式

            epoch_losses = []
            epoch_rot_errors = []
            epoch_trans_errors = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs_pretrain}")
            for batch in pbar:
                images = batch['images'].to(device)      # [B, K, 3, H, W]
                masks = batch['masks'].to(device)        # [B, K, 1, H, W]
                R_gt = batch['rotation'].to(device)      # [B, 3, 3]
                t_gt = batch['translation'].to(device)   # [B, 3]

                optimizer.zero_grad()

                if args.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        predictions = model(images, masks)
                        targets = {'rotation_matrix': R_gt, 'translation': t_gt}
                        losses = criterion(predictions, targets)

                    scaler.scale(losses['total_loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    predictions = model(images, masks)
                    targets = {'rotation_matrix': R_gt, 'translation': t_gt}
                    losses = criterion(predictions, targets)

                    losses['total_loss'].backward()
                    optimizer.step()

                scheduler.step()

                # 计算评估指标
                with torch.no_grad():
                    rot_error = compute_rotation_error(
                        predictions['rotation_matrix'], R_gt
                    ).mean()
                    trans_error = compute_translation_error(
                        predictions['translation'], t_gt
                    ).mean()

                epoch_losses.append(losses['total_loss'].item())
                epoch_rot_errors.append(rot_error.item())
                epoch_trans_errors.append(trans_error.item())

                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'rot_err': f"{rot_error.item():.3f}°",
                    'trans_err': f"{trans_error.item():.3f}mm",
                    'lr': f"{scheduler.get_lr():.2e}",
                })

            # Epoch统计
            avg_loss = np.mean(epoch_losses)
            avg_rot = np.mean(epoch_rot_errors)
            avg_trans = np.mean(epoch_trans_errors)
            print(f"\nEpoch {epoch+1} 训练 | Loss: {avg_loss:.4f} | "
                  f"Rot: {avg_rot:.3f}° | Trans: {avg_trans:.3f}mm")

            # 验证
            if (epoch + 1) % args.val_interval == 0:
                val_rot, val_trans = validate(model, val_loader, device)
                print(f"Epoch {epoch+1} 验证 | Rot: {val_rot:.3f}° | Trans: {val_trans:.3f}mm")

                # 保存最优模型
                if val_rot < best_val_rot_error:
                    best_val_rot_error = val_rot
                    save_path = os.path.join(args.output_dir, 'best_pretrain.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rot_error': val_rot,
                        'val_trans_error': val_trans,
                    }, save_path)
                    print(f"最优模型已保存: {save_path}")

        # 保存阶段一最终模型
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'pretrain_final.pth'))

    # ===== 阶段二：真实域微调 =====
    if args.stage in ['all', 'finetune']:
        print("\n" + "=" * 60)
        print("阶段二：真实域微调")
        print("=" * 60)

        # 加载预训练权重
        if args.stage == 'finetune' and args.pretrain_ckpt:
            model.load_state_dict(torch.load(args.pretrain_ckpt, map_location=device))
            print(f"加载预训练权重: {args.pretrain_ckpt}")

        # 加载真实数据集
        real_train_dataset = MEGHeadDataset(
            data_root=args.real_data_root,
            split='train',
            num_views=args.num_views,
            image_size=args.image_size,
        )
        real_val_dataset = MEGHeadDataset(
            data_root=args.real_data_root,
            split='val',
            num_views=args.num_views,
            image_size=args.image_size,
        )

        real_train_loader = DataLoader(
            real_train_dataset, batch_size=args.batch_size_finetune,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        real_val_loader = DataLoader(
            real_val_dataset, batch_size=args.batch_size_finetune,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"真实训练集大小: {len(real_train_dataset)}")
        print(f"真实验证集大小: {len(real_val_dataset)}")

        # 微调优化器（学习率为预训练的1/10）
        finetune_modules = nn.ModuleList([
            model.cross_view_attention,
            model.pose_head,
        ])
        optimizer_ft = optim.Adam(
            finetune_modules.parameters(),
            lr=args.lr_finetune,
            betas=(0.9, 0.999),
        )

        # 微调训练循环
        best_val_rot = float('inf')

        for epoch in range(args.epochs_finetune):
            model.train()
            model.backbone.eval()

            epoch_losses = []
            pbar = tqdm(real_train_loader, desc=f"微调 Epoch {epoch+1}/{args.epochs_finetune}")

            for batch in pbar:
                images = batch['images'].to(device)
                masks = batch['masks'].to(device)
                R_gt = batch['rotation'].to(device)
                t_gt = batch['translation'].to(device)

                optimizer_ft.zero_grad()

                predictions = model(images, masks)
                targets = {'rotation_matrix': R_gt, 'translation': t_gt}
                losses = criterion(predictions, targets)

                losses['total_loss'].backward()
                optimizer_ft.step()

                epoch_losses.append(losses['total_loss'].item())

                with torch.no_grad():
                    rot_error = compute_rotation_error(
                        predictions['rotation_matrix'], R_gt
                    ).mean()
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'rot_err': f"{rot_error.item():.3f}°",
                })

            # 验证
            val_rot, val_trans = validate(model, real_val_loader, device)
            print(f"微调 Epoch {epoch+1} | Rot: {val_rot:.3f}° | Trans: {val_trans:.3f}mm")

            if val_rot < best_val_rot:
                best_val_rot = val_rot
                save_path = os.path.join(args.output_dir, 'best_finetune.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_rot_error': val_rot,
                    'val_trans_error': val_trans,
                }, save_path)
                print(f"最优微调模型已保存: {save_path}")

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'finetune_final.pth'))
        print(f"\n微调完成！最佳旋转误差: {best_val_rot:.3f}°")


@torch.no_grad()
def validate(model, val_loader, device):
    """验证函数"""
    model.eval()
    rot_errors = []
    trans_errors = []

    for batch in val_loader:
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)
        R_gt = batch['rotation'].to(device)
        t_gt = batch['translation'].to(device)

        predictions = model(images, masks)

        rot_error = compute_rotation_error(predictions['rotation_matrix'], R_gt)
        trans_error = compute_translation_error(predictions['translation'], t_gt)

        rot_errors.extend(rot_error.cpu().numpy().tolist())
        trans_errors.extend(trans_error.cpu().numpy().tolist())

    model.train()
    return np.mean(rot_errors), np.mean(trans_errors)


def main():
    parser = argparse.ArgumentParser(description='VFMReg训练脚本')

    # 数据参数
    parser.add_argument('--synthetic_data_root', type=str, default='./data/synthetic',
                        help='合成数据目录')
    parser.add_argument('--real_data_root', type=str, default='./data/real',
                        help='真实数据目录')
    parser.add_argument('--output_dir', type=str, default='./outputs/vfmreg',
                        help='输出目录')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='dinov2_vitl14',
                        help='视觉基础模型')
    parser.add_argument('--feature_dim', type=int, default=1024, help='特征维度')
    parser.add_argument('--num_views', type=int, default=4, help='视图数量K')
    parser.add_argument('--num_attention_layers', type=int, default=4, help='注意力层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')

    # 损失函数权重
    parser.add_argument('--alpha', type=float, default=1.0, help='几何损失权重')
    parser.add_argument('--beta', type=float, default=0.5, help='渲染L1损失权重')
    parser.add_argument('--gamma', type=float, default=0.3, help='轮廓IoU损失权重')
    parser.add_argument('--lambda_rot', type=float, default=1.0, help='旋转损失权重')
    parser.add_argument('--lambda_trans', type=float, default=0.5, help='平移损失权重')

    # 训练参数
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'pretrain', 'finetune'], help='训练阶段')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--epochs_pretrain', type=int, default=100, help='预训练epoch数')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='微调epoch数')
    parser.add_argument('--batch_size', type=int, default=32, help='预训练批量大小')
    parser.add_argument('--batch_size_finetune', type=int, default=8, help='微调批量大小')
    parser.add_argument('--lr_pretrain', type=float, default=1e-4, help='预训练学习率')
    parser.add_argument('--lr_finetune', type=float, default=1e-5, help='微调学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--val_interval', type=int, default=5, help='验证间隔(epoch)')

    args = parser.parse_args()
    train_vfmreg(args)


if __name__ == '__main__':
    main()
