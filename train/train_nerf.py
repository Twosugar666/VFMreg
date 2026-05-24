"""
NeRF训练脚本
- Adam优化器，初始学习率5e-4
- 指数衰减学习率调度（每10万步衰减0.5）
- 训练总迭代次数200K
- 每次迭代随机采样4096条光线
- 粗-精两级网络联合训练
- 密度正则化损失
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nerf import NeRF, NeRFCoarse, NeRFFine, VolumeRenderer
from preprocess.dataset import NeRFDataset


def train_nerf(args):
    """NeRF训练主函数
    训练过程：
    1. 加载多视角图像和COLMAP位姿
    2. 初始化粗/精两级NeRF网络
    3. 每次迭代随机采样4096条光线
    4. 光度重建损失 + 密度正则化
    5. 200K次迭代，约4小时（单卡A100）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ===== 1. 数据加载 =====
    dataset = NeRFDataset(
        data_root=args.data_root,
        image_size=(args.image_height, args.image_width),
        num_rays_per_batch=args.num_rays,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    print(f"加载了 {len(dataset)} 张训练图像")

    # ===== 2. 模型初始化 =====
    # 粗网络
    nerf_coarse = NeRFCoarse(
        pos_encoding_freqs=args.pos_freqs,
        dir_encoding_freqs=args.dir_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        skip_layer=args.skip_layer,
    ).to(device)

    # 精网络
    nerf_fine = NeRFFine(
        pos_encoding_freqs=args.pos_freqs,
        dir_encoding_freqs=args.dir_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        skip_layer=args.skip_layer,
    ).to(device)

    # 体渲染器
    renderer = VolumeRenderer(
        near=args.near,
        far=args.far,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        white_background=args.white_bg,
    ).to(device)

    print(f"粗网络参数量: {sum(p.numel() for p in nerf_coarse.parameters()) / 1e6:.2f}M")
    print(f"精网络参数量: {sum(p.numel() for p in nerf_fine.parameters()) / 1e6:.2f}M")

    # ===== 3. 优化器和学习率调度 =====
    params = list(nerf_coarse.parameters()) + list(nerf_fine.parameters())
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    # 指数衰减学习率：每10万步衰减0.5
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.5 ** (1.0 / args.lr_decay_steps)
    )

    # ===== 4. 训练循环 =====
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = open(os.path.join(args.output_dir, 'train_log.txt'), 'w')

    global_step = 0
    best_psnr = 0.0
    start_time = time.time()

    print(f"开始训练，总迭代次数: {args.num_iterations}")

    while global_step < args.num_iterations:
        for batch in dataloader:
            if global_step >= args.num_iterations:
                break

            rays_o = batch['rays_o'].squeeze(0).to(device)  # [N_rays, 3]
            rays_d = batch['rays_d'].squeeze(0).to(device)  # [N_rays, 3]
            target_rgb = batch['target_rgb'].squeeze(0).to(device)  # [N_rays, 3]

            # 前向传播
            results = renderer(nerf_coarse, nerf_fine, rays_o, rays_d, perturb=True)

            # ===== 损失计算 =====
            # 粗网络光度损失
            loss_coarse = nn.MSELoss()(results['coarse']['rgb'], target_rgb)

            # 精网络光度损失
            loss_fine = nn.MSELoss()(results['fine']['rgb'], target_rgb)

            # 密度正则化（对空旷区域的密度施加L1惩罚）
            if args.density_reg_weight > 0:
                # 对累积不透明度较低的区域进行正则化
                acc_coarse = results['coarse']['acc']
                density_reg = args.density_reg_weight * torch.mean(
                    torch.relu(0.5 - acc_coarse) ** 2
                )
            else:
                density_reg = torch.tensor(0.0, device=device)

            # 总损失
            loss = loss_coarse + loss_fine + density_reg

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ===== 日志记录 =====
            global_step += 1

            if global_step % args.log_interval == 0:
                # 计算PSNR
                with torch.no_grad():
                    mse = nn.MSELoss()(results['fine']['rgb'], target_rgb)
                    psnr = -10.0 * torch.log10(mse)

                elapsed = time.time() - start_time
                lr_current = optimizer.param_groups[0]['lr']

                log_msg = (
                    f"Step {global_step}/{args.num_iterations} | "
                    f"Loss: {loss.item():.6f} | "
                    f"PSNR: {psnr.item():.2f} dB | "
                    f"LR: {lr_current:.2e} | "
                    f"Time: {elapsed/3600:.2f}h"
                )
                print(log_msg)
                log_file.write(log_msg + '\n')
                log_file.flush()

            # ===== 模型保存 =====
            if global_step % args.save_interval == 0:
                checkpoint = {
                    'step': global_step,
                    'nerf_coarse': nerf_coarse.state_dict(),
                    'nerf_fine': nerf_fine.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_path = os.path.join(args.output_dir, f'checkpoint_{global_step:06d}.pth')
                torch.save(checkpoint, save_path)
                print(f"模型已保存: {save_path}")

                # 保存最优模型
                with torch.no_grad():
                    mse = nn.MSELoss()(results['fine']['rgb'], target_rgb)
                    psnr = -10.0 * torch.log10(mse)
                if psnr.item() > best_psnr:
                    best_psnr = psnr.item()
                    best_path = os.path.join(args.output_dir, 'best_model.pth')
                    torch.save(checkpoint, best_path)

    # 保存最终模型
    final_checkpoint = {
        'step': global_step,
        'nerf_coarse': nerf_coarse.state_dict(),
        'nerf_fine': nerf_fine.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pth'))

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
    print(f"最佳PSNR: {best_psnr:.2f} dB")

    log_file.close()
    return nerf_coarse, nerf_fine


def main():
    parser = argparse.ArgumentParser(description='NeRF训练脚本')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/nerf', help='输出目录')
    parser.add_argument('--image_height', type=int, default=256, help='图像高度')
    parser.add_argument('--image_width', type=int, default=256, help='图像宽度')

    # 模型参数
    parser.add_argument('--pos_freqs', type=int, default=10, help='位置编码阶数')
    parser.add_argument('--dir_freqs', type=int, default=4, help='方向编码阶数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=8, help='网络层数')
    parser.add_argument('--skip_layer', type=int, default=4, help='跳跃连接层')

    # 渲染参数
    parser.add_argument('--near', type=float, default=0.1, help='近平面')
    parser.add_argument('--far', type=float, default=5.0, help='远平面')
    parser.add_argument('--num_coarse_samples', type=int, default=64, help='粗采样点数')
    parser.add_argument('--num_fine_samples', type=int, default=128, help='精采样点数')
    parser.add_argument('--white_bg', action='store_true', help='白色背景')
    parser.add_argument('--num_rays', type=int, default=4096, help='每批次光线数')

    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=200000, help='总迭代次数')
    parser.add_argument('--lr', type=float, default=5e-4, help='初始学习率')
    parser.add_argument('--lr_decay_steps', type=int, default=100000, help='学习率衰减步数')
    parser.add_argument('--density_reg_weight', type=float, default=0.01, help='密度正则化权重')

    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=10000, help='保存间隔')

    args = parser.parse_args()
    train_nerf(args)


if __name__ == '__main__':
    main()
