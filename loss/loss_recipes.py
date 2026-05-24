"""
即拿即用的训练配方
=========================
本文件提供了论文 3 个章节最佳实践的训练 loop 模板，
直接 copy-paste 即可在你的 train.py 中使用。

包含：
- recipe_segmentation()    : 第3章 头部分割（YOLOv8n-seg + Sobel）
- recipe_nerf()            : 第4章 NeRF 隐式配准
- recipe_vfmreg_stage1()   : 第5章 合成数据预训练
- recipe_vfmreg_stage2()   : 第5章 真实域微调
- recipe_adaptive_weights(): 多任务自适应权重训练

每个配方都是一个完整可运行的函数，包含：
  ✓ 损失函数初始化（论文最优超参）
  ✓ 优化器配置（学习率、衰减策略）
  ✓ 训练 loop + 验证 loop
  ✓ 日志记录（LossLogger）
  ✓ 检查点保存

运行示例（伪代码，需自行准备 dataloader）：
    from loss.loss_recipes import recipe_vfmreg_stage1
    recipe_vfmreg_stage1(model, train_loader, val_loader,
                         num_epochs=100, ckpt_dir='ckpts/')
"""

from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


# ============================================================
# 配方 1: 第3章 头部分割
# ============================================================
def recipe_segmentation(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    ckpt_dir: str = "ckpts/seg",
    device: str = "cuda",
    log_path: Optional[str] = "logs/seg_training.json",
):
    """
    🌟 论文第3章 YOLOv8n-seg + 多尺度 + Sobel 训练配方

    超参（论文最优）：
      - 多尺度权重: (0.5, 0.3, 0.2)
      - Sobel τ=0.3, λ=0.1
      - Dice 权重 0.5
      - 优化器: AdamW, lr=1e-3, cosine
      - Epochs: 100, batch_size: 16
    """
    from loss import ComboSegLoss, LossLogger, LossMeter

    Path(ckpt_dir).mkdir(exist_ok=True, parents=True)

    # 1. 损失函数（论文最优组合）
    criterion = ComboSegLoss(
        scale_weights=(0.5, 0.3, 0.2),
        sobel_threshold=0.3,
        w_multiscale=1.0,
        w_dice=0.5,
        w_edge=0.1,
    ).to(device)

    # 2. 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # 3. 日志
    logger = LossLogger(out_path=log_path)
    best_iou = 0.0

    for epoch in range(num_epochs):
        # ---- 训练 ----
        model.train()
        meter = LossMeter()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device).long()
            optimizer.zero_grad()

            # YOLOv8n-seg 输出多尺度 logits + 完整 mask
            outputs = model(images)
            preds_ms = outputs["multiscale"]  # [P3, P4, P5]
            pred_full = outputs["full"]

            total, breakdown = criterion(preds_ms, pred_full, masks)
            total.backward()
            optimizer.step()
            meter.update(breakdown, n=images.size(0))

        scheduler.step()
        logger.log(epoch, meter.avg())

        # ---- 验证 ----
        if (epoch + 1) % 5 == 0:
            iou = _evaluate_seg(model, val_loader, device)
            print(f"[Epoch {epoch+1}] train_loss={meter.avg('L_total'):.4f}, val_mIoU={iou:.3f}")
            if iou > best_iou:
                best_iou = iou
                torch.save({"model": model.state_dict(), "epoch": epoch, "iou": iou},
                           Path(ckpt_dir) / "best.pth")

    logger.save()
    print(f"✅ 训练完成, 最佳 mIoU={best_iou:.3f}")


def _evaluate_seg(model, loader, device):
    """简化的 mIoU 评估"""
    model.eval()
    intersect, union = 0.0, 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device).long()
            pred = model(images)["full"].argmax(dim=1)
            for c in range(2):
                pi = pred == c
                gi = masks == c
                intersect += (pi & gi).sum().item()
                union += (pi | gi).sum().item()
    return intersect / max(union, 1)


# ============================================================
# 配方 2: 第4章 NeRF 隐式配准
# ============================================================
def recipe_nerf(
    nerf_model: nn.Module,
    sensor_points: torch.Tensor,    # [N, 3] 头盔传感器点
    target_imgs: list,              # 多视角真实图像列表
    target_masks: list,             # 对应的 mask
    num_iters: int = 200_000,
    lr_pose: float = 1e-3,
    lr_nerf: float = 5e-4,
    ckpt_dir: str = "ckpts/nerf",
    device: str = "cuda",
    log_path: Optional[str] = "logs/nerf_training.json",
):
    """
    🌟 论文第4章 NeRF 配准训练配方

    超参（论文最优）：
      - 体密度损失 weight=1.0
      - 光度 L1 weight=0.5
      - LPIPS weight=0.1
      - TV weight=0.01
      - 总迭代: 200K
      - 学习率: NeRF 5e-4, pose 1e-3
    """
    from loss import NeRFRegLoss, LossLogger, LossMeter

    Path(ckpt_dir).mkdir(exist_ok=True, parents=True)

    criterion = NeRFRegLoss(
        w_density=1.0, w_photo=0.5,
        w_lpips=0.1, w_tv=0.01,
        photo_mode="l1", use_lpips=True,
    ).to(device)

    # 待优化的位姿参数
    rot_6d = torch.zeros(6, device=device, requires_grad=True)
    rot_6d.data[0] = 1.0; rot_6d.data[4] = 1.0  # 初始为单位旋转
    trans = torch.zeros(3, device=device, requires_grad=True)

    pose_optimizer = Adam([rot_6d, trans], lr=lr_pose)
    nerf_optimizer = Adam(nerf_model.parameters(), lr=lr_nerf)

    # NeRF 阶梯式学习率
    scheduler_nerf = MultiStepLR(nerf_optimizer,
                                  milestones=[50_000, 100_000, 150_000],
                                  gamma=0.5)

    logger = LossLogger(out_path=log_path)
    meter = LossMeter()

    for it in range(num_iters):
        # 采样 4096 条光线
        view_idx = it % len(target_imgs)
        target = target_imgs[view_idx].to(device)
        mask = target_masks[view_idx].to(device)

        # 正向：渲染 + 损失
        rendered, sigma_surface, sigma_air = nerf_model.render(
            rot_6d, trans, sensor_points, n_rays=4096,
        )
        total, breakdown = criterion(
            rendered, target,
            sigma_surface=sigma_surface,
            sigma_air=sigma_air,
        )

        pose_optimizer.zero_grad()
        nerf_optimizer.zero_grad()
        total.backward()
        pose_optimizer.step()
        nerf_optimizer.step()
        scheduler_nerf.step()

        meter.update(breakdown)

        if (it + 1) % 1000 == 0:
            avg = meter.avg()
            logger.log(it + 1, avg)
            print(f"[Iter {it+1:>7d}] L_total={avg.get('L_total', 0):.4f} "
                  f"L_density={avg.get('L_density', 0):.4f} "
                  f"L_photo={avg.get('L_photo', 0):.4f}")
            meter.reset()

        if (it + 1) % 50_000 == 0:
            torch.save({
                "nerf": nerf_model.state_dict(),
                "rot_6d": rot_6d.detach(),
                "trans": trans.detach(),
                "iter": it + 1,
            }, Path(ckpt_dir) / f"iter_{it+1}.pth")

    logger.save()
    print("✅ NeRF 训练完成")


# ============================================================
# 配方 3: 第5章 合成数据预训练
# ============================================================
def recipe_vfmreg_stage1(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    lr: float = 1e-4,
    ckpt_dir: str = "ckpts/vfmreg/stage1",
    device: str = "cuda",
    log_path: Optional[str] = "logs/vfmreg_s1.json",
):
    """
    🌟 论文第5章 Stage-1 合成数据预训练配方

    数据：80K 合成多视角图像
    超参（论文最优）：
      - VFMRegPoseLoss + VFMRegRenderLoss 联合
      - 优化器: AdamW + cosine
      - Epochs: 100, batch=64, K=4 视图
      - DINOv3 backbone 冻结，仅训 head
    """
    from loss import VFMRegPoseLoss, VFMRegRenderLoss, LossLogger, LossMeter

    Path(ckpt_dir).mkdir(exist_ok=True, parents=True)

    pose_loss = VFMRegPoseLoss(
        w_trans=1.0, w_rot6d=1.0, w_geodesic=0.1, w_hinge=0.05,
        hinge_threshold_deg=5.0,
    ).to(device)
    render_loss = VFMRegRenderLoss(
        w_iou=1.0, w_rgb=0.5, w_silhouette=0.2,
    ).to(device)
    w_pose, w_render = 1.0, 0.3

    # 仅训 head（backbone 已 frozen）
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    logger = LossLogger(out_path=log_path)
    best_rot_err = float("inf")

    for epoch in range(num_epochs):
        model.train()
        meter = LossMeter()

        for batch in train_loader:
            images = batch["images"].to(device)         # [B, K, 3, H, W]
            R_gt = batch["rotation"].to(device)         # [B, 3, 3]
            t_gt = batch["translation"].to(device)      # [B, 3]
            target_mask = batch["mask"].to(device)      # [B, 1, H, W]
            target_rgb = batch["rgb"].to(device)

            optimizer.zero_grad()
            output = model(images)
            d6_pred, t_pred = output["d6"], output["trans"]

            l_pose, p_break = pose_loss(d6_pred, t_pred, R_gt, t_gt)

            # 可微渲染（PyTorch3D）
            rendered_mask, rendered_rgb = model.render(d6_pred, t_pred)
            l_render, r_break = render_loss(rendered_mask, rendered_rgb,
                                             target_mask, target_rgb)

            total = w_pose * l_pose + w_render * l_render
            total.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()

            combined = {**p_break, **r_break,
                        "L_pose": l_pose.item(), "L_render": l_render.item(),
                        "L_total_combined": total.item()}
            meter.update(combined, n=images.size(0))

        scheduler.step()
        logger.log(epoch, meter.avg())

        # 验证
        if (epoch + 1) % 5 == 0:
            rot_err = _evaluate_pose(model, val_loader, device)
            print(f"[Stage1 Ep {epoch+1}] L_total={meter.avg('L_total_combined'):.4f}, "
                  f"val_rot_err={rot_err:.3f}°")
            if rot_err < best_rot_err:
                best_rot_err = rot_err
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "rot_err": rot_err}, Path(ckpt_dir) / "best.pth")

    logger.save()
    print(f"✅ Stage-1 完成, 最佳旋转误差={best_rot_err:.3f}°")


# ============================================================
# 配方 4: 第5章 真实域微调
# ============================================================
def recipe_vfmreg_stage2(
    model: nn.Module,
    train_loader,
    val_loader,
    pretrain_ckpt: str,
    num_epochs: int = 30,
    lr: float = 1e-5,
    ckpt_dir: str = "ckpts/vfmreg/stage2",
    device: str = "cuda",
    log_path: Optional[str] = "logs/vfmreg_s2.json",
):
    """
    🌟 论文第5章 Stage-2 真实域微调配方

    数据：少量真实标注（300 样本）
    超参（论文最优）：
      - 加载 Stage-1 预训练
      - lr 比 Stage-1 小 10×（1e-5）
      - 仅 30 epoch
      - 移除 hinge 项（避免对真实噪声过拟合）
    """
    from loss import VFMRegPoseLoss, LossLogger, LossMeter

    # 加载预训练
    ckpt = torch.load(pretrain_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"📂 加载预训练 (epoch={ckpt['epoch']}, rot_err={ckpt.get('rot_err', 'N/A')}°)")

    pose_loss = VFMRegPoseLoss(
        w_trans=1.0, w_rot6d=1.0, w_geodesic=0.15,  # 略增 geodesic
        w_hinge=0.0,  # 关闭 hinge
    ).to(device)

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=5e-5)

    Path(ckpt_dir).mkdir(exist_ok=True, parents=True)
    logger = LossLogger(out_path=log_path)
    best_rot_err = float("inf")

    for epoch in range(num_epochs):
        model.train()
        meter = LossMeter()

        for batch in train_loader:
            images = batch["images"].to(device)
            R_gt = batch["rotation"].to(device)
            t_gt = batch["translation"].to(device)

            optimizer.zero_grad()
            output = model(images)
            total, breakdown = pose_loss(output["d6"], output["trans"], R_gt, t_gt)
            total.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 0.5)  # 更严的裁剪
            optimizer.step()
            meter.update(breakdown)

        logger.log(epoch, meter.avg())
        rot_err = _evaluate_pose(model, val_loader, device)

        print(f"[Stage2 Ep {epoch+1}] L_total={meter.avg('L_total'):.4f}, "
              f"val_rot_err={rot_err:.3f}°")

        if rot_err < best_rot_err:
            best_rot_err = rot_err
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "rot_err": rot_err}, Path(ckpt_dir) / "best.pth")

    logger.save()
    print(f"✅ Stage-2 完成, 最佳旋转误差={best_rot_err:.3f}°")


def _evaluate_pose(model, loader, device):
    """计算平均旋转测地误差（度）"""
    from loss import GeodesicLoss, rotation_6d_to_matrix

    model.eval()
    geo = GeodesicLoss().to(device)
    total_err, count = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            R_gt = batch["rotation"].to(device)
            output = model(images)
            R_pred = rotation_6d_to_matrix(output["d6"])
            err_rad = geo(R_pred, R_gt)
            total_err += err_rad.item() * images.size(0)
            count += images.size(0)

    import math
    return (total_err / count) * 180.0 / math.pi


# ============================================================
# 配方 5: 多任务自适应权重训练
# ============================================================
def recipe_adaptive_weights(
    model: nn.Module,
    train_loader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """
    🌟 使用 Kendall et al. 自适应权重的训练配方
    
    适用于：多任务/多 loss 难以手动调权重的场景
    自动学习各 loss 的最优相对权重
    """
    from loss import (AdaptiveWeights, TranslationLoss, GeodesicLoss,
                       DifferentiableIoULoss, rotation_6d_to_matrix, LossLogger)

    aw = AdaptiveWeights(n_tasks=3).to(device)
    trans_loss = TranslationLoss().to(device)
    geo_loss = GeodesicLoss().to(device)
    iou_loss = DifferentiableIoULoss().to(device)

    # 同时优化模型参数和自适应权重
    optimizer = AdamW(list(model.parameters()) + list(aw.parameters()), lr=lr)

    logger = LossLogger(out_path="logs/adaptive_weights.json")

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            R_gt = batch["rotation"].to(device)
            t_gt = batch["translation"].to(device)
            mask_gt = batch["mask"].to(device)

            optimizer.zero_grad()
            output = model(images)

            l1 = trans_loss(output["trans"], t_gt)
            l2 = geo_loss(rotation_6d_to_matrix(output["d6"]), R_gt)
            l3 = iou_loss(output["mask"], mask_gt)

            total = aw([l1, l2, l3])
            total.backward()
            optimizer.step()

            # 记录当前学到的权重
            with torch.no_grad():
                weights = torch.exp(-aw.log_var).cpu().numpy()
            logger.log(epoch, {
                "L_trans": l1.item(),
                "L_rot": l2.item(),
                "L_iou": l3.item(),
                "L_total": total.item(),
                "w_trans": float(weights[0]),
                "w_rot": float(weights[1]),
                "w_iou": float(weights[2]),
            })

        if (epoch + 1) % 5 == 0:
            print(f"[AdaptiveW Ep {epoch+1}] λ={weights}, total_loss={total.item():.4f}")

    logger.save()
    print("✅ 自适应权重训练完成")


# ============================================================
# Demo: 当作脚本运行时输出帮助信息
# ============================================================
if __name__ == "__main__":
    print(__doc__)
    print()
    print("可用配方：")
    print("  - recipe_segmentation()     第3章分割训练")
    print("  - recipe_nerf()             第4章NeRF配准训练")
    print("  - recipe_vfmreg_stage1()    第5章合成预训练")
    print("  - recipe_vfmreg_stage2()    第5章真实微调")
    print("  - recipe_adaptive_weights() 多任务自适应权重")
    print()
    print("示例：")
    print("  >>> from loss.loss_recipes import recipe_vfmreg_stage1")
    print("  >>> recipe_vfmreg_stage1(model, train_loader, val_loader,")
    print("  ...                       num_epochs=100)")
