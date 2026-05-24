"""
第3章 头部分割损失函数集合
=========================
论文第3章涉及的所有损失函数实现：
- CrossEntropyLoss            : 标准像素级交叉熵
- DiceLoss                    : 适用于类别不平衡的 Dice 系数损失
- FocalLoss                   : 困难样本加权（解决前景/背景不平衡）
- TverskyLoss                 : Dice 推广，可调节 FP/FN 权重
- BoundaryLoss                : 基于距离图的边界感知损失
- MultiScaleCELoss            : 论文核心：多尺度交叉熵 (L3/L4/L5 权重 0.5/0.3/0.2)
- SobelEdgeLoss               : 论文核心：Sobel 边缘增强 (阈值 0.3)
- ComboSegLoss                : 第3章完整复合损失 (Multi-Scale CE + Sobel + Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple


# =====================================================================
# 1) 基础损失：CrossEntropy / Dice / Focal / Tversky
# =====================================================================
class DiceLoss(nn.Module):
    """Dice 损失：1 - 2|P∩G| / (|P|+|G|)"""

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred:   [B, C, H, W] logits
        # target: [B, H, W] long  或  [B, C, H, W] one-hot
        if pred.shape != target.shape:
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()
        prob = pred.softmax(dim=1)
        dims = (0, 2, 3)
        inter = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLoss(nn.Module):
    """Focal Loss: -α(1-p)^γ * log(p)，重点关注困难样本"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target.long(), reduction="none")
        p_t = torch.exp(-ce)
        loss = self.alpha * (1.0 - p_t) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


class TverskyLoss(nn.Module):
    """Tversky Loss：α 控制 FP，β 控制 FN
    α=β=0.5 -> Dice;  α=0.3, β=0.7 -> 偏向召回
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()
        prob = pred.softmax(dim=1)
        dims = (0, 2, 3)
        TP = (prob * target).sum(dim=dims)
        FP = (prob * (1.0 - target)).sum(dim=dims)
        FN = ((1.0 - prob) * target).sum(dim=dims)
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1.0 - tversky).mean()


# =====================================================================
# 2) 边界相关
# =====================================================================
class BoundaryLoss(nn.Module):
    """基于距离变换的边界损失（Kervadec et al., 2019 简化版）
    使用 Sobel 算子近似 ground truth 距离图
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sy", sobel_y.view(1, 1, 3, 3))

    def _edge_map(self, m: torch.Tensor) -> torch.Tensor:
        # m: [B, 1, H, W] float
        gx = F.conv2d(m, self.sx, padding=1)
        gy = F.conv2d(m, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 取前景概率
        prob_fg = pred.softmax(dim=1)[:, 1:2]
        if target.dim() == 3:
            tgt_fg = (target == 1).float().unsqueeze(1)
        else:
            tgt_fg = target[:, 1:2].float()
        edge_pred = self._edge_map(prob_fg)
        edge_tgt = self._edge_map(tgt_fg)
        return F.l1_loss(edge_pred, edge_tgt)


class SobelEdgeLoss(nn.Module):
    """🌟 论文第3章核心：Sobel 边缘增强损失
    L_edge = ||Sobel(P) - Sobel(G)||_1，在边缘区域 (|∇G|>τ) 加权
    论文最优阈值 τ=0.3
    """

    def __init__(self, threshold: float = 0.3, weight: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sy", sobel_y.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob_fg = pred.softmax(dim=1)[:, 1:2]
        if target.dim() == 3:
            tgt_fg = (target == 1).float().unsqueeze(1)
        else:
            tgt_fg = target[:, 1:2].float()
        gp = self._grad(prob_fg)
        gt = self._grad(tgt_fg)
        edge_mask = (gt > self.threshold).float()
        # 仅在边缘区域计算 L1
        diff = (gp - gt).abs() * edge_mask
        denom = edge_mask.sum().clamp(min=1.0)
        return self.weight * diff.sum() / denom


# =====================================================================
# 3) 多尺度损失：论文第3章核心
# =====================================================================
class MultiScaleCELoss(nn.Module):
    """🌟 论文第3章核心：多尺度交叉熵损失
    在 YOLOv8n-seg 的 P3/P4/P5 三个 FPN 输出层分别计算 CE，
    论文最优权重: λ3=0.5, λ4=0.3, λ5=0.2
    """

    def __init__(
        self,
        scale_weights: Sequence[float] = (0.5, 0.3, 0.2),
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.scale_weights = list(scale_weights)
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(
        self,
        preds: Sequence[torch.Tensor],     # [P3, P4, P5] 不同尺度 logits
        target: torch.Tensor,              # [B, H, W]   原始尺寸 long
    ) -> Tuple[torch.Tensor, dict]:
        assert len(preds) == len(self.scale_weights), (
            f"尺度数 {len(preds)} 不匹配权重数 {len(self.scale_weights)}"
        )
        total = 0.0
        breakdown = {}
        for i, (p, w) in enumerate(zip(preds, self.scale_weights)):
            # 把 target 下采样到 p 的尺寸
            tgt = F.interpolate(target.unsqueeze(1).float(), size=p.shape[-2:], mode="nearest")
            tgt = tgt.squeeze(1).long()
            ce = F.cross_entropy(
                p, tgt,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
            )
            breakdown[f"L_scale_{i + 3}"] = ce.detach().item()
            total = total + w * ce
        return total, breakdown


# =====================================================================
# 4) 第3章完整复合损失
# =====================================================================
class ComboSegLoss(nn.Module):
    """🌟 论文第3章最终采用的复合损失 (Eq. 3.6):
    L = λ_ms·L_multi_scale  +  λ_dice·L_dice  +  λ_edge·L_sobel
    论文最优组合: λ_ms=1.0, λ_dice=0.5, λ_edge=0.1
    """

    def __init__(
        self,
        scale_weights: Sequence[float] = (0.5, 0.3, 0.2),
        sobel_threshold: float = 0.3,
        w_multiscale: float = 1.0,
        w_dice: float = 0.5,
        w_edge: float = 0.1,
    ):
        super().__init__()
        self.ms_loss = MultiScaleCELoss(scale_weights=scale_weights)
        self.dice_loss = DiceLoss()
        self.edge_loss = SobelEdgeLoss(threshold=sobel_threshold)
        self.w_ms = w_multiscale
        self.w_dice = w_dice
        self.w_edge = w_edge

    def forward(
        self,
        preds_multiscale: Sequence[torch.Tensor],   # [P3, P4, P5]
        pred_full: torch.Tensor,                    # [B, C, H, W] 上采样到原尺寸
        target: torch.Tensor,                       # [B, H, W] long
    ) -> Tuple[torch.Tensor, dict]:
        l_ms, ms_break = self.ms_loss(preds_multiscale, target)
        l_dice = self.dice_loss(pred_full, target)
        l_edge = self.edge_loss(pred_full, target)
        total = self.w_ms * l_ms + self.w_dice * l_dice + self.w_edge * l_edge
        breakdown = {
            "L_total": total.detach().item(),
            "L_multiscale": l_ms.detach().item(),
            "L_dice": l_dice.detach().item(),
            "L_edge": l_edge.detach().item(),
            **ms_break,
        }
        return total, breakdown
