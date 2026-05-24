"""
可微渲染损失（第5章 VFMReg 框架部分）
=========================
- DifferentiableIoULoss      : 软 IoU 损失，可微渲染轮廓监督
- SilhouetteL1Loss           : 轮廓 L1
- MaskedRGBLoss              : 仅在 mask 内计算 RGB 损失
- MultiViewConsistencyLoss   : 多视角一致性损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1) Soft IoU
# =====================================================================
class DifferentiableIoULoss(nn.Module):
    """🌟 可微 IoU 损失
    L_iou = 1 - sum(P*G) / sum(P+G - P*G)
    其中 P = sigmoid(rendered) 是软 mask
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, rendered_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        # rendered_mask: [B, 1, H, W]  (0~1 软概率)
        # target_mask:   [B, 1, H, W]  (0/1)
        inter = (rendered_mask * target_mask).sum(dim=(1, 2, 3))
        union = rendered_mask.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3)) - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        return (1.0 - iou).mean()


class SilhouetteL1Loss(nn.Module):
    """渲染轮廓与目标 mask 的 L1 损失"""

    def forward(self, rendered_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(rendered_mask, target_mask)


# =====================================================================
# 2) Masked RGB
# =====================================================================
class MaskedRGBLoss(nn.Module):
    """仅在前景 mask 内计算 RGB 损失（论文域随机化用）"""

    def __init__(self, mode: str = "l1"):
        super().__init__()
        assert mode in {"l1", "l2"}
        self.mode = mode

    def forward(
        self,
        rendered_rgb: torch.Tensor,    # [B, 3, H, W]
        target_rgb: torch.Tensor,
        mask: torch.Tensor,            # [B, 1, H, W]
    ) -> torch.Tensor:
        if self.mode == "l1":
            diff = (rendered_rgb - target_rgb).abs()
        else:
            diff = (rendered_rgb - target_rgb) ** 2
        diff = diff * mask
        denom = mask.sum().clamp(min=1.0) * 3.0
        return diff.sum() / denom


# =====================================================================
# 3) 多视角一致性
# =====================================================================
class MultiViewConsistencyLoss(nn.Module):
    """对 K 个视角的预测姿态做一致性约束（同一被试不同视角应预测同一姿态）"""

    def forward(
        self,
        d6_views: torch.Tensor,    # [B, V, 6]
        t_views: torch.Tensor,     # [B, V, 3]
    ) -> torch.Tensor:
        # 中心化
        d6_mean = d6_views.mean(dim=1, keepdim=True)
        t_mean = t_views.mean(dim=1, keepdim=True)
        l_d6 = ((d6_views - d6_mean) ** 2).mean()
        l_t = ((t_views - t_mean) ** 2).mean()
        return l_d6 + l_t


# =====================================================================
# 4) 复合渲染损失（第5章 Stage-1 合成数据预训练阶段）
# =====================================================================
class VFMRegRenderLoss(nn.Module):
    """🌟 第5章合成数据训练用的渲染损失
    L = λ_iou·L_iou + λ_rgb·L_masked_rgb + λ_sil·L_silhouette
    论文最优权重: λ_iou=1.0, λ_rgb=0.5, λ_sil=0.2
    """

    def __init__(
        self,
        w_iou: float = 1.0,
        w_rgb: float = 0.5,
        w_silhouette: float = 0.2,
    ):
        super().__init__()
        self.iou = DifferentiableIoULoss()
        self.rgb = MaskedRGBLoss()
        self.sil = SilhouetteL1Loss()
        self.w_iou = w_iou
        self.w_rgb = w_rgb
        self.w_sil = w_silhouette

    def forward(
        self,
        rendered_mask: torch.Tensor,
        rendered_rgb: torch.Tensor,
        target_mask: torch.Tensor,
        target_rgb: torch.Tensor,
    ):
        l_iou = self.iou(rendered_mask, target_mask)
        l_rgb = self.rgb(rendered_rgb, target_rgb, target_mask)
        l_sil = self.sil(rendered_mask, target_mask)
        total = self.w_iou * l_iou + self.w_rgb * l_rgb + self.w_sil * l_sil
        breakdown = {
            "L_total": total.detach().item(),
            "L_iou": l_iou.detach().item(),
            "L_rgb": l_rgb.detach().item(),
            "L_silhouette": l_sil.detach().item(),
        }
        return total, breakdown
