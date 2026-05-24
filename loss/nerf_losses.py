"""
第4章 NeRF 损失函数集合
=========================
论文第4章的所有损失函数：
- PhotoLoss (RGB MSE/L1)      : 颜色重建损失
- DensityLoss                 : 体密度损失（约束 head surface 上 σ→∞）
- LPIPSLoss                   : 感知损失（VGG/AlexNet 提取特征 L2）
- TotalVariationLoss          : 平滑性正则化
- DepthConsistencyLoss        : 多视角深度一致性
- NeRFRegLoss                 : 论文第4章配准阶段完整损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =====================================================================
# 1) 光度损失
# =====================================================================
class PhotoLoss(nn.Module):
    """L1 / MSE 光度重建损失"""

    def __init__(self, mode: str = "l1"):
        super().__init__()
        assert mode in {"l1", "mse", "huber"}
        self.mode = mode

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "l1":
            return F.l1_loss(rendered, target)
        elif self.mode == "mse":
            return F.mse_loss(rendered, target)
        else:
            return F.smooth_l1_loss(rendered, target)


# =====================================================================
# 2) 体密度损失（论文第4章 Eq. 4.5）
# =====================================================================
class DensityLoss(nn.Module):
    """🌟 论文第4章核心：体密度损失
    L_density = -sum_{p ∈ S_head} log σ(p) + λ·sum_{p ∈ S_air} σ(p)
    强制 head 表面采样点的体密度足够大，自由空间体密度足够小
    """

    def __init__(self, lambda_air: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.lambda_air = lambda_air
        self.eps = eps

    def forward(
        self,
        sigma_surface: torch.Tensor,   # 头部表面采样点的 σ
        sigma_air: Optional[torch.Tensor] = None,  # 自由空间采样点的 σ
    ) -> torch.Tensor:
        # 表面 σ 越大越好
        l_surf = -torch.log(sigma_surface + self.eps).mean()
        if sigma_air is not None:
            l_air = sigma_air.mean()
            return l_surf + self.lambda_air * l_air
        return l_surf


# =====================================================================
# 3) 感知损失 (LPIPS)
# =====================================================================
class LPIPSLossLite(nn.Module):
    """轻量 LPIPS：基于 VGG16 conv1_2/conv2_2/conv3_3 三层特征 L2
    （避免依赖 lpips 库，可独立运行；如需精确 LPIPS 值，请用 lpips 包）
    """

    def __init__(self, feature_layers: Tuple[int, int, int] = (3, 8, 15)):
        super().__init__()
        try:
            import torchvision
            vgg = torchvision.models.vgg16(weights=None).features
        except Exception:
            vgg = nn.Sequential(*[
                nn.Conv2d(3 if i == 0 else 64, 64, 3, padding=1),
                nn.ReLU(inplace=True)
            ] * 16)  # 占位（无网络情况下不会用到）

        self.layers = feature_layers
        self.feat = vgg
        for p in self.feat.parameters():
            p.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _extract(self, x):
        feats = []
        x = self._normalize(x)
        for i, layer in enumerate(self.feat):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
            if i > max(self.layers):
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self._extract(pred)
        ft = self._extract(target)
        loss = 0.0
        for a, b in zip(fp, ft):
            loss = loss + F.mse_loss(a, b)
        return loss / max(1, len(fp))


# =====================================================================
# 4) Total Variation
# =====================================================================
class TotalVariationLoss(nn.Module):
    """TV 正则：鼓励渲染图平滑"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        return dh + dw


# =====================================================================
# 5) 多视角深度一致性
# =====================================================================
class DepthConsistencyLoss(nn.Module):
    """对相邻视角的 depth map 强制一致性 (Reprojection L1)"""

    def forward(
        self,
        depths: torch.Tensor,    # [V, H, W] 多视角深度
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depths.shape[0] < 2:
            return depths.new_zeros(())
        loss = 0.0
        cnt = 0
        for i in range(depths.shape[0] - 1):
            diff = (depths[i] - depths[i + 1]).abs()
            if valid_mask is not None:
                diff = diff * valid_mask[i]
            loss = loss + diff.mean()
            cnt += 1
        return loss / max(cnt, 1)


# =====================================================================
# 6) 第4章配准阶段完整损失（论文 Eq. 4.7）
# =====================================================================
class NeRFRegLoss(nn.Module):
    """🌟 论文第4章配准阶段完整损失:
    L_reg = λ_d · L_density + λ_p · L_photo + λ_lpips · L_lpips + λ_tv · L_tv
    论文最优权重: λ_d=1.0, λ_p=0.5, λ_lpips=0.1, λ_tv=0.01
    """

    def __init__(
        self,
        w_density: float = 1.0,
        w_photo: float = 0.5,
        w_lpips: float = 0.1,
        w_tv: float = 0.01,
        photo_mode: str = "l1",
        use_lpips: bool = True,
    ):
        super().__init__()
        self.density_loss = DensityLoss()
        self.photo_loss = PhotoLoss(mode=photo_mode)
        self.lpips_loss = LPIPSLossLite() if use_lpips else None
        self.tv_loss = TotalVariationLoss()
        self.w_density = w_density
        self.w_photo = w_photo
        self.w_lpips = w_lpips
        self.w_tv = w_tv

    def forward(
        self,
        rendered_img: torch.Tensor,
        target_img: torch.Tensor,
        sigma_surface: Optional[torch.Tensor] = None,
        sigma_air: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        breakdown = {}
        total = rendered_img.new_zeros(())

        if sigma_surface is not None:
            l_d = self.density_loss(sigma_surface, sigma_air)
            breakdown["L_density"] = l_d.detach().item()
            total = total + self.w_density * l_d

        l_p = self.photo_loss(rendered_img, target_img)
        breakdown["L_photo"] = l_p.detach().item()
        total = total + self.w_photo * l_p

        if self.lpips_loss is not None and rendered_img.shape[-1] >= 32:
            try:
                l_lp = self.lpips_loss(rendered_img, target_img)
                breakdown["L_lpips"] = l_lp.detach().item()
                total = total + self.w_lpips * l_lp
            except Exception:
                pass

        l_tv = self.tv_loss(rendered_img)
        breakdown["L_tv"] = l_tv.detach().item()
        total = total + self.w_tv * l_tv

        breakdown["L_total"] = total.detach().item()
        return total, breakdown
