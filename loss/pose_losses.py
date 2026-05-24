"""
第5章 VFMReg 姿态损失函数集合
=========================
针对 6DoF 旋转 + 3D 平移的损失：
- TranslationLoss             : L1 / L2 / Smooth-L1 平移损失
- GeodesicLoss                : 测地线旋转误差 (acos((tr(R^T R')-1)/2))
- ChordalLoss                 : Chordal 距离 ||R - R'||_F
- QuaternionLoss              : 四元数双重最小距离
- Rotation6DLoss              : 论文采用的 6D 连续表示损失
- AnglePenaltyLoss            : 与 GT 旋转角度差超过阈值时惩罚
- VFMRegPoseLoss              : 第5章端到端完整姿态损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =====================================================================
# 0) 工具：6D ↔ R / 四元数 ↔ R
# =====================================================================
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D 表示 → 旋转矩阵 (Zhou et al., 2019)
    输入: [..., 6]，输出: [..., 3, 3]
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """四元数 (w, x, y, z) → 旋转矩阵"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# =====================================================================
# 1) 平移损失
# =====================================================================
class TranslationLoss(nn.Module):
    """平移误差，单位 mm"""

    def __init__(self, mode: str = "l1"):
        super().__init__()
        assert mode in {"l1", "l2", "smooth_l1"}
        self.mode = mode

    def forward(self, t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
        if self.mode == "l1":
            return F.l1_loss(t_pred, t_gt)
        elif self.mode == "l2":
            return F.mse_loss(t_pred, t_gt)
        else:
            return F.smooth_l1_loss(t_pred, t_gt)


# =====================================================================
# 2) 旋转损失：测地线 / Chordal / 6D / Quaternion
# =====================================================================
class GeodesicLoss(nn.Module):
    """SO(3) 测地线损失：θ = acos((tr(R^T R')-1)/2)"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        # R: [..., 3, 3]
        rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cos_theta)
        return theta.mean()


class ChordalLoss(nn.Module):
    """Chordal 距离: ||R - R'||_F^2"""

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        diff = R_pred - R_gt
        return (diff ** 2).sum(dim=(-1, -2)).mean()


class QuaternionLoss(nn.Module):
    """四元数双重最小距离: min(||q-q*||, ||q+q*||)
    解决四元数 q 与 -q 表示同一旋转的对映点问题
    """

    def forward(self, q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
        q_pred = F.normalize(q_pred, dim=-1)
        q_gt = F.normalize(q_gt, dim=-1)
        d1 = ((q_pred - q_gt) ** 2).sum(-1)
        d2 = ((q_pred + q_gt) ** 2).sum(-1)
        return torch.minimum(d1, d2).mean()


class Rotation6DLoss(nn.Module):
    """🌟 论文第5章核心：6D 连续旋转表示损失
    L = ||R(d_pred) - R_gt||_F + λ·L_geodesic
    """

    def __init__(self, w_chordal: float = 1.0, w_geodesic: float = 0.1):
        super().__init__()
        self.chordal = ChordalLoss()
        self.geodesic = GeodesicLoss()
        self.w_c = w_chordal
        self.w_g = w_geodesic

    def forward(self, d6_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        R_pred = rotation_6d_to_matrix(d6_pred)
        return self.w_c * self.chordal(R_pred, R_gt) + self.w_g * self.geodesic(R_pred, R_gt)


# =====================================================================
# 3) 角度惩罚（hinge）
# =====================================================================
class AnglePenaltyLoss(nn.Module):
    """当旋转误差大于阈值 τ 时给予额外 hinge 惩罚（论文增强稳定性）"""

    def __init__(self, threshold_deg: float = 5.0):
        super().__init__()
        self.threshold = threshold_deg / 180.0 * torch.pi
        self.geo = GeodesicLoss()

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        # 已经是平均角度
        angle = self.geo(R_pred, R_gt)
        return F.relu(angle - self.threshold) ** 2


# =====================================================================
# 4) 第5章端到端完整姿态损失（论文 Eq. 5.4）
# =====================================================================
class VFMRegPoseLoss(nn.Module):
    """🌟 论文第5章核心：VFMReg 端到端姿态损失
    L_pose = λ_t · L_trans + λ_r · L_rot6d + λ_g · L_geodesic + λ_h · L_hinge
    论文最优权重: λ_t=1.0, λ_r=1.0, λ_g=0.1, λ_h=0.05
    """

    def __init__(
        self,
        w_trans: float = 1.0,
        w_rot6d: float = 1.0,
        w_geodesic: float = 0.1,
        w_hinge: float = 0.05,
        hinge_threshold_deg: float = 5.0,
    ):
        super().__init__()
        self.trans_loss = TranslationLoss(mode="smooth_l1")
        self.rot6d_loss = Rotation6DLoss(w_chordal=1.0, w_geodesic=0.0)
        self.geo_loss = GeodesicLoss()
        self.hinge_loss = AnglePenaltyLoss(threshold_deg=hinge_threshold_deg)
        self.w_t = w_trans
        self.w_r = w_rot6d
        self.w_g = w_geodesic
        self.w_h = w_hinge

    def forward(
        self,
        d6_pred: torch.Tensor,    # [B, 6]   6D 旋转表示
        t_pred: torch.Tensor,     # [B, 3]   平移 (mm)
        R_gt: torch.Tensor,       # [B, 3, 3]
        t_gt: torch.Tensor,       # [B, 3]
    ) -> Tuple[torch.Tensor, dict]:
        R_pred = rotation_6d_to_matrix(d6_pred)
        l_t = self.trans_loss(t_pred, t_gt)
        l_r = self.rot6d_loss(d6_pred, R_gt)
        l_g = self.geo_loss(R_pred, R_gt)
        l_h = self.hinge_loss(R_pred, R_gt)

        total = self.w_t * l_t + self.w_r * l_r + self.w_g * l_g + self.w_h * l_h
        breakdown = {
            "L_total": total.detach().item(),
            "L_translation": l_t.detach().item(),
            "L_rotation_6d": l_r.detach().item(),
            "L_geodesic": l_g.detach().item(),
            "L_hinge": l_h.detach().item(),
        }
        return total, breakdown
