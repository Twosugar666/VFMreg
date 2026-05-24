"""
utils/geometry.py
=================
位姿与几何工具：旋转表示互转、6D / quat / SO(3) / Rodrigues 之间的桥接，
点云配准误差度量，相机投影/逆投影。

这些函数在 train_vfmreg / inference / evaluate / tools 里都会被复用。
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:                                            # pragma: no cover
    _HAS_TORCH = False


# ===================================================================
# 旋转表示
# ===================================================================
def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues 向量 -> 3×3 旋转矩阵（NumPy）"""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵 -> Rodrigues 向量"""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
    return theta * np.array([rx, ry, rz])


def quaternion_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) -> 旋转矩阵"""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_6d_np(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 6D 表示（论文第5章 Zhou et al. 2019）"""
    R = np.asarray(R)
    return R[:, :2].T.reshape(-1)  # 取前两列拼接


def sixd_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """6D -> 旋转矩阵（Gram-Schmidt 重新正交化）"""
    a1 = d6[0:3]
    a2 = d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - (b1 * a2).sum() * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


# ===================================================================
# 误差度量（用于第3/4/5 章的评估指标）
# ===================================================================
def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """两个旋转矩阵之间的测地线角度差（°）"""
    cos_theta = (np.trace(R_pred.T @ R_gt) - 1) / 2
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def translation_error_mm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """平移误差（欧式距离 mm）"""
    return float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt)))


def pose_error(R_pred, t_pred, R_gt, t_gt) -> Tuple[float, float]:
    """同时返回 (translation_mm, rotation_deg)"""
    return translation_error_mm(t_pred, t_gt), rotation_error_deg(R_pred, R_gt)


# ===================================================================
# 相机投影
# ===================================================================
def project_points(
    pts3d: np.ndarray,           # [N, 3]
    K: np.ndarray,               # [3, 3] 内参
    R: np.ndarray,               # [3, 3] 外参旋转
    t: np.ndarray,               # [3]    外参平移
) -> np.ndarray:
    """3D 点投影到像素平面 -> [N, 2]"""
    pts_cam = (R @ pts3d.T).T + t.reshape(1, 3)
    pts_img = (K @ pts_cam.T).T
    return pts_img[:, :2] / (pts_img[:, 2:3] + 1e-12)


def unproject_depth(
    depth: np.ndarray,           # [H, W]
    K: np.ndarray,               # [3, 3]
) -> np.ndarray:
    """逆投影 RGB-D 深度到相机系点云 -> [N, 3]"""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts[pts[:, 2] > 0]


# ===================================================================
# Torch 兼容版（如可用）
# ===================================================================
if _HAS_TORCH:
    def rotation_6d_to_matrix(d6: "torch.Tensor") -> "torch.Tensor":
        """6D -> R (B, 3, 3) torch 版本，与 loss/pose_losses 对齐"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    def geodesic_distance_torch(R1: "torch.Tensor", R2: "torch.Tensor") -> "torch.Tensor":
        cos = ((R1.transpose(-2, -1) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        return torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
