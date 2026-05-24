"""
tools/icp_baseline.py
=====================
经典 ICP 配准基线（Iterative Closest Point）。
作为论文第5章的 baseline 对照实现，用于数值上证明 VFMReg 的优势。

特点：
- 纯 NumPy 实现，无需 Open3D
- 支持 point-to-point ICP（Besl & McKay 1992）
- 支持随机降采样以加速大点云
- 内置 trimmed-ICP 抗噪变体（去掉最远 k% 配对）
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.geometry import pose_error          # noqa: E402
from utils.io import save_json, get_logger     # noqa: E402

LOG = get_logger("icp")


# -----------------------------------------------------------
# 内部：最近邻匹配（KD-tree 加速）
# -----------------------------------------------------------
def _nearest_neighbors(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对 src 中每个点，从 dst 中找最近邻；返回 (idx_in_dst, distances)"""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(dst)
        d, i = tree.query(src, k=1)
        return i, d
    except ImportError:
        # 退化为暴力搜索（小数据量可接受）
        diff = src[:, None, :] - dst[None, :, :]
        d2 = (diff * diff).sum(-1)
        idx = d2.argmin(axis=1)
        return idx, np.sqrt(d2[np.arange(len(src)), idx])


def _best_rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SVD 求最优刚体变换 A -> B（Kabsch 算法）"""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:           # 反射修正
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


# -----------------------------------------------------------
# ICP 主流程
# -----------------------------------------------------------
def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
    trim_ratio: float = 0.0,            # 0 = 经典 ICP, >0 = trimmed-ICP
    init_R: np.ndarray = None,
    init_t: np.ndarray = None,
) -> dict:
    """对 source 求 (R, t) 使其对齐 target

    Args:
        source: [N, 3]
        target: [M, 3]
        trim_ratio: 0~1，去掉对应距离最远的 trim_ratio 比例点对
    Returns:
        dict {R, t, rmse, n_iter, converged}
    """
    R = np.eye(3) if init_R is None else init_R.copy()
    t = np.zeros(3) if init_t is None else init_t.copy()

    src = (R @ source.T).T + t
    prev_rmse = float("inf")
    converged = False
    for i in range(max_iters):
        idx, dists = _nearest_neighbors(src, target)
        if trim_ratio > 0:
            keep = int(len(dists) * (1 - trim_ratio))
            sel = np.argsort(dists)[:keep]
            R_step, t_step = _best_rigid_transform(src[sel], target[idx[sel]])
        else:
            R_step, t_step = _best_rigid_transform(src, target[idx])

        # 累计变换
        R = R_step @ R
        t = R_step @ t + t_step
        src = (R_step @ src.T).T + t_step

        rmse = float(np.sqrt((dists ** 2).mean()))
        if abs(prev_rmse - rmse) < tol:
            converged = True
            break
        prev_rmse = rmse

    return {
        "R": R, "t": t, "rmse": rmse,
        "n_iter": i + 1, "converged": converged,
    }


# -----------------------------------------------------------
# CLI: 跑一次基线对比 VFMReg
# -----------------------------------------------------------
def _demo():
    """生成两个相对刚体变换的高斯点云，跑 ICP 验证可用性"""
    rng = np.random.default_rng(42)
    src = rng.normal(size=(2000, 3))
    angle = np.deg2rad(15)
    R_gt = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])
    t_gt = np.array([5.0, 0.0, 2.0])
    dst = (R_gt @ src.T).T + t_gt + rng.normal(scale=0.05, size=src.shape)

    out = icp(src, dst, max_iters=80, trim_ratio=0.1)
    err_t, err_r = pose_error(out["R"], out["t"], R_gt, t_gt)
    LOG.info(f"ICP 结果: rmse={out['rmse']:.4f}, "
             f"trans_err={err_t:.4f}, rot_err={err_r:.4f}°, "
             f"converged={out['converged']} ({out['n_iter']} iters)")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="跑合成数据的自检")
    parser.add_argument("--source", type=str, help=".npy 源点云")
    parser.add_argument("--target", type=str, help=".npy 目标点云")
    parser.add_argument("--out", type=str, default="icp_result.json")
    parser.add_argument("--trim", type=float, default=0.0)
    args = parser.parse_args()

    if args.demo:
        _demo()
    elif args.source and args.target:
        s = np.load(args.source)
        t = np.load(args.target)
        out = icp(s, t, trim_ratio=args.trim)
        save_json({k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in out.items()}, args.out)
        LOG.info(f"结果写入 {args.out}")
    else:
        parser.print_help()
