"""
tools/mesh_utils.py
===================
mesh 处理小工具：从 .obj 加载顶点、随机采样、归一化、导出。
用于把 toukui/DiffRegistration/head/*.obj 转为训练用的点云。
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


# -----------------------------------------------------------
# 1. .obj 解析（不依赖 trimesh，纯 Python）
# -----------------------------------------------------------
def load_obj_vertices(path: str) -> np.ndarray:
    """只解析顶点（v 行），返回 [N, 3] float32"""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float32)


def load_obj_full(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """解析顶点 + 三角面索引"""
    verts, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.strip().split()[1:]
                # f a/b/c d/e/f g/h/i 取 a, d, g
                idx = [int(t.split("/")[0]) - 1 for t in p[:3]]
                faces.append(idx)
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


# -----------------------------------------------------------
# 2. 采样
# -----------------------------------------------------------
def sample_surface(verts: np.ndarray, faces: np.ndarray,
                   n_samples: int, seed: int = 0) -> np.ndarray:
    """按面积加权在三角网格表面均匀采样"""
    rng = np.random.default_rng(seed)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    p = areas / areas.sum()
    face_idx = rng.choice(len(faces), size=n_samples, p=p)
    u = rng.random(n_samples)
    v = rng.random(n_samples)
    flip = (u + v) > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    w = 1 - u - v
    pts = (u[:, None] * v0[face_idx] +
           v[:, None] * v1[face_idx] +
           w[:, None] * v2[face_idx])
    return pts.astype(np.float32)


# -----------------------------------------------------------
# 3. 归一化（中心化 + 单位球缩放）
# -----------------------------------------------------------
def normalize(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """返回 (归一化点云, center, scale)"""
    center = pts.mean(0)
    pts0 = pts - center
    scale = float(np.linalg.norm(pts0, axis=1).max())
    return (pts0 / scale).astype(np.float32), center, scale


# -----------------------------------------------------------
# 4. 导出
# -----------------------------------------------------------
def save_npy(pts: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, pts)


# -----------------------------------------------------------
# CLI: obj -> sampled point cloud (.npy)
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", help="输入 .obj 文件")
    parser.add_argument("--n", type=int, default=20000, help="采样点数")
    parser.add_argument("--out", default=None)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    v, f = load_obj_full(args.obj)
    print(f"[mesh] verts={len(v)}, faces={len(f)}")
    if len(f) == 0:
        pts = v
    else:
        pts = sample_surface(v, f, args.n)
    if args.normalize:
        pts, c, s = normalize(pts)
        print(f"[mesh] normalized: center={c}, scale={s:.4f}")

    out = args.out or os.path.splitext(args.obj)[0] + f"_pts{args.n}.npy"
    save_npy(pts, out)
    print(f"[mesh] 已保存: {out}  shape={pts.shape}")
