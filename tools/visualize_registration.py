"""
tools/visualize_registration.py
================================
配准结果可视化：
- 把预测 vs GT 的位姿用 3D 散点 + 矢量图绘出
- 误差散布图（trans_err vs rot_err，含成功阈值标注）
- 多视角覆盖图（matplotlib + ax.view_init）

使用 results/ch5_vfmreg_results.json 作为输入即可生成答辩用图。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io import load_json, ensure_dir, get_logger     # noqa: E402

LOG = get_logger("vis_reg")


# -----------------------------------------------------------
# 1. 误差散点
# -----------------------------------------------------------
def plot_error_scatter(per_sample: List[Dict], out_path: str,
                       trans_thr: float = 2.0, rot_thr: float = 2.0):
    t = np.array([s.get("trans_err_mm", s.get("trans_mm", 0)) for s in per_sample])
    r = np.array([s.get("rot_err_deg", s.get("rot_deg", 0)) for s in per_sample])
    success = (t <= trans_thr) & (r <= rot_thr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t[success], r[success], c="#2e7d32", s=24,
               label=f"Success ({success.sum()})", alpha=0.8)
    ax.scatter(t[~success], r[~success], c="#c62828", s=24,
               label=f"Fail ({(~success).sum()})", alpha=0.8, marker="x")
    ax.axvline(trans_thr, color="gray", linestyle="--", lw=1)
    ax.axhline(rot_thr, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Translation Error (mm)")
    ax.set_ylabel("Rotation Error (deg)")
    ax.set_title(f"VFMReg per-sample errors (n={len(t)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info(f"已保存: {out_path}")


# -----------------------------------------------------------
# 2. 累计误差分布 (CDF)
# -----------------------------------------------------------
def plot_cdf(per_sample: List[Dict], out_path: str):
    t = np.sort([s.get("trans_err_mm", 0) for s in per_sample])
    r = np.sort([s.get("rot_err_deg", 0) for s in per_sample])
    n = len(t)
    if n == 0:
        return
    cdf = np.arange(1, n + 1) / n

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(t, cdf, color="#1565c0", lw=2)
    axs[0].set_xlabel("Translation Error (mm)")
    axs[0].set_ylabel("Cumulative %")
    axs[0].set_title("Translation Error CDF")
    axs[0].grid(alpha=0.3)

    axs[1].plot(r, cdf, color="#c62828", lw=2)
    axs[1].set_xlabel("Rotation Error (deg)")
    axs[1].set_ylabel("Cumulative %")
    axs[1].set_title("Rotation Error CDF")
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info(f"已保存: {out_path}")


# -----------------------------------------------------------
# 3. 3D 位姿轨迹（pred vs gt）
# -----------------------------------------------------------
def plot_pose_3d(per_sample: List[Dict], out_path: str, max_n: int = 50):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for s in per_sample[:max_n]:
        pred = s.get("pred_translation") or s.get("translation_pred")
        gt = s.get("gt_translation") or s.get("translation_gt")
        if pred is None or gt is None:
            continue
        pred, gt = np.asarray(pred), np.asarray(gt)
        ax.scatter(*pred, c="#1565c0", s=18, alpha=0.7)
        ax.scatter(*gt, c="#2e7d32", s=18, alpha=0.7)
        ax.plot([pred[0], gt[0]], [pred[1], gt[1]], [pred[2], gt[2]],
                color="gray", alpha=0.4, lw=0.6)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Pred (blue) vs GT (green) translation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info(f"已保存: {out_path}")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "results", "ch5_vfmreg_results.json")))
    parser.add_argument("--out_dir", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "results", "vis_reg")))
    args = parser.parse_args()

    data = load_json(args.results)
    per_sample = (data.get("per_sample")
                  or data.get("samples")
                  or data.get("real", {}).get("per_sample")
                  or [])
    if not per_sample:
        LOG.warning("results 中未找到 per_sample 字段，"
                    "请确认 ch5_vfmreg_results.json 的结构")
        return

    ensure_dir(args.out_dir)
    plot_error_scatter(per_sample, os.path.join(args.out_dir, "error_scatter.png"))
    plot_cdf(per_sample, os.path.join(args.out_dir, "error_cdf.png"))
    plot_pose_3d(per_sample, os.path.join(args.out_dir, "pose_3d.png"))
    LOG.info(f"全部图已生成至 {args.out_dir}")


if __name__ == "__main__":
    main()
