"""
evaluate/metrics.py
===================
评估指标实现：
- 第3章：mIoU / Boundary-F1 / Pixel-Acc
- 第4章 / 第5章：trans_mm / rot_deg / 成功率 (success rate)
- 通用：AUC、ECE 校准误差
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ===================================================================
# 分割指标
# ===================================================================
def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """像素级混淆矩阵 [C, C]"""
    mask = (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
    bc = np.bincount(idx, minlength=num_classes * num_classes)
    return bc.reshape(num_classes, num_classes)


def miou_from_cm(cm: np.ndarray) -> float:
    intersection = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    return float(iou.mean())


def pixel_acc_from_cm(cm: np.ndarray) -> float:
    return float(np.diag(cm).sum() / max(cm.sum(), 1))


def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tol_px: int = 2) -> float:
    """Boundary F1 (BF1)：取轮廓后做点对点匹配（容差 tol_px）"""
    import cv2
    pred_edge = cv2.Canny(pred_mask.astype(np.uint8) * 255, 100, 200)
    gt_edge = cv2.Canny(gt_mask.astype(np.uint8) * 255, 100, 200)
    if pred_edge.sum() == 0 or gt_edge.sum() == 0:
        return 0.0

    # 距离变换实现快速容差匹配
    dt_pred = cv2.distanceTransform(255 - pred_edge, cv2.DIST_L2, 3)
    dt_gt = cv2.distanceTransform(255 - gt_edge, cv2.DIST_L2, 3)

    precision = ((dt_gt[pred_edge > 0] <= tol_px).sum()
                 / max((pred_edge > 0).sum(), 1))
    recall = ((dt_pred[gt_edge > 0] <= tol_px).sum()
              / max((gt_edge > 0).sum(), 1))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# ===================================================================
# 配准指标
# ===================================================================
def aggregate_pose_errors(
    trans_errs: Sequence[float],
    rot_errs: Sequence[float],
    success_trans_mm: float = 2.0,
    success_rot_deg: float = 2.0,
) -> Dict[str, float]:
    """汇总平移/旋转误差，给出均值/中位/标准差/成功率"""
    if not trans_errs:
        return {}
    t = np.asarray(trans_errs, dtype=np.float64)
    r = np.asarray(rot_errs, dtype=np.float64)
    success = ((t <= success_trans_mm) & (r <= success_rot_deg)).mean()
    return {
        "trans_mean_mm":   float(t.mean()),
        "trans_median_mm": float(np.median(t)),
        "trans_std_mm":    float(t.std()),
        "rot_mean_deg":    float(r.mean()),
        "rot_median_deg":  float(np.median(r)),
        "rot_std_deg":     float(r.std()),
        "success_rate":    float(success),
        "n_samples":       int(len(t)),
    }


# ===================================================================
# 通用
# ===================================================================
def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """ECE：模型置信度与实际正确率的偏差"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = confidences[m].mean()
        ece += (m.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """快速 AUROC（避免依赖 sklearn）"""
    order = np.argsort(-scores)
    labels = labels[order]
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 0.5
    cum_pos = np.cumsum(labels == 1)
    tp = cum_pos
    fp = np.cumsum(labels == 0)
    tpr = tp / pos
    fpr = fp / neg
    return float(np.trapz(tpr, fpr))
