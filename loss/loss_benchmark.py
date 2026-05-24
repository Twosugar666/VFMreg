"""
损失函数性能基准测试
=========================
对所有损失函数进行 wall-clock 时间和显存占用基准测试。
帮助开发者在大批量训练前选择合适的损失函数组合。

运行：
    python loss/loss_benchmark.py --device cuda
    python loss/loss_benchmark.py --device cpu

输出：
    loss/output/benchmark.json       # 详细 JSON
    loss/output/benchmark_table.txt  # 终端友好的表格
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict

# 让脚本可独立运行
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import torch


def measure(fn: Callable, n_warmup: int = 5, n_runs: int = 50) -> Dict:
    """精确测量函数执行时间和显存"""
    # Warmup
    for _ in range(n_warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024 ** 2

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
        mem_used = peak_mem - mem_before
    else:
        mem_used = -1.0

    arr = torch.tensor(times)
    return {
        "mean_ms": arr.mean().item(),
        "std_ms": arr.std().item(),
        "min_ms": arr.min().item(),
        "max_ms": arr.max().item(),
        "median_ms": arr.median().item(),
        "peak_mem_mb": mem_used,
        "n_runs": n_runs,
    }


def benchmark_segmentation(device: str) -> Dict:
    """第3章：分割损失"""
    from loss import (
        DiceLoss, FocalLoss, TverskyLoss, BoundaryLoss,
        SobelEdgeLoss, MultiScaleCELoss, ComboSegLoss,
    )

    B, C, H, W = 4, 2, 256, 256
    pred = torch.randn(B, C, H, W, device=device, requires_grad=True)
    target = torch.randint(0, C, (B, H, W), device=device)

    preds_ms = [torch.randn(B, C, H // (2 ** i), W // (2 ** i),
                            device=device, requires_grad=True) for i in range(3)]

    cases = {
        "DiceLoss": DiceLoss().to(device),
        "FocalLoss": FocalLoss().to(device),
        "TverskyLoss": TverskyLoss().to(device),
        "BoundaryLoss": BoundaryLoss().to(device),
        "SobelEdgeLoss": SobelEdgeLoss().to(device),
    }

    results = {}
    for name, loss_fn in cases.items():
        results[name] = measure(lambda fn=loss_fn: fn(pred, target).item())

    # 多尺度
    ms = MultiScaleCELoss().to(device)
    results["MultiScaleCELoss"] = measure(
        lambda: ms(preds_ms, target)[0].item()
    )

    # 复合
    combo = ComboSegLoss().to(device)
    results["ComboSegLoss"] = measure(
        lambda: combo(preds_ms, pred, target)[0].item()
    )

    return results


def benchmark_nerf(device: str) -> Dict:
    """第4章：NeRF 损失"""
    from loss import (
        PhotoLoss, DensityLoss, TotalVariationLoss,
        DepthConsistencyLoss, NeRFRegLoss,
    )

    B = 4
    img1 = torch.rand(B, 3, 128, 128, device=device, requires_grad=True)
    img2 = torch.rand(B, 3, 128, 128, device=device)
    sigma = torch.rand(1024, device=device, requires_grad=True) + 0.1
    depths = torch.rand(4, 128, 128, device=device)

    cases = {
        "PhotoLoss(L1)": (PhotoLoss("l1").to(device), lambda fn: fn(img1, img2)),
        "PhotoLoss(MSE)": (PhotoLoss("mse").to(device), lambda fn: fn(img1, img2)),
        "DensityLoss": (DensityLoss().to(device), lambda fn: fn(sigma)),
        "TotalVariationLoss": (TotalVariationLoss().to(device), lambda fn: fn(img1)),
        "DepthConsistency": (DepthConsistencyLoss().to(device), lambda fn: fn(depths)),
    }

    results = {}
    for name, (loss_fn, call) in cases.items():
        results[name] = measure(lambda fn=loss_fn, c=call: c(fn).item())

    nerf_reg = NeRFRegLoss(use_lpips=False).to(device)
    results["NeRFRegLoss"] = measure(
        lambda: nerf_reg(img1, img2, sigma_surface=sigma)[0].item()
    )

    return results


def benchmark_pose(device: str) -> Dict:
    """第5章：姿态损失"""
    from loss import (
        TranslationLoss, GeodesicLoss, ChordalLoss,
        QuaternionLoss, Rotation6DLoss, AnglePenaltyLoss,
        VFMRegPoseLoss, rotation_6d_to_matrix,
    )

    B = 32
    d6 = torch.randn(B, 6, device=device, requires_grad=True)
    t_pred = torch.randn(B, 3, device=device, requires_grad=True)
    R_gt = torch.eye(3, device=device).expand(B, 3, 3).contiguous()
    t_gt = torch.randn(B, 3, device=device)
    q_pred = torch.randn(B, 4, device=device, requires_grad=True)
    q_gt = torch.randn(B, 4, device=device)
    R_pred = rotation_6d_to_matrix(d6).detach().requires_grad_(True)

    cases = {
        "TranslationLoss": (TranslationLoss().to(device), lambda fn: fn(t_pred, t_gt)),
        "GeodesicLoss": (GeodesicLoss().to(device), lambda fn: fn(R_pred, R_gt)),
        "ChordalLoss": (ChordalLoss().to(device), lambda fn: fn(R_pred, R_gt)),
        "QuaternionLoss": (QuaternionLoss().to(device), lambda fn: fn(q_pred, q_gt)),
        "Rotation6DLoss": (Rotation6DLoss().to(device), lambda fn: fn(d6, R_gt)),
        "AnglePenaltyLoss": (AnglePenaltyLoss().to(device), lambda fn: fn(R_pred, R_gt)),
    }

    results = {}
    for name, (loss_fn, call) in cases.items():
        results[name] = measure(lambda fn=loss_fn, c=call: c(fn).item())

    full = VFMRegPoseLoss().to(device)
    results["VFMRegPoseLoss"] = measure(
        lambda: full(d6, t_pred, R_gt, t_gt)[0].item()
    )

    return results


def benchmark_render(device: str) -> Dict:
    """第5章：渲染损失"""
    from loss import (
        DifferentiableIoULoss, SilhouetteL1Loss,
        MaskedRGBLoss, MultiViewConsistencyLoss, VFMRegRenderLoss,
    )

    B = 4
    mask_p = torch.sigmoid(torch.randn(B, 1, 128, 128, device=device, requires_grad=True))
    mask_t = torch.randint(0, 2, (B, 1, 128, 128), device=device).float()
    rgb_p = torch.sigmoid(torch.randn(B, 3, 128, 128, device=device, requires_grad=True))
    rgb_t = torch.rand(B, 3, 128, 128, device=device)
    d6_v = torch.randn(B, 4, 6, device=device, requires_grad=True)
    t_v = torch.randn(B, 4, 3, device=device, requires_grad=True)

    cases = {
        "DifferentiableIoULoss": (DifferentiableIoULoss().to(device),
                                  lambda fn: fn(mask_p, mask_t)),
        "SilhouetteL1Loss":      (SilhouetteL1Loss().to(device),
                                  lambda fn: fn(mask_p, mask_t)),
        "MaskedRGBLoss":         (MaskedRGBLoss().to(device),
                                  lambda fn: fn(rgb_p, rgb_t, mask_t)),
        "MultiViewConsistency":  (MultiViewConsistencyLoss().to(device),
                                  lambda fn: fn(d6_v, t_v)),
    }

    results = {}
    for name, (loss_fn, call) in cases.items():
        results[name] = measure(lambda fn=loss_fn, c=call: c(fn).item())

    full = VFMRegRenderLoss().to(device)
    results["VFMRegRenderLoss"] = measure(
        lambda: full(mask_p, rgb_p, mask_t, rgb_t)[0].item()
    )

    return results


def format_table(results: Dict[str, Dict[str, Dict]]) -> str:
    """格式化为终端表格"""
    lines = []
    header = f"{'Loss Function':<28s} {'Mean (ms)':>10s} {'Std':>8s} {'Median':>8s} {'Mem (MB)':>10s}"
    sep = "─" * len(header)

    for category, items in results.items():
        lines.append("")
        lines.append(f"┌{'─' * (len(header) - 2)}┐")
        lines.append(f"│ {category:<{len(header) - 4}s} │")
        lines.append(f"└{'─' * (len(header) - 2)}┘")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for name, stats in items.items():
            mem_str = f"{stats['peak_mem_mb']:>8.2f}" if stats['peak_mem_mb'] >= 0 else "    N/A"
            lines.append(
                f"{name:<28s} "
                f"{stats['mean_ms']:>10.4f} "
                f"{stats['std_ms']:>8.4f} "
                f"{stats['median_ms']:>8.4f} "
                f"{mem_str:>10s}"
            )
        lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="测试设备")
    parser.add_argument("--n_runs", type=int, default=50)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    print("=" * 70)
    print(f"🚀 Loss 函数性能基准测试 (device={args.device}, n_runs={args.n_runs})")
    print("=" * 70)

    all_results = {}

    print("\n[1/4] 第3章 分割损失...")
    all_results["Ch.3 Segmentation Losses"] = benchmark_segmentation(args.device)
    gc.collect()

    print("[2/4] 第4章 NeRF 损失...")
    all_results["Ch.4 NeRF Losses"] = benchmark_nerf(args.device)
    gc.collect()

    print("[3/4] 第5章 姿态损失...")
    all_results["Ch.5 Pose Losses"] = benchmark_pose(args.device)
    gc.collect()

    print("[4/4] 第5章 渲染损失...")
    all_results["Ch.5 Render Losses"] = benchmark_render(args.device)
    gc.collect()

    # 格式化输出
    table = format_table(all_results)
    print(table)

    # 保存
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True, parents=True)

    json_path = out_dir / f"benchmark_{args.device}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "device": args.device,
            "n_runs": args.n_runs,
            "torch_version": torch.__version__,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    table_path = out_dir / f"benchmark_{args.device}_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)

    print()
    print(f"✅ 详细数据已保存至: {json_path}")
    print(f"📊 表格已保存至: {table_path}")


if __name__ == "__main__":
    main()
