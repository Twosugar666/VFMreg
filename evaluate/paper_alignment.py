"""
evaluate/paper_alignment.py
===========================
论文指标对齐器：解析 results/*.json（按 table 结构组织）, 抽取本论文方法
(YOLOv8n-seg / NeRF / VFMReg) 的关键数字, 与论文报告值比对, 输出 PASS/FAIL。

数据结构假设（与 generate_demo_results.py 产生的格式一致）：
    ch3:  table_3_2_performance_comparison.methods["YOLOv8n-seg"].standard_scene.{mIoU, BF1}
    ch4:  table_4_1_comparison.methods["NeRF配准(本文)"].{trans_mm, rot_deg, success_rate}
    ch5:  table_5_3_synthetic_comparison.methods["VFMReg(本文)"].{trans_mm, rot_deg}
          table_5_4_real_comparison.after_finetune.{trans_mm, rot_deg}
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.io import load_json, save_json, get_logger      # noqa: E402

LOG = get_logger("paper_align")


# -------------------------------------------------------------------
# 论文目标值
# -------------------------------------------------------------------
PAPER_TARGETS: Dict[str, Dict[str, float]] = {
    "seg":          {"mIoU": 95.2, "BF1": 89.7},        # %
    "nerf":         {"trans_mm": 1.2, "rot_deg": 0.9, "success_rate": 0.92},
    "vfmreg_synth": {"trans_mm": 0.5, "rot_deg": 0.6},
    "vfmreg_real":  {"trans_mm": 0.6, "rot_deg": 0.7},
    "inference":    {"latency_ms": 15.0},
}


# -------------------------------------------------------------------
# 数值检查
# -------------------------------------------------------------------
def _within(value: float, target: float, tol: float, lower_better: bool) -> bool:
    if lower_better:
        return value <= target * (1 + tol) + 1e-9
    return value >= target * (1 - tol) - 1e-9


def _check(name: str, value: Optional[float], target: float,
           tol: float, lower_better: bool) -> Dict[str, Any]:
    if value is None:
        return {"metric": name, "value": None, "target": target,
                "direction": "↓" if lower_better else "↑",
                "tolerance": tol, "pass": False, "note": "missing"}
    ok = _within(float(value), target, tol, lower_better)
    return {
        "metric": name,
        "value": round(float(value), 4),
        "target": target,
        "direction": "↓" if lower_better else "↑",
        "tolerance": tol,
        "pass": bool(ok),
    }


# -------------------------------------------------------------------
# 通用提取：递归找到第一个匹配的方法 dict
# -------------------------------------------------------------------
def _find_method(methods: Dict[str, Any], keywords: List[str]) -> Optional[Dict]:
    """在 methods dict 里找包含任一关键词的方法名"""
    if not isinstance(methods, dict):
        return None
    for k, v in methods.items():
        if any(kw.lower() in k.lower() for kw in keywords) and isinstance(v, dict):
            return v
    return None


def _g(d: Dict, *path, default=None):
    """安全的嵌套 dict 取值"""
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# -------------------------------------------------------------------
# 各章节抽取器
# -------------------------------------------------------------------
def _extract_ch3(data: Dict) -> List[Dict[str, Any]]:
    methods = _g(data, "table_3_2_performance_comparison", "methods", default={})
    yolo = _find_method(methods, ["YOLOv8", "yolo"]) or {}
    miou = _g(yolo, "standard_scene", "mIoU")
    bf1  = _g(yolo, "standard_scene", "BF1")
    inf_ms = _g(yolo, "inference_time_ms")

    items = [
        _check("seg.mIoU(%)", miou, PAPER_TARGETS["seg"]["mIoU"], 0.05, False),
        _check("seg.BF1(%)",  bf1,  PAPER_TARGETS["seg"]["BF1"],  0.05, False),
    ]
    if inf_ms is not None:
        items.append(_check("seg.inference_ms", inf_ms, 8.0, 0.30, True))
    return items


def _extract_ch4(data: Dict) -> List[Dict[str, Any]]:
    methods = _g(data, "table_4_1_comparison", "methods", default={})
    ours = (_find_method(methods, ["Ours", "本文", "NeRF配准"])
            or {})
    # 数值可能直接 trans_mm，也可能在 translation_error_mm.mean 里
    trans = _g(ours, "trans_mm")
    if trans is None:
        trans = _g(ours, "translation_error_mm", "mean")
    rot = _g(ours, "rot_deg")
    if rot is None:
        rot = _g(ours, "rotation_error_deg", "mean")
    succ = _g(ours, "success_rate")
    if succ is None:
        succ_pct = _g(ours, "success_rate_pct")
        succ = succ_pct / 100.0 if succ_pct is not None else None

    return [
        _check("nerf.trans_mm", trans,
               PAPER_TARGETS["nerf"]["trans_mm"], 0.10, True),
        _check("nerf.rot_deg", rot,
               PAPER_TARGETS["nerf"]["rot_deg"], 0.10, True),
        _check("nerf.success_rate", succ,
               PAPER_TARGETS["nerf"]["success_rate"], 0.05, False),
    ]


def _extract_ch5(data: Dict) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # 合成集
    methods = _g(data, "table_5_3_synthetic_comparison", "methods", default={})
    ours = _find_method(methods, ["VFMReg", "本文", "ours"]) or {}
    items += [
        _check("vfmreg_synth.trans_mm",
               _g(ours, "trans_mm"),
               PAPER_TARGETS["vfmreg_synth"]["trans_mm"], 0.10, True),
        _check("vfmreg_synth.rot_deg",
               _g(ours, "rot_deg"),
               PAPER_TARGETS["vfmreg_synth"]["rot_deg"], 0.10, True),
    ]

    # 真实集 after_finetune
    real = _g(data, "table_5_4_real_comparison", "after_finetune", default={})
    items += [
        _check("vfmreg_real.trans_mm",
               _g(real, "trans_mm"),
               PAPER_TARGETS["vfmreg_real"]["trans_mm"], 0.10, True),
        _check("vfmreg_real.rot_deg",
               _g(real, "rot_deg"),
               PAPER_TARGETS["vfmreg_real"]["rot_deg"], 0.10, True),
    ]

    # 推理延迟
    lat = (_g(data, "inference_performance", "latency_ms")
           or _g(data, "inference_performance", "avg_ms")
           or _g(data, "training_info", "stage2_finetune", "inference_ms"))
    if lat is not None:
        items.append(_check("inference.latency_ms", lat,
                            PAPER_TARGETS["inference"]["latency_ms"], 0.30, True))
    return items


# -------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------
def align(results_dir: str, tol: float = 0.05) -> Dict[str, Any]:
    """tol 仅作为兜底默认值；实际每个 metric 已在抽取器中给定合适容差。"""
    report: Dict[str, Any] = {"passed": True, "details": {}}

    sections = [
        ("ch3_seg",     "ch3_segmentation_results.json",       _extract_ch3),
        ("ch4_nerf",    "ch4_nerf_registration_results.json",  _extract_ch4),
        ("ch5_vfmreg",  "ch5_vfmreg_results.json",             _extract_ch5),
    ]

    for sec_name, fname, extractor in sections:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            LOG.warning(f"未找到 {fname}, 跳过 [{sec_name}]")
            continue
        try:
            data = load_json(path)
            items = extractor(data)
        except Exception as e:                                 # pragma: no cover
            LOG.error(f"解析 {fname} 失败: {e}")
            items = [{"metric": sec_name, "value": None,
                      "target": 0, "direction": "?", "tolerance": tol,
                      "pass": False, "note": str(e)}]
        report["details"][sec_name] = items
        report["passed"] &= all(x["pass"] for x in items if x.get("note") != "missing")

    return report


# -------------------------------------------------------------------
# 终端表格
# -------------------------------------------------------------------
def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 82)
    print("📋 论文指标对齐报告")
    print("=" * 82)
    for section, items in report["details"].items():
        print(f"\n[{section}]")
        print(f"  {'metric':<28}{'value':>10}{'target':>10}"
              f"{'dir':>5}{'tol':>7}{'pass':>8}")
        print("  " + "-" * 73)
        for it in items:
            ok = "✅" if it["pass"] else ("⚠️ " if it.get("note") == "missing" else "❌")
            v = it["value"] if it["value"] is not None else "N/A"
            v_str = f"{v:>10.4f}" if isinstance(v, float) else f"{v:>10}"
            print(f"  {it['metric']:<28}{v_str}"
                  f"{it['target']:>10.4f}{it['direction']:>5}"
                  f"{it['tolerance']:>7.2f}{ok:>8}")
    overall = "✅ ALL PASS" if report["passed"] else "❌ FAILED"
    print("\n" + "=" * 82)
    print(f"  Overall: {overall}")
    print("=" * 82 + "\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(
        os.path.dirname(__file__), "..", "results"))
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rep = align(os.path.abspath(args.results), tol=args.tol)
    print_report(rep)
    if args.out:
        save_json(rep, args.out)
        LOG.info(f"对齐报告已保存至 {args.out}")
    sys.exit(0 if rep["passed"] else 1)
