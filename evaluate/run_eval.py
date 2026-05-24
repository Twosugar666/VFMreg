"""
evaluate/run_eval.py
====================
顶层评估入口：调用 metrics + paper_alignment，给出本项目所有章节的统一评估摘要。

用法：
    python -m evaluate.run_eval                  # 用默认 results/ 目录
    python -m evaluate.run_eval --results <dir>  # 自定义
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluate.paper_alignment import align, print_report   # noqa: E402
from utils.io import save_json, ensure_dir, get_logger     # noqa: E402

LOG = get_logger("eval")


def main():
    parser = argparse.ArgumentParser(description="论文指标统一评估入口")
    parser.add_argument("--results", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results")))
    parser.add_argument("--out", default=None)
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()

    LOG.info(f"读取结果目录: {args.results}")
    report = align(args.results, tol=args.tol)
    report["meta"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tolerance": args.tol,
        "results_dir": args.results,
    }
    print_report(report)

    out = args.out or os.path.join(args.results, "alignment_report.json")
    ensure_dir(os.path.dirname(out))
    save_json(report, out)
    LOG.info(f"报告写入: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
