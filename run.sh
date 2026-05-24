#!/usr/bin/env bash
# ===================================================================
# 脑磁配准系统 一键脚本（answers.sh / Makefile 替代品）
# ===================================================================
# 用法：
#   bash run.sh test       # 跑单元测试
#   bash run.sh eval       # 跑论文指标对齐评估
#   bash run.sh report     # 生成数据集摘要报告
#   bash run.sh demo-icp   # 跑 ICP baseline 自检
#   bash run.sh visreg     # 生成配准可视化图
#   bash run.sh all        # 依次跑全部
# ===================================================================
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

cmd=${1:-help}

case "$cmd" in

  test)
    echo "[run.sh] 端到端测试..."
    python -m unittest discover -s tests -v
    echo "[run.sh] loss 单元测试..."
    python -m unittest discover -s loss/tests -v
    ;;

  eval)
    echo "[run.sh] 论文指标对齐..."
    python -m evaluate.run_eval --tol 0.10
    ;;

  report)
    echo "[run.sh] 数据集摘要报告..."
    python demo_reports/run_all.py
    ;;

  demo-icp)
    echo "[run.sh] ICP baseline 自检..."
    python tools/icp_baseline.py --demo
    ;;

  visreg)
    echo "[run.sh] 配准可视化..."
    python tools/visualize_registration.py
    ;;

  figures)
    echo "[run.sh] 生成 8 张实验效果图..."
    python figure/generate_figures.py
    ;;

  configs)
    echo "[run.sh] 列出 ablation 变体..."
    python configs/loader.py configs/ablation_vfmreg.yaml
    ;;

  all)
    bash "$0" test
    bash "$0" demo-icp
    bash "$0" eval
    bash "$0" visreg
    bash "$0" figures
    ;;

  help|*)
    grep -E "^# +" run.sh | head -20
    ;;
esac
