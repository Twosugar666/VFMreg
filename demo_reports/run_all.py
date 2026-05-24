"""
================================================================
一键运行: 依次执行 01 → 02 → 03 → 04 生成完整数据集报告
================================================================
用法:
    python demo_reports/run_all.py
然后打开:
    demo_reports/output/04_final_report/index.html
================================================================
"""

import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

SCRIPTS = [
    ('01_dataset_overview.py',     '📊 数据集总览统计'),
    ('02_sample_visualization.py', '🖼️ 样本可视化网格'),
    ('03_data_distribution.py',    '📈 数据分布分析'),
    ('04_html_report.py',          '📄 生成HTML报告'),
]


def run(script):
    py = sys.executable
    cmd = [py, str(ROOT / script)]
    return subprocess.run(cmd, capture_output=False)


def main():
    print('\n' + '=' * 70)
    print('  🎬  训练数据集报告 — 一键生成')
    print('=' * 70)

    success_count = 0
    total_t = 0
    for script, desc in SCRIPTS:
        print(f'\n\n{"━" * 70}')
        print(f'  ▶ 步骤: {desc}  ({script})')
        print(f'{"━" * 70}')
        t0 = time.time()
        try:
            r = run(script)
            dt = time.time() - t0
            total_t += dt
            if r.returncode == 0:
                print(f'\n  ✅ 完成 (耗时 {dt:.1f}s)')
                success_count += 1
            else:
                print(f'\n  ❌ 失败 (返回码 {r.returncode})')
        except Exception as e:
            print(f'\n  ❌ 异常: {e}')

    print('\n' + '=' * 70)
    print(f'  🎉 全部完成: {success_count}/{len(SCRIPTS)} 步成功，'
          f'总耗时 {total_t:.1f}s')
    print('=' * 70)
    report = ROOT / 'output/04_final_report/index.html'
    if report.exists():
        print(f'\n  📄 打开报告:')
        print(f'     file://{report.resolve()}')
        print()


if __name__ == '__main__':
    main()
