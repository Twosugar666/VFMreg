"""
================================================================
脚本 01: 数据集总览统计与可视化
================================================================
功能:
    扫描 ./data/ 目录下所有训练数据集，统计：
    - 每个数据集的样本量（parquet 行数 / zip 内文件数）
    - 数据集总体积
    - 用途分类（头部分割 / 背景纹理 / NeRF多视角）
输出:
    ./demo_reports/output/01_overview/
        ├── dataset_size_chart.png     (体积柱状图)
        ├── dataset_count_chart.png    (样本量柱状图)
        ├── dataset_pie.png            (用途占比饼图)
        └── overview_stats.json        (JSON 统计数据)
用法:
    python demo_reports/01_dataset_overview.py
================================================================
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ----------------------- 配置 -----------------------
DATA_ROOT = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/data')
OUTPUT_DIR = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/demo_reports/output/01_overview')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集元信息（与论文章节对应）
DATASETS = [
    {'name': 'human_parsing',         'path': 'head_seg/human_parsing',        'category': '头部分割（第3章）',     'paper_chapter': 'Ch.3', 'color': '#3498db'},
    {'name': 'celebA_faces',          'path': 'head_seg/celebA_faces',         'category': '头部分割（第3章）',     'paper_chapter': 'Ch.3', 'color': '#5dade2'},
    {'name': 'cppe5',                 'path': 'head_seg/cppe5',                'category': '头部分割（第3章）',     'paper_chapter': 'Ch.3', 'color': '#85c1e9'},
    {'name': 'protective_equipment',  'path': 'head_seg/protective_equipment', 'category': '头盔分割（第3章）',     'paper_chapter': 'Ch.3', 'color': '#1f618d'},
    {'name': 'afhqv2',                'path': 'head_seg/afhqv2',               'category': '域外测试（第3章）',     'paper_chapter': 'Ch.3', 'color': '#a9cce3'},
    {'name': 'dog_food_bg',           'path': 'hdri/dog_food_bg',              'category': '背景域随机化（第5章）', 'paper_chapter': 'Ch.5', 'color': '#27ae60'},
    {'name': 'wikiart_bg',            'path': 'hdri/wikiart_bg',               'category': '背景域随机化（第5章）', 'paper_chapter': 'Ch.5', 'color': '#52be80'},
    {'name': 'aloha_multiview',       'path': 'nerf_test/aloha_multiview',     'category': 'NeRF多视角（第4章）',   'paper_chapter': 'Ch.4', 'color': '#e67e22'},
]


def get_dir_size(path: Path) -> int:
    """递归计算目录总大小（字节）"""
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def count_samples(ds_path: Path) -> int:
    """估算数据集样本量"""
    if not ds_path.exists():
        return 0
    n = 0
    # 1) 先看 parquet 文件
    for pq in ds_path.rglob('*.parquet'):
        try:
            # 只读 metadata, 不加载数据
            df = pd.read_parquet(pq, columns=[pd.read_parquet(pq, engine='pyarrow').columns[0]])
            n += len(df)
        except Exception:
            try:
                import pyarrow.parquet as papq
                n += papq.ParquetFile(pq).metadata.num_rows
            except Exception:
                pass
    # 2) 再看 zip（如 protective_equipment）
    for zf in ds_path.rglob('*.zip'):
        try:
            with zipfile.ZipFile(zf) as z:
                n += sum(1 for nm in z.namelist() if nm.lower().endswith(('.jpg', '.jpeg', '.png')))
        except Exception:
            pass
    # 3) 散图
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        n += len(list(ds_path.rglob(ext)))
    return n


def fmt_size(bytes_: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if bytes_ < 1024:
            return f'{bytes_:.1f} {unit}'
        bytes_ /= 1024
    return f'{bytes_:.1f} TB'


def collect_stats():
    """扫描收集所有数据集统计"""
    print('=' * 70)
    print(f'  数据集总览扫描  ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
    print('=' * 70)
    stats = []
    for ds in DATASETS:
        full_path = DATA_ROOT / ds['path']
        size = get_dir_size(full_path)
        count = count_samples(full_path) if size > 0 else 0
        exists = full_path.exists() and size > 1024
        stats.append({
            **ds,
            'size_bytes': size,
            'size_human': fmt_size(size),
            'sample_count': count,
            'exists': exists,
        })
        status = '✅' if exists else '⏳'
        print(f'{status} {ds["name"]:25s}  {fmt_size(size):>10s}   {count:>8d} 样本   [{ds["category"]}]')
    print('-' * 70)
    total_size = sum(s['size_bytes'] for s in stats)
    total_count = sum(s['sample_count'] for s in stats)
    print(f'   合计: {fmt_size(total_size)}   {total_count} 样本')
    print('=' * 70)
    return stats


# ----------------------- 可视化 -----------------------

def plot_size_bar(stats, save_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    sizes_mb = [s['size_bytes'] / 1024 / 1024 for s in stats]
    names = [s['name'] for s in stats]
    colors = [s['color'] for s in stats]

    bars = ax.bar(names, sizes_mb, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Dataset Size (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Training Datasets - Disk Size Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=25, ha='right')

    # 标注数值
    for bar, s in zip(bars, stats):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(sizes_mb) * 0.01,
                s['size_human'], ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 保存: {save_path}')


def plot_count_bar(stats, save_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    counts = [s['sample_count'] for s in stats]
    names = [s['name'] for s in stats]
    colors = [s['color'] for s in stats]

    bars = ax.bar(names, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
    ax.set_title('Training Datasets - Sample Count Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=25, ha='right')
    if max(counts) > 0:
        ax.set_yscale('log')
        ax.set_ylabel('Sample Count (log scale)', fontsize=12, fontweight='bold')

    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, c,
                    f'{c:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 保存: {save_path}')


def plot_category_pie(stats, save_path):
    """按论文章节聚合的占比饼图"""
    by_chapter = {}
    for s in stats:
        if s['size_bytes'] == 0:
            continue
        ch = s['paper_chapter']
        by_chapter[ch] = by_chapter.get(ch, 0) + s['size_bytes']

    if not by_chapter:
        print('  ⚠ 无数据可绘制饼图')
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = list(by_chapter.keys())
    sizes = list(by_chapter.values())
    colors_pie = ['#3498db', '#e67e22', '#27ae60', '#9b59b6'][:len(labels)]

    def fmt_pct(pct):
        absolute = int(pct / 100. * sum(sizes))
        return f'{pct:.1f}%\n({fmt_size(absolute)})'

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie, autopct=fmt_pct,
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11, 'fontweight': 'bold'},
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10)

    ax.set_title('Datasets Distribution by Paper Chapter',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 保存: {save_path}')


def main():
    stats = collect_stats()

    print('\n📊 生成可视化图表...')
    plot_size_bar(stats, OUTPUT_DIR / 'dataset_size_chart.png')
    plot_count_bar(stats, OUTPUT_DIR / 'dataset_count_chart.png')
    plot_category_pie(stats, OUTPUT_DIR / 'dataset_pie.png')

    # 保存JSON
    json_path = OUTPUT_DIR / 'overview_stats.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'data_root': str(DATA_ROOT),
            'datasets': stats,
            'summary': {
                'total_size_bytes': sum(s['size_bytes'] for s in stats),
                'total_size_human': fmt_size(sum(s['size_bytes'] for s in stats)),
                'total_samples': sum(s['sample_count'] for s in stats),
                'available_count': sum(1 for s in stats if s['exists']),
                'total_count': len(stats),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f'  ✓ 保存: {json_path}')

    print(f'\n✅ 数据集总览报告已生成至: {OUTPUT_DIR}\n')


if __name__ == '__main__':
    main()
