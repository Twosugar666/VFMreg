"""
================================================================
脚本 03: 数据分布分析（图像尺寸 / 颜色 / 类别分布）
================================================================
功能:
    针对采样到的图像，分析其分布特征：
    - 图像分辨率分布（宽×高散点图 + 直方图）
    - 平均亮度分布
    - 每个数据集的颜色直方图（RGB通道）
    - 头部解析mask 的语义类别频率
输出: ./demo_reports/output/03_distribution/
用法: python demo_reports/03_data_distribution.py
================================================================
"""

import io
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_ROOT = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/data')
OUTPUT_DIR = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/demo_reports/output/03_distribution')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# Human Parsing 数据集类别名（含20类）
HUMAN_PARSING_LABELS = {
    0: 'Background', 1: 'Hat', 2: 'Hair', 3: 'Glove', 4: 'Sunglasses',
    5: 'Upper-clothes', 6: 'Dress', 7: 'Coat', 8: 'Socks', 9: 'Pants',
    10: 'Jumpsuits', 11: 'Scarf', 12: 'Skirt', 13: 'Face', 14: 'L-arm',
    15: 'R-arm', 16: 'L-leg', 17: 'R-leg', 18: 'L-shoe', 19: 'R-shoe',
}
HEAD_RELATED = {1, 2, 4, 11, 13}  # Hat, Hair, Sunglasses, Scarf, Face


def sample_parquet(parquet_path, n=200):
    """随机采样 N 张图，返回 PIL.Image 列表"""
    if not parquet_path.exists():
        return [], []
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f'  读取失败: {e}')
        return [], []
    indices = random.sample(range(len(df)), min(n, len(df)))
    images, masks = [], []
    for idx in indices:
        row = df.iloc[idx]
        try:
            img_field = row['image']
            ib = img_field['bytes'] if isinstance(img_field, dict) else img_field
            img = Image.open(io.BytesIO(ib)).convert('RGB')
            images.append(img)
            if 'mask' in df.columns:
                mf = row['mask']
                mb = mf['bytes'] if isinstance(mf, dict) else mf
                masks.append(np.array(Image.open(io.BytesIO(mb))))
        except Exception:
            continue
    return images, masks


def analyze_resolution(datasets):
    """分析各数据集图像分辨率分布"""
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6', '#e74c3c', '#f39c12']

    for i, (name, images) in enumerate(datasets.items()):
        if not images:
            continue
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        ax.scatter(widths, heights, alpha=0.6, s=30,
                   color=colors[i % len(colors)],
                   label=f'{name} (n={len(images)})')

    ax.set_xlabel('Width (px)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (px)', fontsize=12, fontweight='bold')
    ax.set_title('Image Resolution Distribution',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'resolution_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ resolution_scatter.png')


def analyze_brightness(datasets):
    """各数据集平均亮度直方图（用于域随机化分析）"""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#3498db', '#e67e22', '#27ae60', '#9b59b6', '#e74c3c', '#f39c12']

    for i, (name, images) in enumerate(datasets.items()):
        if not images:
            continue
        brightness = [np.mean(np.array(img.convert('L'))) for img in images]
        ax.hist(brightness, bins=30, alpha=0.5,
                color=colors[i % len(colors)], label=f'{name}', edgecolor='black')

    ax.set_xlabel('Mean Brightness (0-255)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Image Count', fontsize=12, fontweight='bold')
    ax.set_title('Image Brightness Distribution (for domain randomization coverage)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'brightness_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ brightness_hist.png')


def analyze_rgb_histogram(datasets):
    """各数据集 RGB 通道颜色直方图"""
    n_ds = len([k for k, v in datasets.items() if v])
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 4, 4.5))
    if n_ds == 1:
        axes = [axes]
    elif n_ds == 0:
        return

    plot_idx = 0
    for name, images in datasets.items():
        if not images:
            continue
        ax = axes[plot_idx]
        plot_idx += 1
        # 收集所有像素
        all_r, all_g, all_b = [], [], []
        for img in images[:50]:  # 节省时间
            arr = np.array(img.resize((64, 64)))
            all_r.append(arr[:, :, 0].flatten())
            all_g.append(arr[:, :, 1].flatten())
            all_b.append(arr[:, :, 2].flatten())
        all_r = np.concatenate(all_r)
        all_g = np.concatenate(all_g)
        all_b = np.concatenate(all_b)

        ax.hist(all_r, bins=32, alpha=0.5, color='red', label='R')
        ax.hist(all_g, bins=32, alpha=0.5, color='green', label='G')
        ax.hist(all_b, bins=32, alpha=0.5, color='blue', label='B')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Pixel value')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('RGB Channel Distribution per Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rgb_histogram.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  ✓ rgb_histogram.png')


def analyze_mask_classes(masks):
    """统计 Human Parsing mask 中各语义类别像素占比"""
    if not masks:
        return

    total_pixels = 0
    class_pixels = Counter()
    for m in masks:
        unique, counts = np.unique(m, return_counts=True)
        for u, c in zip(unique, counts):
            class_pixels[int(u)] += int(c)
        total_pixels += m.size

    # 排序
    sorted_classes = sorted(class_pixels.items(), key=lambda x: -x[1])
    labels = [HUMAN_PARSING_LABELS.get(c, f'class_{c}') for c, _ in sorted_classes]
    pcts = [v / total_pixels * 100 for _, v in sorted_classes]
    colors = ['#e74c3c' if HUMAN_PARSING_LABELS.get(c, '') in ['Hat', 'Hair', 'Sunglasses', 'Scarf', 'Face']
              else '#bdc3c7' for c, _ in sorted_classes]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(labels, pcts, color=colors, edgecolor='black')
    ax.set_ylabel('Pixel Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Human Parsing Class Distribution (red = head-related classes)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # 添加图例
    head_total = sum(p for (c, _), p in zip(sorted_classes, pcts)
                     if HUMAN_PARSING_LABELS.get(c, '') in
                     ['Hat', 'Hair', 'Sunglasses', 'Scarf', 'Face'])
    ax.text(0.98, 0.95,
            f'Head-related total: {head_total:.1f}%\n(used as binary head mask)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor='#e67e22'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mask_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ mask_class_distribution.png')


def main():
    print('=' * 70)
    print('  数据分布分析（采样统计）')
    print('=' * 70)

    print('\n📁 采样图像中...')
    targets = {
        'human_parsing': DATA_ROOT / 'head_seg/human_parsing/data/train-00000-of-00002-f3a663f7140ee7fd.parquet',
        'celebA_faces': DATA_ROOT / 'head_seg/celebA_faces/data/train-00000-of-00003.parquet',
        'cppe5': DATA_ROOT / 'head_seg/cppe5/data/train-00000-of-00001.parquet',
        'afhqv2': DATA_ROOT / 'head_seg/afhqv2/data/train-00000-of-00013.parquet',
        'dog_food_bg': DATA_ROOT / 'hdri/dog_food_bg/data/train-00000-of-00001-9bf5abf8b080cbba.parquet',
    }

    images_by_ds = {}
    masks_by_ds = {}
    for name, path in targets.items():
        print(f'  采样: {name}')
        imgs, msks = sample_parquet(path, n=200)
        images_by_ds[name] = imgs
        masks_by_ds[name] = msks
        print(f'    → 获得 {len(imgs)} 张图像, {len(msks)} 张mask')

    print('\n📊 生成分析图表...')
    analyze_resolution(images_by_ds)
    analyze_brightness(images_by_ds)
    analyze_rgb_histogram(images_by_ds)
    analyze_mask_classes(masks_by_ds.get('human_parsing', []))

    # 保存分析数据
    summary = {}
    for name, imgs in images_by_ds.items():
        if not imgs:
            continue
        widths = [img.width for img in imgs]
        heights = [img.height for img in imgs]
        brightness = [float(np.mean(np.array(img.convert('L')))) for img in imgs]
        summary[name] = {
            'n_samples_analyzed': len(imgs),
            'resolution': {
                'width_min': min(widths), 'width_max': max(widths),
                'width_mean': float(np.mean(widths)),
                'height_min': min(heights), 'height_max': max(heights),
                'height_mean': float(np.mean(heights)),
            },
            'brightness': {
                'mean': float(np.mean(brightness)),
                'std': float(np.std(brightness)),
                'min': float(np.min(brightness)),
                'max': float(np.max(brightness)),
            },
        }

    with open(OUTPUT_DIR / 'distribution_stats.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  ✓ distribution_stats.json')

    print(f'\n✅ 分布分析报告已生成至: {OUTPUT_DIR}\n')


if __name__ == '__main__':
    main()
