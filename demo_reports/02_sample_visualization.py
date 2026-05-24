"""
================================================================
脚本 02: 数据样本可视化网格图
================================================================
功能: 从每个数据集中随机抽取若干样本，生成可视化网格图
输出: ./demo_reports/output/02_samples/
用法: python demo_reports/02_sample_visualization.py
================================================================
"""

import io
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_ROOT = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/data')
OUTPUT_DIR = Path('/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/demo_reports/output/02_samples')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)


def load_parquet_samples(parquet_path, n_samples=8, has_mask=False):
    """从 parquet 文件随机抽取样本"""
    if not parquet_path.exists():
        return []
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f'  读取 parquet 失败: {e}')
        return []

    indices = random.sample(range(len(df)), min(n_samples, len(df)))
    samples = []
    for idx in indices:
        row = df.iloc[idx]
        try:
            img_field = row['image'] if 'image' in df.columns else row.get('img')
            img_bytes = img_field['bytes'] if isinstance(img_field, dict) else img_field
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            mask = None
            if has_mask and 'mask' in df.columns:
                mf = row['mask']
                mb = mf['bytes'] if isinstance(mf, dict) else mf
                mask = Image.open(io.BytesIO(mb))
            samples.append({'image': img, 'mask': mask})
        except Exception:
            continue
    return samples


def load_zip_samples(zip_path, n_samples=8):
    """从 zip 文件抽取图像"""
    if not zip_path.exists():
        return []
    samples = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith(('.jpg', '.jpeg', '.png'))]
            members = random.sample(members, min(n_samples, len(members)))
            for m in members:
                try:
                    with z.open(m) as f:
                        img = Image.open(io.BytesIO(f.read())).convert('RGB')
                        samples.append({'image': img, 'mask': None})
                except Exception:
                    continue
    except Exception as e:
        print(f'  读取 zip 失败: {e}')
    return samples


def plot_grid(samples, title, save_path, n_cols=4, with_mask=False):
    if not samples:
        print(f'  ⚠ 无样本: {title}')
        return False

    n = len(samples)
    n_cols = min(n_cols, n)

    if with_mask and any(s.get('mask') is not None for s in samples):
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows * 2, n_cols,
                                  figsize=(n_cols * 3, n_rows * 6))
        axes = np.atleast_2d(axes)
        for i, s in enumerate(samples):
            r, c = (i // n_cols) * 2, i % n_cols
            axes[r, c].imshow(s['image'])
            axes[r, c].set_title(f'image #{i+1}', fontsize=9)
            axes[r, c].axis('off')
            if s.get('mask') is not None:
                axes[r + 1, c].imshow(np.array(s['mask']), cmap='tab20')
                axes[r + 1, c].set_title(f'mask #{i+1}', fontsize=9)
            axes[r + 1, c].axis('off')
        for idx in range(n, n_rows * n_cols):
            r, c = (idx // n_cols) * 2, idx % n_cols
            axes[r, c].axis('off')
            axes[r + 1, c].axis('off')
    else:
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 3, n_rows * 3))
        axes = np.atleast_2d(axes)
        for i, s in enumerate(samples):
            r, c = i // n_cols, i % n_cols
            axes[r, c].imshow(s['image'])
            axes[r, c].set_title(f'#{i+1} {s["image"].size}', fontsize=8)
            axes[r, c].axis('off')
        for idx in range(n, n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r, c].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  ✓ 保存: {save_path}')
    return True


DATASET_CFG = [
        {'name': 'human_parsing',
     'title': 'Human Parsing Dataset (head segmentation with semantic mask)',
     'parquet': DATA_ROOT / 'head_seg/human_parsing/data/train-00000-of-00002-f3a663f7140ee7fd.parquet',
     'has_mask': True, 'n': 6},
    {'name': 'celebA_faces',
     'title': 'CelebA-Faces (high-quality face images)',
     'parquet': DATA_ROOT / 'head_seg/celebA_faces/data/train-00000-of-00003.parquet',
     'has_mask': False, 'n': 8},
    {'name': 'cppe5',
     'title': 'CPPE-5 (Medical Protective Equipment)',
     'parquet': DATA_ROOT / 'head_seg/cppe5/data/train-00000-of-00001.parquet',
     'has_mask': False, 'n': 8},
    {'name': 'afhqv2',
     'title': 'AFHQv2 (Animal Faces - OOD test data)',
     'parquet': DATA_ROOT / 'head_seg/afhqv2/data/train-00000-of-00013.parquet',
     'has_mask': False, 'n': 8},
    {'name': 'dog_food_bg',
     'title': 'Dog/Food Images (background domain randomization)',
     'parquet': DATA_ROOT / 'hdri/dog_food_bg/data/train-00000-of-00001-9bf5abf8b080cbba.parquet',
     'has_mask': False, 'n': 8},
]

ZIP_CFG = [
    {'name': 'protective_equipment',
     'title': 'Protective Equipment (Helmet - closest to OPM-MEG scene)',
     'zip': DATA_ROOT / 'head_seg/protective_equipment/data/valid.zip',
     'n': 8},
]


def main():
    print('=' * 70)
    print('  数据集样本可视化')
    print('=' * 70)

    success = []
    for ds in DATASET_CFG:
        print(f'\n[{ds["name"]}] 加载样本...')
        samples = load_parquet_samples(ds['parquet'], ds['n'], ds['has_mask'])
        sp = OUTPUT_DIR / f'grid_{ds["name"]}.png'
        if plot_grid(samples, ds['title'], sp, with_mask=ds['has_mask']):
            success.append((ds['name'], samples[:2]))

    for ds in ZIP_CFG:
        print(f'\n[{ds["name"]}] 加载 zip 样本...')
        samples = load_zip_samples(ds['zip'], ds['n'])
        sp = OUTPUT_DIR / f'grid_{ds["name"]}.png'
        if plot_grid(samples, ds['title'], sp):
            success.append((ds['name'], samples[:2]))

    # 综合对比图
    if len(success) >= 3:
        print('\n[combined] 生成跨数据集对比图...')
        imgs, titles = [], []
        for name, samples in success:
            for s in samples[:2]:
                imgs.append(s['image'])
                titles.append(name)
        n_cols = 4
        n_rows = (len(imgs) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        axes = np.atleast_2d(axes)
        for i, (img, t) in enumerate(zip(imgs, titles)):
            r, c = i // n_cols, i % n_cols
            axes[r, c].imshow(img)
            axes[r, c].set_title(t, fontsize=9, fontweight='bold')
            axes[r, c].axis('off')
        for idx in range(len(imgs), n_rows * n_cols):
            r, c = idx // n_cols, idx % n_cols
            axes[r, c].axis('off')
        fig.suptitle('Cross-Dataset Sample Overview',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'grid_combined.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  ✓ 保存: {OUTPUT_DIR}/grid_combined.png')

    print(f'\n✅ 样本可视化已生成至: {OUTPUT_DIR}\n')


if __name__ == '__main__':
    main()
