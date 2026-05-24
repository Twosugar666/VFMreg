"""
数据集转换工具
将开源数据集（HuggingFace parquet 格式）转换为 YOLOv8 训练格式
- 输入：HuggingFace parquet（image + mask 列）
- 输出：YOLO segmentation 标注（YOLOv8-seg 格式）

使用方式：
    python -m preprocess.convert_datasets --dataset human_parsing
    python -m preprocess.convert_datasets --dataset protective_equipment
    python -m preprocess.convert_datasets --dataset all
"""

import os
import sys
import argparse
import io
import json
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm


# ============================================================
# Human Parsing 数据集类别 → "头部"二分类映射
# 原始类别: 0=Background, 1=Hat, 2=Hair, 3=Glove, 4=Sunglasses,
#          5=Upper-clothes, 6=Dress, 7=Coat, 8=Socks, 9=Pants,
#          10=Jumpsuits, 11=Scarf, 12=Skirt, 13=Face, 14=L-arm,
#          15=R-arm, 16=L-leg, 17=R-leg, 18=L-shoe, 19=R-shoe
# ============================================================
HUMAN_PARSING_HEAD_CLASSES = {1, 2, 4, 11, 13}  # Hat, Hair, Sunglasses, Scarf, Face


def mask_to_yolo_polygon(mask: np.ndarray, normalize_size: Tuple[int, int]) -> List[List[float]]:
    """将二值 mask 转换为 YOLO segmentation 多边形坐标
    Args:
        mask: [H, W] 二值掩码 (0/1)
        normalize_size: (W, H) 归一化尺寸
    Returns:
        polygons: 每个轮廓的归一化坐标列表 [[x1,y1,x2,y2,...], ...]
    """
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    W, H = normalize_size
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # 过滤小区域
            continue
        # 简化轮廓
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        if len(contour) < 3:
            continue
        # 归一化坐标
        coords = contour.reshape(-1, 2).astype(np.float32)
        coords[:, 0] /= W
        coords[:, 1] /= H
        polygons.append(coords.flatten().tolist())
    return polygons


def convert_human_parsing_to_yolo(
    parquet_dir: str,
    output_dir: str,
    head_class_ids: set = HUMAN_PARSING_HEAD_CLASSES,
    train_ratio: float = 0.9,
    max_samples: Optional[int] = None,
):
    """转换 human_parsing_dataset 为 YOLO 头部分割格式
    Args:
        parquet_dir: parquet 文件所在目录
        output_dir: 输出目录（YOLO 格式）
        head_class_ids: 视为"头部"的原始类别 ID 集合
        train_ratio: 训练集比例
        max_samples: 最大样本数（None=全部）
    """
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)

    # 创建 YOLO 标准目录结构
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 读取所有 parquet 文件
    parquet_files = sorted(parquet_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"未找到 parquet 文件: {parquet_dir}")
        return

    sample_idx = 0
    train_count = 0
    val_count = 0

    for pq_file in parquet_files:
        print(f"处理 {pq_file.name}...")
        df = pd.read_parquet(pq_file)

        for _, row in tqdm(df.iterrows(), total=len(df), desc='转换中'):
            if max_samples is not None and sample_idx >= max_samples:
                break

            try:
                # 解析图像和 mask
                img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                mask_pil = Image.open(io.BytesIO(row['mask']['bytes']))
                mask = np.array(mask_pil)

                # 提取头部区域
                head_mask = np.zeros_like(mask, dtype=np.uint8)
                for cls_id in head_class_ids:
                    head_mask[mask == cls_id] = 1

                if head_mask.sum() < 200:  # 太小则跳过
                    continue

                # 转 YOLO 多边形
                polygons = mask_to_yolo_polygon(
                    head_mask, normalize_size=(img.width, img.height)
                )
                if not polygons:
                    continue

                # 划分 train/val
                is_train = np.random.rand() < train_ratio
                split = 'train' if is_train else 'val'
                if is_train:
                    train_count += 1
                else:
                    val_count += 1

                # 保存图像（转 jpg 节省空间）
                img_save_path = output_dir / 'images' / split / f'{sample_idx:07d}.jpg'
                img.save(img_save_path, quality=92)

                # 保存 YOLO 标注（class_id=0 表示 head）
                label_path = output_dir / 'labels' / split / f'{sample_idx:07d}.txt'
                with open(label_path, 'w') as f:
                    for poly in polygons:
                        coords_str = ' '.join(f'{c:.6f}' for c in poly)
                        f.write(f'0 {coords_str}\n')

                sample_idx += 1
            except Exception as e:
                continue

        if max_samples is not None and sample_idx >= max_samples:
            break

    # 生成 YOLOv8 数据配置 yaml
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f'''# 头部分割 YOLO 数据集配置
# 来源：HuggingFace mattmdjaga/human_parsing_dataset
# 二分类：0 = head（含 face/hair/hat/sunglasses/scarf）

path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: head

# 统计
# 总样本: {sample_idx}, 训练: {train_count}, 验证: {val_count}
''')

    print(f"\n✅ 转换完成！")
    print(f"   总样本: {sample_idx}")
    print(f"   训练集: {train_count}")
    print(f"   验证集: {val_count}")
    print(f"   输出目录: {output_dir}")
    print(f"   YOLO 配置: {yaml_path}")


def convert_celeba_to_images(
    parquet_path: str,
    output_dir: str,
    max_samples: int = 10000,
):
    """从 CelebA-faces parquet 中提取图像作为头部分割训练补充
    使用预训练分割模型生成伪标签（推理时调用）
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    print(f"加载 {len(df)} 条数据，最多保存 {max_samples} 张")

    saved = 0
    for idx, row in tqdm(df.iterrows(), total=min(len(df), max_samples)):
        if saved >= max_samples:
            break
        try:
            img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
            img.save(output_dir / f'celeba_{saved:06d}.jpg', quality=92)
            saved += 1
        except Exception:
            continue

    print(f"✅ 已保存 {saved} 张 CelebA 人脸图像到 {output_dir}")


def convert_protective_equipment_to_yolo(
    zip_dir: str,
    output_dir: str,
):
    """转换 keremberke/protective-equipment-detection 数据集为头部 YOLO 格式
    数据集本身已是 YOLO 格式，仅需解压和重映射类别
    原始类别可能包含: helmet, head, person, vest, gloves, ...
    我们统一映射为 head 类别（包含人头 + 头盔区域）
    """
    import zipfile

    zip_dir = Path(zip_dir)
    output_dir = Path(output_dir)
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 解压所有 zip
    for zip_name in ['train.zip', 'valid.zip', 'test.zip']:
        zip_path = zip_dir / zip_name
        if not zip_path.exists():
            continue
        split = 'train' if zip_name == 'train.zip' else 'val'
        print(f"解压 {zip_name} → {split}...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(output_dir / 'tmp' / split)

    print("✅ 已解压。请进一步处理类别映射（数据集原始格式为 YOLO，类别需重新映射）")
    print(f"   输出目录: {output_dir}/tmp/")


def main():
    parser = argparse.ArgumentParser(description='数据集转换工具')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['human_parsing', 'celeba', 'protective_equipment', 'all'],
                        help='要转换的数据集')
    parser.add_argument('--data_root', type=str,
                        default='/apdcephfs/ceph-sz1-csp/user_xuanboguo/26-04/school/code/data',
                        help='数据根目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于调试）')

    args = parser.parse_args()
    data_root = Path(args.data_root)

    if args.dataset in ['human_parsing', 'all']:
        print("=" * 60)
        print("转换 Human Parsing → YOLO 头部分割")
        print("=" * 60)
        convert_human_parsing_to_yolo(
            parquet_dir=data_root / 'head_seg/human_parsing/data',
            output_dir=data_root / 'head_seg/human_parsing_yolo',
            max_samples=args.max_samples,
        )

    if args.dataset in ['celeba', 'all']:
        print("=" * 60)
        print("提取 CelebA-faces 图像")
        print("=" * 60)
        convert_celeba_to_images(
            parquet_path=data_root / 'head_seg/celebA_faces/data/train-00000-of-00003.parquet',
            output_dir=data_root / 'head_seg/celeba_images',
            max_samples=args.max_samples or 10000,
        )

    if args.dataset in ['protective_equipment', 'all']:
        print("=" * 60)
        print("转换 Protective Equipment → YOLO 头盔分割")
        print("=" * 60)
        convert_protective_equipment_to_yolo(
            zip_dir=data_root / 'head_seg/protective_equipment/data',
            output_dir=data_root / 'head_seg/helmet_yolo',
        )


if __name__ == '__main__':
    main()
