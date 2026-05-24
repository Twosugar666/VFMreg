"""
数据集定义
- MEGHeadDataset: 真实数据集（多视角图像 + COLMAP位姿）
- SyntheticDataset: 合成数据集（Blender渲染 + 精确6D位姿标注）
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, List, Tuple


class MEGHeadDataset(Dataset):
    """MEG-Head-360真实数据集
    每个样本包含K=4张多视角图像及对应的6D位姿标注
    用于VFMReg的真实域微调和测试
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_views: int = 4,
        image_size: int = 224,
        use_masks: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            num_views: 每组视图数量
            image_size: 输入图像尺寸
            use_masks: 是否使用分割掩码
            transform: 额外的数据变换
        """
        self.data_root = data_root
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.use_masks = use_masks

        # 加载数据索引
        self.samples = self._load_annotations()

        # 图像预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # 掩码预处理
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_annotations(self) -> List[Dict]:
        """加载数据标注文件"""
        anno_path = os.path.join(self.data_root, f'{self.split}_annotations.json')
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        else:
            # 如果标注文件不存在，扫描目录结构
            return self._scan_directory()

    def _scan_directory(self) -> List[Dict]:
        """扫描目录结构构建样本列表"""
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_dir):
            return samples

        for subject_dir in sorted(os.listdir(split_dir)):
            subject_path = os.path.join(split_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            # 每个被试目录下包含多组多视角图像
            pose_file = os.path.join(subject_path, 'poses.json')
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    poses = json.load(f)

                for group_id, pose_data in poses.items():
                    sample = {
                        'subject': subject_dir,
                        'group_id': group_id,
                        'images': [
                            os.path.join(subject_path, f'view_{i}.png')
                            for i in range(self.num_views)
                        ],
                        'masks': [
                            os.path.join(subject_path, f'mask_{i}.png')
                            for i in range(self.num_views)
                        ],
                        'rotation': pose_data['rotation'],      # 3x3旋转矩阵
                        'translation': pose_data['translation'],  # 3D平移向量
                    }
                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本
        Returns:
            dict: {
                'images': [K, 3, H, W] 多视角图像
                'masks': [K, 1, H, W] 分割掩码
                'rotation': [3, 3] 旋转矩阵
                'translation': [3] 平移向量
            }
        """
        sample = self.samples[idx]

        # 加载多视角图像
        images = []
        for img_path in sample['images'][:self.num_views]:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
            else:
                img = torch.zeros(3, self.image_size, self.image_size)
            images.append(img)
        images = torch.stack(images, dim=0)  # [K, 3, H, W]

        # 加载分割掩码
        masks = []
        if self.use_masks:
            for mask_path in sample['masks'][:self.num_views]:
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = self.mask_transform(mask)
                else:
                    mask = torch.ones(1, self.image_size, self.image_size)
                masks.append(mask)
        else:
            masks = [torch.ones(1, self.image_size, self.image_size)] * self.num_views
        masks = torch.stack(masks, dim=0)  # [K, 1, H, W]

        # 位姿标注
        rotation = torch.tensor(sample['rotation'], dtype=torch.float32)
        translation = torch.tensor(sample['translation'], dtype=torch.float32)

        return {
            'images': images,
            'masks': masks,
            'rotation': rotation,
            'translation': translation,
        }


class SyntheticDataset(Dataset):
    """合成数据集
    基于Blender物理渲染引擎生成
    每个样本包含K=4张多视角图像 + 精确6D位姿真值 + 像素级分割掩码
    域随机化参数：
    - 头型：50+ SMPL-X形状参数
    - 相机焦距：35-85mm
    - 光源：2-5个，随机位置和强度
    - 色温：3000-8000K
    - 背景：HDRI/纯色/纹理
    - 噪声：高斯噪声σ=0~0.02
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_views: int = 4,
        image_size: int = 224,
        augment: bool = True,
    ):
        """
        Args:
            data_root: 合成数据根目录
            split: 数据集划分
            num_views: 视图数量
            image_size: 图像尺寸
            augment: 是否启用数据增强
        """
        self.data_root = data_root
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.augment = augment

        # 加载样本索引
        self.samples = self._load_samples()

        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # 数据增强变换
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomGrayscale(p=0.1),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.3, hue=0.1
                ),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            self.aug_transform = None

    def _load_samples(self) -> List[Dict]:
        """加载合成数据样本索引"""
        manifest_path = os.path.join(self.data_root, f'{self.split}_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)

        # 扫描目录
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_dir):
            return samples

        for sample_dir in sorted(os.listdir(split_dir)):
            sample_path = os.path.join(split_dir, sample_dir)
            if not os.path.isdir(sample_path):
                continue

            meta_file = os.path.join(sample_path, 'meta.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                meta['path'] = sample_path
                samples.append(meta)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个合成样本"""
        sample = self.samples[idx]
        sample_path = sample.get('path', os.path.join(self.data_root, self.split, f'{idx:06d}'))

        # 加载多视角图像
        images = []
        masks = []
        for i in range(self.num_views):
            # 图像
            img_path = os.path.join(sample_path, f'render_{i:02d}.png')
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.new('RGB', (self.image_size, self.image_size))

            # 掩码
            mask_path = os.path.join(sample_path, f'mask_{i:02d}.png')
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                mask = Image.new('L', (self.image_size, self.image_size), 255)

            # 变换
            img = self.base_transform(img)
            if self.augment and self.aug_transform is not None:
                img = self.aug_transform(img)

            mask = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])(mask)

            images.append(img)
            masks.append(mask)

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        # 位姿标注（精确真值）
        rotation = torch.tensor(sample.get('rotation', np.eye(3).tolist()), dtype=torch.float32)
        translation = torch.tensor(sample.get('translation', [0., 0., 0.]), dtype=torch.float32)

        return {
            'images': images,
            'masks': masks,
            'rotation': rotation,
            'translation': translation,
        }


class NeRFDataset(Dataset):
    """NeRF训练数据集
    单个被试的多视角图像 + COLMAP估计的相机位姿
    用于训练被试专属的NeRF模型
    """

    def __init__(
        self,
        data_root: str,
        image_size: Tuple[int, int] = (256, 256),
        num_rays_per_batch: int = 4096,
    ):
        """
        Args:
            data_root: 数据目录（包含images/和transforms.json）
            image_size: 图像尺寸
            num_rays_per_batch: 每批次采样光线数
        """
        self.data_root = data_root
        self.image_size = image_size
        self.num_rays_per_batch = num_rays_per_batch

        # 加载相机参数和图像
        self.images, self.poses, self.intrinsics = self._load_data()

    def _load_data(self) -> Tuple[List, List, Dict]:
        """加载COLMAP估计的相机参数"""
        transforms_path = os.path.join(self.data_root, 'transforms.json')

        if not os.path.exists(transforms_path):
            return [], [], {}

        with open(transforms_path, 'r') as f:
            data = json.load(f)

        # 相机内参
        intrinsics = {
            'fx': data.get('fl_x', 500.0),
            'fy': data.get('fl_y', 500.0),
            'cx': data.get('cx', self.image_size[1] / 2),
            'cy': data.get('cy', self.image_size[0] / 2),
            'w': data.get('w', self.image_size[1]),
            'h': data.get('h', self.image_size[0]),
        }

        images = []
        poses = []

        for frame in data.get('frames', []):
            img_path = os.path.join(self.data_root, frame['file_path'])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img = np.array(img).astype(np.float32) / 255.0
                images.append(img)

                # 4x4变换矩阵（camera-to-world）
                pose = np.array(frame['transform_matrix'], dtype=np.float32)
                poses.append(pose)

        return images, poses, intrinsics

    def get_rays(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """根据相机位姿生成光线
        Args:
            pose: [4, 4] camera-to-world变换矩阵
        Returns:
            rays_o: [H*W, 3] 光线起点
            rays_d: [H*W, 3] 光线方向
        """
        H, W = self.image_size
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        # 生成像素坐标网格
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )

        # 相机坐标系下的方向
        dirs = np.stack([
            (i - cx) / fx,
            -(j - cy) / fy,
            -np.ones_like(i)
        ], axis=-1)

        # 变换到世界坐标系
        rays_d = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        return rays_o, rays_d

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单张图像的随机光线批次"""
        image = self.images[idx]
        pose = self.poses[idx]

        # 生成所有光线
        rays_o, rays_d = self.get_rays(pose)
        target_rgb = image.reshape(-1, 3)

        # 随机采样光线
        num_rays = min(self.num_rays_per_batch, rays_o.shape[0])
        indices = np.random.choice(rays_o.shape[0], num_rays, replace=False)

        return {
            'rays_o': torch.from_numpy(rays_o[indices]),
            'rays_d': torch.from_numpy(rays_d[indices]),
            'target_rgb': torch.from_numpy(target_rgb[indices]),
        }
