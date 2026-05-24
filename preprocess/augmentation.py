"""
数据增强与域随机化模块
用于弥合合成数据与真实数据之间的域差距
增强策略：
- 随机值填充（模拟局部遮挡）
- 随机灰度化（概率0.1）
- 颜色抖动（亮度/对比度/饱和度/色调）
- 高斯噪声（σ=0~0.02）
- 随机裁剪和缩放
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from typing import Dict, Tuple
import random


class DomainRandomization:
    """域随机化增强
    在训练时对输入图像施加多种增强变换，
    使模型学会对这些变换保持不变性
    """

    def __init__(
        self,
        grayscale_prob: float = 0.1,
        noise_std_range: Tuple[float, float] = (0.0, 0.02),
        color_jitter: bool = True,
        random_erasing_prob: float = 0.2,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.1),
    ):
        """
        Args:
            grayscale_prob: 随机灰度化概率
            noise_std_range: 高斯噪声标准差范围
            color_jitter: 是否启用颜色抖动
            random_erasing_prob: 随机擦除概率
            random_erasing_scale: 随机擦除面积比例范围
        """
        self.grayscale_prob = grayscale_prob
        self.noise_std_range = noise_std_range
        self.random_erasing_prob = random_erasing_prob
        self.random_erasing_scale = random_erasing_scale

        # 构建增强管线
        aug_list = []

        if color_jitter:
            aug_list.append(
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.15,
                )
            )

        aug_list.append(transforms.RandomGrayscale(p=grayscale_prob))

        self.color_aug = transforms.Compose(aug_list)

        # 随机擦除（模拟局部遮挡）
        self.random_erasing = transforms.RandomErasing(
            p=random_erasing_prob,
            scale=random_erasing_scale,
            ratio=(0.3, 3.3),
            value='random',
        )

    def add_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声
        Args:
            image: [C, H, W] 输入图像张量
        Returns:
            noisy_image: [C, H, W] 添加噪声后的图像
        """
        std = random.uniform(*self.noise_std_range)
        if std > 0:
            noise = torch.randn_like(image) * std
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """应用域随机化增强
        Args:
            image: [C, H, W] 输入图像张量（归一化前）
        Returns:
            augmented: [C, H, W] 增强后的图像
        """
        # 颜色增强
        image = self.color_aug(image)

        # 高斯噪声
        image = self.add_gaussian_noise(image)

        # 随机擦除
        image = self.random_erasing(image)

        return image


class BlenderDataGenerator:
    """Blender合成数据生成管线配置
    通过Python脚本控制Blender进行程序化渲染
    域随机化参数：
    - 头型：50+ SMPL-X形状参数随机采样
    - 相机焦距：35-85mm
    - 光源数量：2-5个，随机位置和强度
    - 色温范围：3000-8000K
    - 背景类型：HDRI/纯色/纹理
    - 噪声标准差：0-0.02
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config: 渲染配置字典
        """
        self.config = config or self.default_config()

    @staticmethod
    def default_config() -> Dict:
        """默认渲染配置"""
        return {
            # 头型参数
            'num_head_shapes': 50,
            'smplx_betas_std': 3.0,

            # 相机参数
            'focal_length_range': (35, 85),  # mm
            'num_views': 4,
            'camera_distance_range': (0.5, 1.5),  # m
            'camera_elevation_range': (-15, 30),  # degrees
            'camera_azimuth_step': 90,  # degrees between views

            # 光照参数
            'num_lights_range': (2, 5),
            'light_intensity_range': (100, 1000),  # W
            'color_temperature_range': (3000, 8000),  # K

            # 背景参数
            'background_types': ['hdri', 'solid', 'texture'],
            'num_hdri_envs': 100,

            # 渲染参数
            'render_engine': 'CYCLES',
            'render_samples': 128,
            'image_resolution': (512, 512),

            # 噪声参数
            'noise_std_range': (0.0, 0.02),

            # 输出参数
            'total_samples': 100000,
            'output_format': 'png',
        }

    def generate_blender_script(self, output_dir: str, sample_id: int) -> str:
        """生成Blender Python渲染脚本
        Args:
            output_dir: 输出目录
            sample_id: 样本ID
        Returns:
            script: Blender Python脚本内容
        """
        config = self.config
        script = f'''
import bpy
import numpy as np
import json
import os
import mathutils

# 清除默认场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 设置渲染引擎
bpy.context.scene.render.engine = '{config["render_engine"]}'
bpy.context.scene.cycles.samples = {config["render_samples"]}
bpy.context.scene.render.resolution_x = {config["image_resolution"][0]}
bpy.context.scene.render.resolution_y = {config["image_resolution"][1]}

# 随机种子
np.random.seed({sample_id})

# ===== 1. 加载头部模型（SMPL-X） =====
# 随机形状参数
betas = np.random.randn(10) * {config["smplx_betas_std"]}

# ===== 2. 加载头盔模型 =====
# helmet_path = "path/to/helmet.obj"
# bpy.ops.import_scene.obj(filepath=helmet_path)

# ===== 3. 随机位姿 =====
# 随机旋转（欧拉角）
rot_x = np.random.uniform(-15, 15) * np.pi / 180
rot_y = np.random.uniform(-30, 30) * np.pi / 180
rot_z = np.random.uniform(-10, 10) * np.pi / 180

# 随机平移
trans_x = np.random.uniform(-0.02, 0.02)
trans_y = np.random.uniform(-0.02, 0.02)
trans_z = np.random.uniform(-0.02, 0.02)

# 构建变换矩阵
rotation_matrix = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
translation = mathutils.Vector((trans_x, trans_y, trans_z))

# ===== 4. 设置光照 =====
num_lights = np.random.randint({config["num_lights_range"][0]}, {config["num_lights_range"][1]} + 1)
for i in range(num_lights):
    light_data = bpy.data.lights.new(name=f"Light_{{i}}", type='POINT')
    light_data.energy = np.random.uniform({config["light_intensity_range"][0]}, {config["light_intensity_range"][1]})
    # 色温
    color_temp = np.random.uniform({config["color_temperature_range"][0]}, {config["color_temperature_range"][1]})
    light_data.color = (1.0, 0.9, 0.8)  # 简化的色温映射
    
    light_obj = bpy.data.objects.new(name=f"Light_{{i}}", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (
        np.random.uniform(-2, 2),
        np.random.uniform(-2, 2),
        np.random.uniform(1, 3)
    )

# ===== 5. 设置相机并渲染多视角 =====
focal_length = np.random.uniform({config["focal_length_range"][0]}, {config["focal_length_range"][1]})
camera_distance = np.random.uniform({config["camera_distance_range"][0]}, {config["camera_distance_range"][1]})

output_dir = "{output_dir}/{sample_id:06d}"
os.makedirs(output_dir, exist_ok=True)

camera_poses = []
for view_idx in range({config["num_views"]}):
    azimuth = view_idx * {config["camera_azimuth_step"]}
    elevation = np.random.uniform({config["camera_elevation_range"][0]}, {config["camera_elevation_range"][1]})
    
    # 设置相机位置（球坐标）
    cam_x = camera_distance * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    cam_y = camera_distance * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
    cam_z = camera_distance * np.sin(np.radians(elevation))
    
    # 创建相机
    cam_data = bpy.data.cameras.new(name=f"Camera_{{view_idx}}")
    cam_data.lens = focal_length
    cam_obj = bpy.data.objects.new(name=f"Camera_{{view_idx}}", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (cam_x, cam_y, cam_z)
    
    # 相机朝向原点
    direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    # 设置为活动相机并渲染
    bpy.context.scene.camera = cam_obj
    
    # 渲染RGB图像
    bpy.context.scene.render.filepath = f"{{output_dir}}/render_{{view_idx:02d}}.png"
    bpy.ops.render.render(write_still=True)
    
    # 渲染分割掩码（使用材质ID）
    bpy.context.scene.render.filepath = f"{{output_dir}}/mask_{{view_idx:02d}}.png"
    # 切换到掩码渲染模式...
    bpy.ops.render.render(write_still=True)
    
    # 记录相机位姿
    cam_matrix = cam_obj.matrix_world
    camera_poses.append({{
        'view_idx': view_idx,
        'transform_matrix': [list(row) for row in cam_matrix],
    }})

# ===== 6. 保存元数据 =====
meta = {{
    'sample_id': {sample_id},
    'rotation': [list(row) for row in rotation_matrix],
    'translation': list(translation),
    'focal_length': focal_length,
    'camera_distance': camera_distance,
    'num_lights': num_lights,
    'camera_poses': camera_poses,
    'betas': betas.tolist(),
}}

with open(f"{{output_dir}}/meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

print(f"Sample {sample_id} rendered successfully.")
'''
        return script

    def generate_batch_script(self, output_dir: str, start_id: int, end_id: int) -> str:
        """生成批量渲染的Shell脚本"""
        script = f'''#!/bin/bash
# 批量渲染合成数据
# 使用方法: bash render_batch.sh

BLENDER_PATH="blender"
OUTPUT_DIR="{output_dir}"

mkdir -p $OUTPUT_DIR

for i in $(seq {start_id} {end_id}); do
    echo "Rendering sample $i..."
    $BLENDER_PATH --background --python render_sample_$i.py
done

echo "Batch rendering complete. Total samples: {end_id - start_id + 1}"
'''
        return script
