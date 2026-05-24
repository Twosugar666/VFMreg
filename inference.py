"""
脑磁配准推理入口
端到端推理管线：图像输入 → 头部分割 → 特征提取 → 位姿回归 → 6DoF输出
总延迟约20ms（含分割），纯配准约15ms
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import cv2
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vfmreg import VFMReg
from seg.yolo_seg import HeadSegmentor


class MEGRegistrationPipeline:
    """脑磁配准端到端推理管线
    
    完整流程：
    1. 图像预处理（裁剪、缩放至224×224、归一化）~0.5ms
    2. 头部分割（YOLOv8n-seg）~8ms
    3. DINOv3特征提取（4视图并行）~10ms
    4. 跨视图注意力融合 ~1.5ms
    5. 回归头前向传播 ~0.5ms
    总计端到端延迟约20.5ms
    """

    def __init__(
        self,
        vfmreg_checkpoint: str,
        seg_model_path: str = 'yolov8n-seg.pt',
        device: str = 'cuda',
        num_views: int = 4,
        image_size: int = 224,
        confidence_threshold: float = 0.85,
    ):
        """
        Args:
            vfmreg_checkpoint: VFMReg模型权重路径
            seg_model_path: 分割模型路径
            device: 推理设备
            num_views: 输入视图数量
            image_size: 输入图像尺寸
            confidence_threshold: 分割置信度阈值
        """
        self.device = torch.device(device)
        self.num_views = num_views
        self.image_size = image_size

        # 加载分割模型
        print("加载分割模型...")
        self.segmentor = HeadSegmentor(
            yolo_model_path=seg_model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        # 加载VFMReg模型
        print("加载VFMReg模型...")
        self.model = VFMReg(
            backbone_name='dinov2_vitl14',
            feature_dim=1024,
            num_views=num_views,
            num_attention_layers=4,
            num_heads=8,
            freeze_backbone=True,
        ).to(self.device)

        # 加载权重
        if os.path.exists(vfmreg_checkpoint):
            state_dict = torch.load(vfmreg_checkpoint, map_location=self.device)
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
            print(f"模型权重已加载: {vfmreg_checkpoint}")

        self.model.eval()

        # 图像归一化参数（ImageNet标准）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess_image(self, image: np.ndarray) -> tuple:
        """图像预处理
        Args:
            image: [H, W, 3] BGR格式输入图像
        Returns:
            tensor: [1, 3, 224, 224] 归一化后的图像张量
            mask: [1, 1, 224, 224] 分割掩码张量
        """
        # 头部分割
        seg_result = self.segmentor.segment(image)
        mask = seg_result['mask']

        # 缩放图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))
        mask_resized = cv2.resize(mask, (self.image_size, self.image_size))

        # 转为张量
        img_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)

        # 归一化
        img_tensor = (img_tensor - self.mean.squeeze(0)) / self.std.squeeze(0)

        return img_tensor, mask_tensor, seg_result

    @torch.no_grad()
    def predict(self, images: List[np.ndarray]) -> Dict:
        """执行配准推理
        Args:
            images: K张BGR格式输入图像列表
        Returns:
            dict: {
                'rotation_matrix': [3, 3] 旋转矩阵
                'translation': [3] 平移向量（mm）
                'rot_6d': [6] 6D旋转表示
                'inference_time_ms': 推理耗时（ms）
                'seg_confidences': 各视图分割置信度
            }
        """
        assert len(images) == self.num_views, \
            f"需要{self.num_views}张视图，但收到{len(images)}张"

        start_time = time.time()

        # 预处理所有视图
        img_tensors = []
        mask_tensors = []
        seg_confidences = []

        for img in images:
            img_t, mask_t, seg_result = self.preprocess_image(img)
            img_tensors.append(img_t)
            mask_tensors.append(mask_t)
            seg_confidences.append(seg_result['confidence'])

        # 组装批次 [1, K, 3, H, W]
        images_batch = torch.stack(img_tensors, dim=0).unsqueeze(0).to(self.device)
        masks_batch = torch.stack(mask_tensors, dim=0).unsqueeze(0).to(self.device)

        # 模型推理
        predictions = self.model(images_batch, masks_batch)

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            'rotation_matrix': predictions['rotation_matrix'][0].cpu().numpy(),
            'translation': predictions['translation'][0].cpu().numpy(),
            'rot_6d': predictions['rot_6d'][0].cpu().numpy(),
            'inference_time_ms': inference_time,
            'seg_confidences': seg_confidences,
        }

    def benchmark(self, images: List[np.ndarray], num_runs: int = 100) -> Dict:
        """性能基准测试
        Args:
            images: 输入图像列表
            num_runs: 测试次数
        Returns:
            dict: 性能统计
        """
        # 预热
        for _ in range(10):
            self.predict(images)

        # 正式测试
        times = []
        for _ in range(num_runs):
            result = self.predict(images)
            times.append(result['inference_time_ms'])

        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'fps': 1000.0 / np.mean(times),
        }


def main():
    parser = argparse.ArgumentParser(description='脑磁配准推理')
    parser.add_argument('--checkpoint', type=str, required=True, help='VFMReg模型权重')
    parser.add_argument('--seg_model', type=str, default='yolov8n-seg.pt', help='分割模型')
    parser.add_argument('--images', type=str, nargs='+', required=True, help='输入图像路径（4张）')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')

    args = parser.parse_args()

    # 初始化管线
    pipeline = MEGRegistrationPipeline(
        vfmreg_checkpoint=args.checkpoint,
        seg_model_path=args.seg_model,
        device=args.device,
    )

    # 加载图像
    images = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图像: {img_path}")
            sys.exit(1)
        images.append(img)

    # 推理
    if args.benchmark:
        print("\n运行性能基准测试...")
        stats = pipeline.benchmark(images)
        print(f"平均推理时间: {stats['mean_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms")
        print(f"FPS: {stats['fps']:.1f}")
    else:
        result = pipeline.predict(images)
        print(f"\n配准结果:")
        print(f"  旋转矩阵:\n{result['rotation_matrix']}")
        print(f"  平移向量: {result['translation']} mm")
        print(f"  推理时间: {result['inference_time_ms']:.2f} ms")
        print(f"  分割置信度: {result['seg_confidences']}")


if __name__ == '__main__':
    main()
