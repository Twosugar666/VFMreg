"""
基于YOLOv8n-seg的头部轮廓分割
- 多尺度交叉熵损失函数（P3/P4/P5三个尺度加权）
- Sobel边缘增强策略
- 级联策略支持（置信度阈值切换）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from ultralytics import YOLO


class SobelEdgeEnhancement(nn.Module):
    """Sobel边缘增强模块
    利用3x3 Sobel卷积核计算水平和垂直方向梯度，
    梯度幅值作为边缘响应，与标准分割损失加权融合
    """

    def __init__(self, threshold: float = 0.3):
        """
        Args:
            threshold: 边缘响应阈值，高于此值的像素被视为边缘区域
        """
        super().__init__()
        self.threshold = threshold

        # Sobel卷积核（水平方向）
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Sobel卷积核（垂直方向）
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_edge_map(self, mask: torch.Tensor) -> torch.Tensor:
        """计算边缘响应图
        Args:
            mask: [batch, 1, H, W] 分割掩码
        Returns:
            edge_map: [batch, 1, H, W] 边缘响应图
        """
        # 计算水平和垂直梯度
        grad_x = F.conv2d(mask, self.sobel_x, padding=1)
        grad_y = F.conv2d(mask, self.sobel_y, padding=1)

        # 梯度幅值
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 归一化到[0, 1]
        edge_map = edge_map / (edge_map.max() + 1e-8)

        return edge_map

    def get_edge_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """获取边缘区域的二值掩码
        Args:
            mask: [batch, 1, H, W] 分割掩码
        Returns:
            edge_binary: [batch, 1, H, W] 边缘区域二值掩码
        """
        edge_map = self.compute_edge_map(mask)
        edge_binary = (edge_map > self.threshold).float()
        return edge_binary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边缘区域的损失
        仅在Sobel梯度响应高于阈值的像素上计算交叉熵损失
        Args:
            pred: [batch, 1, H, W] 预测分割logits
            target: [batch, 1, H, W] 真值分割掩码
        Returns:
            edge_loss: 边缘区域损失
        """
        # 计算真值掩码的边缘区域
        edge_mask = self.get_edge_mask(target)

        # 仅在边缘区域计算BCE损失
        if edge_mask.sum() > 0:
            pred_edge = pred[edge_mask > 0]
            target_edge = target[edge_mask > 0]
            edge_loss = F.binary_cross_entropy_with_logits(pred_edge, target_edge)
        else:
            edge_loss = torch.tensor(0.0, device=pred.device)

        return edge_loss


class MultiScaleLoss(nn.Module):
    """多尺度交叉熵损失函数
    在P3(步长8)、P4(步长16)、P5(步长32)三个分辨率上分别计算分割损失并加权求和
    L_multi = λ3*L_P3 + λ4*L_P4 + λ5*L_P5
    权重系数：λ3=0.5, λ4=0.3, λ5=0.2
    """

    def __init__(
        self,
        lambda_p3: float = 0.5,
        lambda_p4: float = 0.3,
        lambda_p5: float = 0.2,
        lambda_edge: float = 0.1,
        edge_threshold: float = 0.3,
    ):
        """
        Args:
            lambda_p3: P3尺度权重（空间分辨率最高）
            lambda_p4: P4尺度权重
            lambda_p5: P5尺度权重（感受野最大）
            lambda_edge: 边缘增强系数
            edge_threshold: Sobel边缘响应阈值
        """
        super().__init__()
        self.lambda_p3 = lambda_p3
        self.lambda_p4 = lambda_p4
        self.lambda_p5 = lambda_p5
        self.lambda_edge = lambda_edge

        # Sobel边缘增强模块
        self.edge_enhancer = SobelEdgeEnhancement(threshold=edge_threshold)

    def weighted_bce_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """加权二值交叉熵损失
        正负样本权重通过类别像素比例的倒数动态计算
        """
        # 计算正负样本比例
        pos_ratio = target.mean()
        neg_ratio = 1.0 - pos_ratio

        # 权重为比例的倒数（缓解前景-背景不平衡）
        pos_weight = neg_ratio / (pos_ratio + 1e-6)

        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=torch.tensor([pos_weight], device=pred.device)
        )
        return loss

    def forward(self, predictions: Dict[str, torch.Tensor],
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算多尺度损失
        Args:
            predictions: 包含P3/P4/P5三个尺度预测的字典
                - 'p3': [batch, 1, H/8, W/8]
                - 'p4': [batch, 1, H/16, W/16]
                - 'p5': [batch, 1, H/32, W/32]
            target: [batch, 1, H, W] 原始分辨率的真值掩码
        Returns:
            dict: 包含各项损失
        """
        losses = {}

        # P3尺度损失（步长8）
        target_p3 = F.interpolate(target, size=predictions['p3'].shape[-2:], mode='nearest')
        loss_p3 = self.weighted_bce_loss(predictions['p3'], target_p3)
        losses['loss_p3'] = loss_p3

        # P4尺度损失（步长16）
        target_p4 = F.interpolate(target, size=predictions['p4'].shape[-2:], mode='nearest')
        loss_p4 = self.weighted_bce_loss(predictions['p4'], target_p4)
        losses['loss_p4'] = loss_p4

        # P5尺度损失（步长32）
        target_p5 = F.interpolate(target, size=predictions['p5'].shape[-2:], mode='nearest')
        loss_p5 = self.weighted_bce_loss(predictions['p5'], target_p5)
        losses['loss_p5'] = loss_p5

        # 多尺度加权总损失
        seg_loss = (self.lambda_p3 * loss_p3 +
                    self.lambda_p4 * loss_p4 +
                    self.lambda_p5 * loss_p5)

        # Sobel边缘增强损失（在最高分辨率P3上计算）
        edge_loss = self.edge_enhancer(predictions['p3'], target_p3)
        losses['edge_loss'] = edge_loss

        # 总损失 L_total = L_seg + λ_edge * L_edge
        total_loss = seg_loss + self.lambda_edge * edge_loss
        losses['total_loss'] = total_loss
        losses['seg_loss'] = seg_loss

        return losses


class HeadSegmentor:
    """头部分割器
    基于YOLOv8n-seg的快速分割 + 级联策略
    - 默认使用YOLOv8n-seg（~8ms/帧）
    - 置信度低于阈值时切换到VLM备份
    """

    def __init__(
        self,
        yolo_model_path: str = 'yolov8n-seg.pt',
        confidence_threshold: float = 0.85,
        device: str = 'cuda',
    ):
        """
        Args:
            yolo_model_path: YOLOv8n-seg模型权重路径
            confidence_threshold: 置信度阈值，低于此值切换VLM
            device: 推理设备
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # 加载YOLOv8n-seg模型
        self.yolo_model = YOLO(yolo_model_path)

    def segment(self, image: np.ndarray) -> Dict:
        """执行头部分割
        Args:
            image: [H, W, 3] BGR格式输入图像
        Returns:
            dict: 包含分割掩码、置信度、边界框等
        """
        # YOLOv8推理
        results = self.yolo_model(image, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            return {
                'mask': np.zeros(image.shape[:2], dtype=np.uint8),
                'confidence': 0.0,
                'need_vlm_backup': True,
            }

        # 获取最高置信度的分割结果
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            confidence = confidences[best_idx]

            # 获取分割掩码
            mask = result.masks.data[best_idx].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)

            # 判断是否需要VLM备份
            need_backup = confidence < self.confidence_threshold

            return {
                'mask': mask,
                'confidence': float(confidence),
                'bbox': result.boxes.xyxy[best_idx].cpu().numpy(),
                'need_vlm_backup': need_backup,
            }

        return {
            'mask': np.zeros(image.shape[:2], dtype=np.uint8),
            'confidence': 0.0,
            'need_vlm_backup': True,
        }

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor,
                   gray_value: float = 0.5) -> torch.Tensor:
        """应用分割掩码（掩码外区域置灰）
        Args:
            image: [batch, 3, H, W] 输入图像
            mask: [batch, 1, H, W] 分割掩码
            gray_value: 背景灰度值
        Returns:
            masked_image: [batch, 3, H, W] 掩码后的图像
        """
        masked_image = image * mask + gray_value * (1 - mask)
        return masked_image


class YOLOSegTrainer:
    """YOLOv8n-seg训练器
    使用ultralytics框架进行训练，配合自定义多尺度损失
    """

    def __init__(
        self,
        model_path: str = 'yolov8n-seg.pt',
        data_yaml: str = 'data.yaml',
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str = '0',
    ):
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.device = device

    def train(self, project: str = 'runs/seg', name: str = 'head_seg'):
        """启动训练
        Args:
            project: 输出目录
            name: 实验名称
        """
        results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            project=project,
            name=name,
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )
        return results

    def export(self, format: str = 'onnx'):
        """导出模型
        Args:
            format: 导出格式 ('onnx', 'tensorrt', 'torchscript')
        """
        self.model.export(format=format)
