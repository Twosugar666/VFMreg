"""
VFMReg端到端视觉配准框架
- 冻结的DINOv3 ViT-L/14作为backbone特征提取器
- 跨视图注意力融合模块（4层Transformer编码器）
- 6D连续旋转表示的几何回归头
- 可微渲染优化模块（训练阶段）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossViewAttention(nn.Module):
    """跨视图注意力融合模块
    将K个视图的特征向量视为长度为K的序列，
    通过Transformer自注意力学习视图间的几何关系和信息互补性
    - L=4层Transformer编码器
    - h=8个注意力头
    - 可学习的视图位置编码
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 2048,
        num_views: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: 输入特征维度 (D=1024 for DINOv3 ViT-L/14)
            num_heads: 注意力头数
            num_layers: Transformer层数
            ffn_dim: FFN隐藏层维度
            num_views: 视图数量K
            dropout: Dropout率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_views = num_views

        # 可学习的视图位置编码
        self.view_pos_embedding = nn.Parameter(
            torch.randn(1, num_views, feature_dim) * 0.02
        )

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出LayerNorm
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, K, D] K个视图的特征向量
        Returns:
            fused: [batch, D] 融合后的全局特征
        """
        batch_size = features.shape[0]

        # 添加视图位置编码
        features = features + self.view_pos_embedding[:, :features.shape[1], :]

        # Transformer自注意力
        attended = self.transformer(features)

        # 平均池化聚合
        fused = self.output_norm(attended.mean(dim=1))

        return fused


class PoseRegressionHead(nn.Module):
    """几何回归头
    3层MLP，输出9维向量：
    - 6D连续旋转表示 (a1, a2) 各3维
    - 3D平移向量 t
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 9),  # 6D rotation + 3D translation
        )

        # 初始化最后一层为小值，使初始预测接近单位变换
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, D] 全局特征
        Returns:
            rot_6d: [batch, 6] 6D旋转表示
            translation: [batch, 3] 平移向量
        """
        output = self.mlp(features)
        rot_6d = output[..., :6]
        translation = output[..., 6:]
        return rot_6d, translation


class VFMReg(nn.Module):
    """VFMReg端到端视觉配准框架
    完整管线：
    1. DINOv3特征提取（冻结backbone）
    2. 跨视图注意力融合
    3. 6D位姿回归
    4. 可微渲染优化（仅训练阶段）
    """

    def __init__(
        self,
        backbone_name: str = 'dinov2_vitl14',
        feature_dim: int = 1024,
        num_views: int = 4,
        num_attention_layers: int = 4,
        num_heads: int = 8,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            backbone_name: 视觉基础模型名称
            feature_dim: 特征维度
            num_views: 输入视图数量K
            num_attention_layers: 注意力层数
            num_heads: 注意力头数
            freeze_backbone: 是否冻结backbone
        """
        super().__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone

        # 视觉基础模型（DINOv3 ViT-L/14）
        # 使用torch.hub加载DINOv2作为替代（DINOv3接口兼容）
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', backbone_name, pretrained=True
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # 跨视图注意力融合模块
        self.cross_view_attention = CrossViewAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            num_views=num_views,
        )

        # 几何回归头
        self.pose_head = PoseRegressionHead(
            input_dim=feature_dim,
            hidden_dim=512,
        )

    @staticmethod
    def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """6D连续旋转表示转旋转矩阵（Gram-Schmidt正交化）
        Args:
            rot_6d: [..., 6] 旋转矩阵前两列
        Returns:
            R: [..., 3, 3] 正交旋转矩阵
        """
        a1 = rot_6d[..., :3]
        a2 = rot_6d[..., 3:6]

        # Gram-Schmidt正交化
        e1 = F.normalize(a1, dim=-1)
        e2 = a2 - (a2 * e1).sum(dim=-1, keepdim=True) * e1
        e2 = F.normalize(e2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        return R

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """提取视觉特征
        Args:
            images: [batch*K, 3, H, W] 输入图像
        Returns:
            features: [batch*K, D] CLS token特征
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        # DINOv2返回CLS token作为全局特征
        if isinstance(features, dict):
            features = features['x_norm_clstoken']

        return features

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> dict:
        """前向推理
        Args:
            images: [batch, K, 3, H, W] K个视图的输入图像
            masks: [batch, K, 1, H, W] 可选的分割掩码
        Returns:
            dict: 包含预测的旋转矩阵和平移向量
        """
        batch_size = images.shape[0]
        K = images.shape[1]

        # 应用分割掩码（掩码外区域置灰）
        if masks is not None:
            gray_value = 0.5
            images = images * masks + gray_value * (1 - masks)

        # 重塑为 [batch*K, 3, H, W]
        images_flat = images.reshape(batch_size * K, *images.shape[2:])

        # 提取特征 [batch*K, D]
        features = self.extract_features(images_flat)

        # 重塑为 [batch, K, D]
        features = features.reshape(batch_size, K, self.feature_dim)

        # 跨视图注意力融合 [batch, D]
        fused_features = self.cross_view_attention(features)

        # 位姿回归
        rot_6d, translation = self.pose_head(fused_features)

        # 6D转旋转矩阵
        rotation_matrix = self.rotation_6d_to_matrix(rot_6d)

        return {
            'rotation_matrix': rotation_matrix,  # [batch, 3, 3]
            'translation': translation,          # [batch, 3]
            'rot_6d': rot_6d,                    # [batch, 6]
            'fused_features': fused_features,    # [batch, D]
        }


class VFMRegLoss(nn.Module):
    """VFMReg训练损失函数
    L_total = α * L_geo + β * L_render_L1 + γ * L_render_IoU
    其中：
    - L_geo = λ_rot * ||R_gt - R_pred||_F + λ_trans * ||t_gt - t_pred||_2
    - L_render_L1 = ||I_render - I_real||_1
    - L_render_IoU = 1 - IoU(M_render, M_real)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        lambda_rot: float = 1.0,
        lambda_trans: float = 0.5,
    ):
        """
        Args:
            alpha: 几何损失权重
            beta: 渲染L1损失权重
            gamma: 轮廓IoU损失权重
            lambda_rot: 旋转损失权重
            lambda_trans: 平移损失权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans

    def geometric_loss(self, R_pred: torch.Tensor, R_gt: torch.Tensor,
                       t_pred: torch.Tensor, t_gt: torch.Tensor) -> dict:
        """几何回归损失
        L_geo = λ_rot * ||R_gt - R_pred||_F + λ_trans * ||t_gt - t_pred||_2
        """
        # 旋转损失（Frobenius范数）
        rot_loss = torch.norm(R_gt - R_pred, p='fro', dim=(-2, -1)).mean()

        # 平移损失（L2范数）
        trans_loss = torch.norm(t_gt - t_pred, p=2, dim=-1).mean()

        geo_loss = self.lambda_rot * rot_loss + self.lambda_trans * trans_loss

        return {
            'geo_loss': geo_loss,
            'rot_loss': rot_loss,
            'trans_loss': trans_loss,
        }

    def render_loss(self, rendered_img: torch.Tensor, target_img: torch.Tensor,
                    rendered_mask: torch.Tensor, target_mask: torch.Tensor) -> dict:
        """可微渲染损失
        L_render = L1(I_render, I_target) + λ_IoU * (1 - IoU(M_render, M_target))
        """
        # L1渲染损失
        l1_loss = F.l1_loss(rendered_img, target_img)

        # IoU轮廓损失
        intersection = (rendered_mask * target_mask).sum(dim=(-2, -1))
        union = rendered_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = (1.0 - iou).mean()

        return {
            'render_l1_loss': l1_loss,
            'render_iou_loss': iou_loss,
        }

    def forward(self, predictions: dict, targets: dict,
                rendered: Optional[dict] = None) -> dict:
        """计算总损失
        Args:
            predictions: 模型预测 {'rotation_matrix', 'translation'}
            targets: 真值标签 {'rotation_matrix', 'translation'}
            rendered: 可微渲染结果 {'image', 'mask', 'target_image', 'target_mask'}
        Returns:
            dict: 各项损失
        """
        # 几何损失
        geo_result = self.geometric_loss(
            predictions['rotation_matrix'], targets['rotation_matrix'],
            predictions['translation'], targets['translation']
        )

        total_loss = self.alpha * geo_result['geo_loss']

        result = {
            'total_loss': total_loss,
            'geo_loss': geo_result['geo_loss'],
            'rot_loss': geo_result['rot_loss'],
            'trans_loss': geo_result['trans_loss'],
        }

        # 可微渲染损失（仅训练阶段）
        if rendered is not None:
            render_result = self.render_loss(
                rendered['image'], rendered['target_image'],
                rendered['mask'], rendered['target_mask']
            )
            total_loss = total_loss + self.beta * render_result['render_l1_loss']
            total_loss = total_loss + self.gamma * render_result['render_iou_loss']

            result['total_loss'] = total_loss
            result['render_l1_loss'] = render_result['render_l1_loss']
            result['render_iou_loss'] = render_result['render_iou_loss']

        return result


def compute_rotation_error(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """计算旋转误差（测地距离，单位：度）
    Args:
        R_pred: [batch, 3, 3] 预测旋转矩阵
        R_gt: [batch, 3, 3] 真值旋转矩阵
    Returns:
        error: [batch] 旋转误差（度）
    """
    # R_diff = R_gt^T @ R_pred
    R_diff = torch.bmm(R_gt.transpose(-1, -2), R_pred)
    # 测地距离 = arccos((tr(R_diff) - 1) / 2)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    trace = torch.clamp(trace, -1.0 + 1e-6, 3.0 - 1e-6)
    angle = torch.acos((trace - 1.0) / 2.0)
    return angle * 180.0 / math.pi


def compute_translation_error(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    """计算平移误差（L2距离，单位：mm）
    Args:
        t_pred: [batch, 3] 预测平移
        t_gt: [batch, 3] 真值平移
    Returns:
        error: [batch] 平移误差
    """
    return torch.norm(t_pred - t_gt, p=2, dim=-1)
