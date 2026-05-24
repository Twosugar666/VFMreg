"""
数据预处理模块
1. 合成数据生成管线（Blender渲染）
2. 真实数据预处理（COLMAP位姿估计）
3. 数据增强策略
4. 数据集类定义
"""

from .dataset import MEGHeadDataset, SyntheticDataset
from .augmentation import DomainRandomization
