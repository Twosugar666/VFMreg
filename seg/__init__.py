"""
头部分割模块
1. 基于YOLOv8n-seg的快速轮廓分割
   - 多尺度交叉熵损失函数
   - Sobel边缘增强策略
2. 基于Qwen2.5-VL LoRA微调的语义分割（鲁棒备份）
3. 级联策略：默认YOLOv8n-seg，置信度低于0.85时切换VLM
"""

from .yolo_seg import HeadSegmentor, MultiScaleLoss, SobelEdgeEnhancement
