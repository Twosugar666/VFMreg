"""
训练模块
1. NeRF训练脚本
2. VFMReg训练脚本（两阶段：合成预训练 + 真实域微调）
3. YOLOv8n-seg分割训练
"""

from .train_nerf import train_nerf
from .train_vfmreg import train_vfmreg
