"""
Training 模块 - 模型训练相关实现

本模块实现了训练语言模型所需的组件，包括：
- TextDataset: 文本数据集处理
- Trainer: 训练器
"""

from .dataset import TextDataset, create_dataloader
from .trainer import Trainer

__all__ = ['TextDataset', 'create_dataloader', 'Trainer']
