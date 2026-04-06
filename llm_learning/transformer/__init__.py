"""
Transformer 模块 - Transformer 架构实现

本模块实现了 Transformer 模型的核心组件，包括：
- SelfAttention: 自注意力机制
- MultiHeadAttention: 多头注意力
- TransformerBlock: Transformer 块
- TransformerModel: 完整的 Transformer 模型
"""

from .attention import SelfAttention, MultiHeadAttention
from .layers import TransformerBlock, PositionalEncoding
from .model import TransformerModel

__all__ = [
    'SelfAttention',
    'MultiHeadAttention', 
    'TransformerBlock',
    'PositionalEncoding',
    'TransformerModel'
]
