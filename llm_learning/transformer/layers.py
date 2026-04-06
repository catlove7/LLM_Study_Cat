"""
Transformer 层实现

实现了 Positional Encoding、FeedForward Network 和 TransformerBlock
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码
    
    使用正弦和余弦函数为序列添加位置信息。
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
        dropout: dropout 概率
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算不同维度的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        
        # 调整形状为 (1, max_seq_len, d_model) 以便广播
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，这样不会被当作模型参数
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        # x 的形状：(batch, seq_len, d_model)
        # pe 的形状：(1, max_seq_len, d_model)
        # 截取实际需要的序列长度
        pe = self.pe[:, :x.size(1), :]
        
        x = x + pe
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    两个线性变换中间加一个 ReLU 激活函数。
    
    Args:
        d_model: 模型维度
        d_ff: 前馈网络中间层维度（通常是 d_model 的 4 倍）
        dropout: dropout 概率
    """
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用 GELU 激活函数（现代 LLM 常用）
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            输出张量，形状与输入相同
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    层归一化
    
    对每个样本的特征维度进行归一化。
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            归一化后的张量
        """
        # 计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1)
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和平移
        return self.weight * x_norm + self.bias


class TransformerBlock(nn.Module):
    """
    Transformer 块
    
    包含多头自注意力层和前馈网络，使用残差连接和层归一化。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        dropout: dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        # 多头自注意力
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量
            
        Returns:
            输出张量，形状与输入相同
        """
        # Pre-Norm 架构（现代 LLM 常用）
        # 1. 自注意力子层（带残差连接）
        attention_input = self.norm1(x)
        attention_output = self.attention(attention_input, mask)
        x = x + self.dropout(attention_output)
        
        # 2. 前馈子层（带残差连接）
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x


# 需要在文件顶部导入 MultiHeadAttention
# 为避免循环依赖，在这里延迟导入
try:
    from .attention import MultiHeadAttention
except ImportError:
    from attention import MultiHeadAttention


# 测试代码
if __name__ == '__main__':
    print("Testing PositionalEncoding...")
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    pos_enc = PositionalEncoding(d_model)
    x_pos = pos_enc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_pos.shape}")
    
    print("\nTesting FeedForward...")
    ff = FeedForward(d_model, d_ff=2048)
    x_ff = ff(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_ff.shape}")
    
    print("\nTesting LayerNorm...")
    ln = LayerNorm(d_model)
    x_norm = ln(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_norm.shape}")
    print(f"Mean: {x_norm.mean().item():.6f}, Std: {x_norm.std().item():.6f}")
    
    print("\nTesting TransformerBlock...")
    n_heads = 8
    block = TransformerBlock(d_model, n_heads)
    x_block = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_block.shape}")
    
    # 测试带掩码的情况
    print("\nTesting TransformerBlock with causal mask...")
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    x_masked = block(x, mask)
    print(f"Masked output shape: {x_masked.shape}")
    
    print("\nAll tests passed!")
