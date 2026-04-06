"""
注意力机制实现

实现了 Self-Attention 和 Multi-Head Attention
参考论文：https://arxiv.org/abs/1706.03762
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    自注意力机制
    
    计算输入序列的自注意力权重，捕捉序列内部的依赖关系。
    
    Args:
        d_model: 模型维度（嵌入维度）
        dropout: dropout 概率
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Q, K, V 线性变换
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量，形状为 (batch_size, 1, seq_len) 或 (batch_size, seq_len, seq_len)
            
        Returns:
            注意力输出，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.query_linear(x)  # (batch, seq_len, d_model)
        K = self.key_linear(x)    # (batch, seq_len, d_model)
        V = self.value_linear(x)  # (batch, seq_len, d_model)
        
        # 计算注意力分数：Q * K^T / sqrt(d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和：attention * V
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_model)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将注意力机制分成多个头，每个头学习不同的表示子空间。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: dropout 概率
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性变换
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一个维度拆分为 (n_heads, d_k)
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            
        Returns:
            拆分后的张量，形状为 (batch_size, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个头合并
        
        Args:
            x: 输入张量，形状为 (batch_size, n_heads, seq_len, d_k)
            
        Returns:
            合并后的张量，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_k)
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 掩码张量，形状为 (batch_size, 1, 1, seq_len) 或可广播的形状
            
        Returns:
            注意力输出，形状为 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V 并拆分到多个头
        Q = self._split_heads(self.query_linear(x))  # (batch, n_heads, seq_len, d_k)
        K = self._split_heads(self.key_linear(x))
        V = self._split_heads(self.value_linear(x))
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, n_heads, seq_len, d_k)
        
        # 合并多头输出
        output = self._combine_heads(output)  # (batch, seq_len, d_model)
        
        # 最终线性变换
        output = self.output_linear(output)
        
        return output


# 测试代码
if __name__ == '__main__':
    # 测试 SelfAttention
    print("Testing SelfAttention...")
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    self_attn = SelfAttention(d_model)
    output = self_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试 MultiHeadAttention
    print("\nTesting MultiHeadAttention...")
    n_heads = 8
    multi_attn = MultiHeadAttention(d_model, n_heads)
    output = multi_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试带掩码的情况（causal mask）
    print("\nTesting with causal mask...")
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output_masked = multi_attn(x, mask)
    print(f"Masked output shape: {output_masked.shape}")
    
    print("\nAll tests passed!")
