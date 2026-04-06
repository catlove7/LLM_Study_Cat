"""
完整的 Transformer 模型实现

实现了 Decoder-only 架构的 Transformer 模型（类似 GPT）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import TransformerBlock, PositionalEncoding
    from .attention import MultiHeadAttention
except ImportError:
    from layers import TransformerBlock, PositionalEncoding
    from attention import MultiHeadAttention


class TransformerModel(nn.Module):
    """
    Transformer 模型（Decoder-only 架构）
    
    类似于 GPT 系列的架构，使用 decoder-only 的 transformer。
    
    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度（嵌入维度）
        n_heads: 注意力头数
        n_layers: Transformer 块数量
        max_seq_len: 最大序列长度
        d_ff: 前馈网络中间层维度（默认为 4 * d_model）
        dropout: dropout 概率
        pad_token_id: padding token 的 ID
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 512,
        d_ff: int = None,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 输出层（语言模型头）
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（防止看到未来位置）
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            因果掩码张量，形状为 (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)，1 表示有效位置，0 表示 padding
            
        Returns:
            logits，形状为 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token 嵌入
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 创建因果掩码
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # 如果有 attention_mask，合并到 causal_mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * extended_mask
        
        # 通过所有 transformer 块
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask)
        
        # 最终层归一化
        x = self.final_norm(x)
        
        # 输出 logits
        logits = self.output_layer(x)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        自回归生成文本
        
        Args:
            input_ids: 初始输入 token IDs，形状为 (batch_size, seq_len)
            max_length: 最大生成长度
            temperature: 采样温度（越高越随机）
            top_k: top-k 采样
            top_p: nucleus sampling (top-p)
            eos_token_id: 结束标记 ID
            
        Returns:
            生成的 token IDs，形状为 (batch_size, generated_length)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取最后位置的输出
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :] / temperature  # (batch, vocab_size)
                
                # Top-k 采样
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) 采样
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率 > top_p 的 tokens
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样下一个 token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                
                # 拼接到生成的序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 检查是否所有序列都生成了 EOS
                if eos_token_id is not None:
                    if (generated == eos_token_id).all():
                        break
        
        return generated
    
    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 测试代码
if __name__ == '__main__':
    print("Testing TransformerModel...")
    
    # 创建模型
    vocab_size = 1000
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=4,
        max_seq_len=128
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 测试生成
    print("\nTesting text generation...")
    generated = model.generate(
        input_ids[:, :5],  # 使用前 5 个 token 作为 prompt
        max_length=20,
        temperature=0.8,
        top_k=50
    )
    print(f"Generated shape: {generated.shape}")
    
    print("\nAll tests passed!")
