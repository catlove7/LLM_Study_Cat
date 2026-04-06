"""
数据集处理

实现了文本数据集的加载和预处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict


class TextDataset(Dataset):
    """
    文本数据集
    
    将文本数据转换为模型可用的输入格式。
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        max_length: 最大序列长度
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预编码所有文本
        self.encoded_data = self._encode_all()
    
    def _encode_all(self) -> List[torch.Tensor]:
        """编码所有文本"""
        encoded = []
        for text in self.texts:
            token_ids = self.tokenizer.encode(text)
            # 截断或填充到 max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))
            encoded.append(torch.tensor(token_ids, dtype=torch.long))
        return encoded
    
    def __len__(self) -> int:
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.encoded_data[idx]
        
        # 创建 attention mask（非 padding 位置为 1）
        attention_mask = (input_ids != 0).long()
        
        # labels 与 input_ids 相同（语言模型任务）
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: 数据加载 worker 数量
        
    Returns:
        DataLoader
    """
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x])
        }
    )
    return dataloader


# 测试代码
if __name__ == '__main__':
    print("Testing TextDataset...")
    
    # 模拟一个简单的分词器
    class SimpleTokenizer:
        def encode(self, text):
            # 简单地将每个字符转换为 ID
            return [ord(c) % 100 for c in text.replace(' ', '')]
    
    # 创建测试数据
    texts = [
        "hello world",
        "this is a test",
        "python programming",
        "machine learning"
    ]
    
    tokenizer = SimpleTokenizer()
    dataset = TextDataset(texts, tokenizer, max_length=20)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # 创建数据加载器
    dataloader = create_dataloader(texts, tokenizer, batch_size=2, max_length=20)
    
    print(f"\nDataloader created with {len(dataloader)} batches")
    
    for batch in dataloader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        break
    
    print("\nAll tests passed!")
