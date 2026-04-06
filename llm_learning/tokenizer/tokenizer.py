"""
基础分词器类 - 定义分词器的基本接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class BaseTokenizer(ABC):
    """
    基础分词器抽象类
    
    所有分词器都应该实现这个接口
    """
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}  # token -> id
        self.id_to_token: Dict[int, str] = {}  # id -> token
        self.special_tokens: Dict[str, int] = {}  # 特殊标记
        
    @abstractmethod
    def train(self, texts: List[str], vocab_size: int) -> None:
        """
        训练分词器
        
        Args:
            texts: 训练文本列表
            vocab_size: 词汇表大小
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            token IDs 列表
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        将 token IDs 解码为文本
        
        Args:
            token_ids: token IDs 列表
            
        Returns:
            解码后的文本
        """
        pass
    
    def add_special_token(self, token: str) -> int:
        """
        添加特殊标记
        
        Args:
            token: 特殊标记字符串
            
        Returns:
            特殊标记的 ID
        """
        if token not in self.special_tokens:
            token_id = len(self.vocab) + len(self.special_tokens)
            self.special_tokens[token] = token_id
        return self.special_tokens[token]
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小（包括特殊标记）"""
        return len(self.vocab) + len(self.special_tokens)
    
    def save(self, path: str) -> None:
        """保存分词器到文件"""
        import json
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """从文件加载分词器"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.special_tokens = data.get('special_tokens', {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}
