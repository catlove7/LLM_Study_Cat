"""
BPE (Byte Pair Encoding) 分词器实现

BPE 是一种子词分词算法，广泛应用于现代语言模型中。
参考：https://arxiv.org/abs/1508.07909
"""

import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# 相对导入改为条件导入，支持直接运行
try:
    from .tokenizer import BaseTokenizer
except ImportError:
    from tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """
    BPE 分词器实现
    
    BPE 通过迭代合并最频繁的字符对来构建词汇表。
    """
    
    def __init__(self):
        super().__init__()
        self.merges: List[Tuple[str, str]] = []  # 记录合并操作
        
    def _get_vocab(self, texts: List[str]) -> Counter:
        """
        获取初始词汇（字符级别）
        
        Args:
            texts: 文本列表
            
        Returns:
            词频计数器
        """
        vocab = Counter()
        for text in texts:
            # 将单词拆分为字符，并在末尾添加特殊符号</w>表示词尾
            words = text.split()
            for word in words:
                # 在字符间添加空格，最后一个字符后加</w>
                chars = list(word[:-1]) + [word[-1] + '</w>']
                word_tuple = tuple(chars)
                vocab[word_tuple] += 1
        return vocab
    
    def _get_pairs(self, vocab: Counter) -> Counter:
        """
        获取所有相邻字符对的频率
        
        Args:
            vocab: 词汇计数器
            
        Returns:
            字符对频率计数器
        """
        pairs = Counter()
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pairs[pair] += freq
        return pairs
    
    def _merge_vocab(self, vocab: Counter, pair: Tuple[str, str]) -> Counter:
        """
        合并词汇表中的字符对
        
        Args:
            vocab: 词汇计数器
            pair: 要合并的字符对
            
        Returns:
            合并后的词汇计数器
        """
        new_vocab = Counter()
        bigram = pair[0] + pair[1]
        
        for word_tuple, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] += freq
        
        return new_vocab
    
    def train(self, texts: List[str], vocab_size: int, num_merges: int = None) -> None:
        """
        训练 BPE 分词器
        
        Args:
            texts: 训练文本列表
            vocab_size: 目标词汇表大小
            num_merges: 合并次数（如果不指定，则根据 vocab_size 自动计算）
        """
        # 获取初始词汇
        vocab = self._get_vocab(texts)
        
        # 获取所有唯一字符作为初始词汇表
        unique_chars = set()
        for word_tuple in vocab.keys():
            unique_chars.update(word_tuple)
        
        # 初始化词汇表
        self.vocab = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        
        # 确定合并次数
        if num_merges is None:
            num_merges = vocab_size - len(self.vocab)
        
        # 迭代合并
        self.merges = []
        for i in range(num_merges):
            # 获取最频繁的字符对
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
                
            best_pair = pairs.most_common(1)[0][0]
            
            # 合并词汇表
            vocab = self._merge_vocab(vocab, best_pair)
            
            # 记录合并操作并添加到词汇表
            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
        
        # 构建反向映射
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        print(f"BPE 训练完成：词汇表大小={len(self.vocab)}, 合并次数={len(self.merges)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        对单个单词进行分词
        
        Args:
            word: 输入单词
            
        Returns:
            token 列表
        """
        # 添加词尾标记
        if not word.endswith('</w>'):
            word = word + '</w>'
        
        # 初始化为字符列表
        tokens = list(word[:-1]) + [word[-1:] if word.endswith('</w>') else word + '</w>']
        if word.endswith('</w>'):
            tokens = list(word[:-4]) + [word[-4:]]
        else:
            tokens = list(word) + ['</w>']
        
        # 应用所有合并操作
        for first, second in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            token IDs 列表
        """
        tokens = []
        words = text.split()
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # 处理未登录词（OOV）- 回退到字符级别
                    for char in token:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
                        else:
                            # 未知字符，使用特殊标记或跳过
                            pass
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将 token IDs 解码为文本
        
        Args:
            token_ids: token IDs 列表
            
        Returns:
            解码后的文本
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<unk>')
        
        # 合并 tokens 并处理词尾标记
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def save(self, path: str) -> None:
        """保存分词器到文件"""
        import json
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
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
        self.merges = [tuple(m) for m in data['merges']]
        self.special_tokens = data.get('special_tokens', {})
        self.id_to_token = {v: k for k, v in self.vocab.items()}


# 使用示例
if __name__ == '__main__':
    # 训练数据
    training_data = [
        "hello world",
        "hello python",
        "world peace",
        "python programming",
        "hello hello world"
    ]
    
    # 创建并训练分词器
    tokenizer = BPETokenizer()
    tokenizer.train(training_data, vocab_size=50)
    
    # 测试编码
    test_text = "hello world"
    encoded = tokenizer.encode(test_text)
    print(f"原文：{test_text}")
    print(f"编码：{encoded}")
    
    # 测试解码
    decoded = tokenizer.decode(encoded)
    print(f"解码：{decoded}")
    
    # 添加特殊标记
    tokenizer.add_special_token('<pad>')
    tokenizer.add_special_token('<bos>')
    tokenizer.add_special_token('<eos>')
    tokenizer.add_special_token('<unk>')
    
    print(f"\n最终词汇表大小：{tokenizer.get_vocab_size()}")
