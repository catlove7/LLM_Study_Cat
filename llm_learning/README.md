# LLM 学习路径

本仓库是一个完整的大语言模型（LLM）学习项目，从底层的分词器开始，逐步构建到 Transformer 架构，最后到强化学习（RL）。

## 📚 学习路线

### 1. Tokenizer（分词器）
- **目标**: 理解文本如何被转换为模型可处理的数字序列
- **内容**:
  - 字符级分词
  - 词级分词
  - 子词分词（BPE, WordPiece, SentencePiece）
  - 特殊标记处理
- **实现**: `tokenizer/` 目录

### 2. Transformer 架构
- **目标**: 掌握现代 LLM 的核心架构
- **内容**:
  - Self-Attention 机制
  - Multi-Head Attention
  - Positional Encoding
  - Feed-Forward Networks
  - Layer Normalization
  - Decoder-only 架构（如 GPT）
- **实现**: `transformer/` 目录

### 3. 模型训练
- **目标**: 学习如何训练语言模型
- **内容**:
  - 数据预处理和数据集加载
  - 损失函数（交叉熵）
  - 优化器（AdamW）
  - 学习率调度
  - 训练循环
  - 分布式训练基础
- **实现**: `training/` 目录

### 4. 强化学习（RL）
- **目标**: 理解如何使用 RL 优化语言模型
- **内容**:
  - 策略梯度方法
  - PPO（Proximal Policy Optimization）
  - 奖励建模
  - RLHF（Reinforcement Learning from Human Feedback）
- **实现**: `rl/` 目录

## 📁 项目结构

```
llm_learning/
├── tokenizer/          # 分词器实现
│   ├── __init__.py
│   ├── bpe.py         # BPE 分词器
│   └── tokenizer.py   # 基础分词器类
├── transformer/        # Transformer 架构
│   ├── __init__.py
│   ├── attention.py   # Attention 机制
│   ├── layers.py      # Transformer 层
│   └── model.py       # 完整模型
├── training/           # 训练相关
│   ├── __init__.py
│   ├── dataset.py     # 数据集处理
│   └── trainer.py     # 训练器
├── rl/                 # 强化学习
│   ├── __init__.py
│   ├── ppo.py         # PPO 算法
│   └── reward.py      # 奖励模型
├── examples/           # 示例代码
├── tests/              # 测试代码
├── utils/              # 工具函数
└── README.md
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy tqdm
```

### 使用示例

```python
# 1. 使用分词器
from llm_learning.tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.train(["hello world", "hello python"])
tokens = tokenizer.encode("hello world")
print(tokens)

# 2. 创建模型
from llm_learning.transformer import TransformerModel

model = TransformerModel(
    vocab_size=1000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=512
)

# 3. 训练模型
from llm_learning.training import Trainer

trainer = Trainer(model, learning_rate=1e-4)
trainer.train(dataset)

# 4. 强化学习优化
from llm_learning.rl import PPOTrainer

ppo_trainer = PPOTrainer(model)
ppo_trainer.optimize(policy, reward_model)
```

## 📖 学习资源

### 推荐阅读
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 论文
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT/RLHF

### 代码参考
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy 的简化实现
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 工业级实现

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
