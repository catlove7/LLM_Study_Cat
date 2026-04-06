"""
PPO (Proximal Policy Optimization) 实现

实现了用于语言模型优化的 PPO 算法
参考论文：https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional, Dict, List, Tuple
import numpy as np


class PPOTrainer:
    """
    PPO 训练器
    
    使用 PPO 算法优化语言模型策略。
    
    Args:
        policy_model: 策略模型（要优化的语言模型）
        ref_model: 参考模型（通常是 SFT 后的模型，用于 KL 散度计算）
        learning_rate: 学习率
        ppo_epochs: PPO 更新轮数
        clip_epsilon: PPO 裁剪参数
        kl_coef: KL 散度系数
        device: 训练设备
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.1,
        device: str = None
    ):
        self.policy_model = policy_model
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.policy_model.to(self.device)
        
        # 参考模型（用于 KL 散度约束）
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # PPO 参数
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        
        # 优化器
        self.optimizer = AdamW(
            policy_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95)
        )
        
        # 价值网络（Critic）- 用于估计状态价值
        self.value_head = nn.Linear(policy_model.d_model, 1)
        self.value_head.to(self.device)
        self.value_optimizer = AdamW(self.value_head.parameters(), lr=learning_rate)
        
        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'rewards': [],
            'kl_divergence': []
        }
    
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            model: 模型
            input_ids: 输入 token IDs
            actions: 采取的动作（生成的 token）
            attention_mask: 注意力掩码
            
        Returns:
            对数概率
        """
        logits = model(input_ids, attention_mask)
        
        # 获取对应动作的 logits
        batch_size, seq_len, vocab_size = logits.shape
        
        # 选择每个位置对应的动作的 logit
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集动作的对数概率
        action_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        return action_log_probs
    
    def compute_rewards(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算带 KL 惩罚的奖励
        
        Args:
            rewards: 原始奖励
            log_probs: 当前策略的对数概率
            ref_log_probs: 参考策略的对数概率
            
        Returns:
            调整后的奖励和 KL 散度
        """
        # 计算 KL 散度
        kl_div = ref_log_probs - log_probs
        
        # 带 KL 惩罚的奖励
        adjusted_rewards = rewards + self.kl_coef * kl_div
        
        return adjusted_rewards, kl_div.mean()
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95
    ) -> torch.Tensor:
        """
        使用 GAE 计算优势函数
        
        Args:
            rewards: 奖励，形状为 (batch_size, seq_len)
            values: 状态价值估计，形状为 (batch_size, seq_len)
            gamma: 折扣因子
            lam: GAE 参数
            
        Returns:
            优势函数，形状为 (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, 1, device=rewards.device)
        
        # 添加最后一个时间步的价值估计（假设为 0）
        next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)
        
        for t in reversed(range(seq_len)):
            # 计算 TD error
            delta = rewards[:, t] + gamma * next_values[:, t] - values[:, t]
            gae = delta.unsqueeze(1) + gamma * lam * gae
            advantages[:, t] = gae.squeeze(1)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train_step(
        self,
        queries: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        执行一步 PPO 训练
        
        Args:
            queries: 查询/提示
            actions: 采取的动作（生成的 token）
            rewards: 奖励
            attention_mask: 注意力掩码
            
        Returns:
            训练统计信息
        """
        self.policy_model.train()
        self.value_head.train()
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'reward_mean': rewards.mean().item(),
            'kl_divergence': 0.0
        }
        
        # 构建完整序列
        input_ids = torch.cat([queries, actions], dim=1)
        
        with torch.no_grad():
            # 计算旧策略的对数概率
            old_log_probs = self.compute_log_probs(
                self.policy_model, input_ids, actions, attention_mask
            )
            
            # 计算参考模型的对数概率（如果有）
            if self.ref_model is not None:
                ref_log_probs = self.compute_log_probs(
                    self.ref_model, input_ids, actions, attention_mask
                )
            else:
                ref_log_probs = torch.zeros_like(old_log_probs)
            
            # 计算带 KL 惩罚的奖励
            adjusted_rewards, kl_div = self.compute_rewards(
                rewards, old_log_probs, ref_log_probs
            )
            stats['kl_divergence'] = kl_div.item()
            
            # 计算价值估计（只对 action 部分）
            outputs = self.policy_model(input_ids)
            # outputs 是 logits，需要获取 last hidden state
            # 简单起见，我们使用 token embedding 作为 hidden state
            hidden_states = self.policy_model.token_embedding(input_ids)
            all_values = self.value_head(hidden_states).squeeze(-1)
            values = all_values[:, query_len:query_len + action_len]  # 只取 action 部分的价值
            
            # 计算优势（在 no_grad 中计算，因为 advantage 不需要梯度）
            with torch.no_grad():
                advantages = self.compute_advantages(adjusted_rewards, values.detach())
        
        # PPO 更新
        for _ in range(self.ppo_epochs):
            # 计算新策略的对数概率
            new_log_probs = self.compute_log_probs(
                self.policy_model, input_ids, actions, attention_mask
            )
            
            # 计算重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 更新策略
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新价值网络
            value_loss = F.mse_loss(values, adjusted_rewards)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            stats['policy_loss'] += policy_loss.item() / self.ppo_epochs
            stats['value_loss'] += value_loss.item() / self.ppo_epochs
        
        # 记录统计信息
        for key in ['policy_loss', 'value_loss', 'rewards', 'kl_divergence']:
            if key == 'rewards':
                self.training_stats[key].append(stats['reward_mean'])
            else:
                self.training_stats[key].append(stats[key.replace('_divergence', '_divergence')])
        
        return stats
    
    def generate_responses(
        self,
        queries: List[str],
        tokenizer,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> List[str]:
        """
        生成响应
        
        Args:
            queries: 查询列表
            tokenizer: 分词器
            max_length: 最大长度
            temperature: 温度
            
        Returns:
            生成的响应列表
        """
        self.policy_model.eval()
        responses = []
        
        with torch.no_grad():
            for query in queries:
                # 编码查询
                query_ids = tokenizer.encode(query)
                query_tensor = torch.tensor([query_ids], dtype=torch.long, device=self.device)
                
                # 生成响应
                generated = self.policy_model.generate(
                    query_tensor,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # 解码
                response_ids = generated[0, len(query_ids):].tolist()
                response = tokenizer.decode(response_ids)
                responses.append(response)
        
        return responses
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'policy_state_dict': self.policy_model.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        if self.ref_model is not None:
            checkpoint['ref_model_state_dict'] = self.ref_model.state_dict()
        
        torch.save(checkpoint, path)
        print(f"PPO checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        if self.ref_model is not None and 'ref_model_state_dict' in checkpoint:
            self.ref_model.load_state_dict(checkpoint['ref_model_state_dict'])
        
        print(f"PPO checkpoint loaded from {path}")


# 测试代码
if __name__ == '__main__':
    print("Testing PPOTrainer...")
    
    import sys
    sys.path.insert(0, '/workspace/llm_learning')
    
    from transformer.model import TransformerModel
    
    # 创建策略模型和参考模型
    vocab_size = 100
    policy_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )
    
    ref_model = TransformerModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )
    
    # 创建 PPO 训练器
    trainer = PPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        learning_rate=1e-4,
        ppo_epochs=2
    )
    
    print(f"Device: {trainer.device}")
    print(f"Policy parameters: {policy_model.count_parameters():,}")
    
    # 模拟数据
    batch_size = 2
    query_len = 5
    action_len = 10
    
    queries = torch.randint(1, vocab_size, (batch_size, query_len))
    actions = torch.randint(1, vocab_size, (batch_size, action_len))
    rewards = torch.randn(batch_size, action_len)
    
    print(f"\nQueries shape: {queries.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # 执行训练步骤
    print("\nRunning PPO training step...")
    stats = trainer.train_step(queries, actions, rewards)
    
    print(f"\nTraining stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nPPO test passed!")
