"""
RL 模块 - 强化学习实现

本模块实现了用于语言模型优化的强化学习算法，包括：
- PPOTrainer: Proximal Policy Optimization (PPO) 训练器
- RewardModel: 奖励模型
"""

from .ppo import PPOTrainer
from .reward import RewardModel

__all__ = ['PPOTrainer', 'RewardModel']
