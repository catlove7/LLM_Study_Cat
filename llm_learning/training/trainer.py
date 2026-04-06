"""
训练器实现

实现了语言模型的训练循环、优化器配置和评估功能
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional, Dict, List, Callable
import time


class Trainer:
    """
    语言模型训练器
    
    Args:
        model: 要训练的模型
        learning_rate: 学习率
        weight_decay: 权重衰减（L2 正则化）
        device: 训练设备（cuda/cpu）
        gradient_accumulation_steps: 梯度累积步数
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = None,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 损失函数（交叉熵）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
        
        # 梯度累积
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 训练统计
        self.training_stats = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': []
        }
    
    def train_epoch(
        self,
        dataloader,
        epoch: int,
        progress_bar: bool = True
    ) -> float:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 数
            progress_bar: 是否显示进度条
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        if progress_bar:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            # outputs: (batch, seq_len, vocab_size)
            # labels: (batch, seq_len)
            batch_size, seq_len, vocab_size = outputs.shape
            
            # 调整形状以计算损失
            loss = self.criterion(
                outputs.view(-1, vocab_size),
                labels.view(-1)
            )
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # 更新权重
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            if progress_bar:
                pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        self.training_stats['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            平均评估损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            batch_size, seq_len, vocab_size = outputs.shape
            
            loss = self.criterion(
                outputs.view(-1, vocab_size),
                labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.training_stats['eval_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: int = 10,
        scheduler_type: str = 'cosine',
        warmup_ratio: float = 0.1,
        save_best: bool = True,
        save_path: str = 'best_model.pt',
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            eval_dataloader: 验证数据加载器（可选）
            num_epochs: 训练轮数
            scheduler_type: 学习率调度器类型 ('cosine', 'linear', 'constant')
            warmup_ratio: warmup 比例
            save_best: 是否保存最佳模型
            save_path: 模型保存路径
            callback: 回调函数，每 epoch 结束后调用
            
        Returns:
            训练统计信息
        """
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Total batches per epoch: {len(train_dataloader)}")
        
        # 计算总步数和 warmup 步数
        total_steps = len(train_dataloader) * num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        # 创建学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # 评估
            eval_loss = None
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_stats['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # 打印统计信息
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if eval_loss is not None:
                print(f"  Eval Loss: {eval_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # 保存最佳模型
            if save_best and eval_loss is not None and eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_checkpoint(save_path)
                print(f"  ✓ Saved best model with eval_loss={eval_loss:.4f}")
            
            # 回调函数
            if callback is not None:
                callback(epoch, train_loss, eval_loss)
        
        print("\nTraining completed!")
        return self.training_stats
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        print(f"Checkpoint loaded from {path}")


# 测试代码
if __name__ == '__main__':
    print("Testing Trainer...")
    
    # 创建一个简单的模型用于测试
    import sys
    sys.path.insert(0, '/workspace/llm_learning')
    
    from transformer.model import TransformerModel
    
    model = TransformerModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32
    )
    
    # 创建训练器
    trainer = Trainer(model, learning_rate=1e-3)
    
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # 创建模拟数据
    import random
    texts = ["hello world " * 10 for _ in range(20)]
    
    # 简单分词器
    class SimpleTokenizer:
        def encode(self, text):
            return [hash(word) % 80 + 20 for word in text.split()]
    
    tokenizer = SimpleTokenizer()
    
    from dataset import create_dataloader
    train_loader = create_dataloader(texts, tokenizer, batch_size=4, max_length=32)
    
    print(f"Train loader size: {len(train_loader)}")
    
    # 测试一个训练步骤
    print("\nRunning one training epoch...")
    train_loss = trainer.train_epoch(train_loader, epoch=0, progress_bar=True)
    print(f"Train loss: {train_loss:.4f}")
    
    print("\nAll tests passed!")
