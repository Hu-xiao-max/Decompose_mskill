#!/usr/bin/env python3
"""
不带仿真评估的Diffusion Policy训练脚本
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict
import time
from datetime import datetime

# 设置无头模式，确保没有GUI依赖
os.environ['QT_QPA_PLATFORM'] = 'minimal'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.pop('DISPLAY', None)

from data_loader import create_data_loaders
from diffusion_model import create_diffusion_policy


class SimpleTrainer:
    """简单训练器（无仿真评估）"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器 - 使用Adam，因为AdamW可能不可用
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 早停机制
        self.patience = config.get('patience', 20)
        self.patience_counter = 0
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        images = batch['images'].to(self.device)
        robot_states = batch['robot_states'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        batch_size = actions.shape[0]
        
        # 随机采样时间步
        timesteps = torch.randint(
            0, self.model.num_diffusion_steps, 
            (batch_size,), device=self.device
        ).long()
        
        # 为动作添加噪声
        noisy_actions, noise = self.model.add_noise(actions, timesteps)
        
        # 前向传播
        if self.use_amp:
            with autocast():
                predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
                mse_loss = nn.functional.mse_loss(predicted_noise, noise)
        else:
            predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
            mse_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        losses = {'mse_loss': mse_loss, 'total_loss': mse_loss}
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            try:
                losses = self.compute_loss(batch)
                total_loss = losses['total_loss']
                
                # 反向传播
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    if self.config.get('grad_clip_norm', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['grad_clip_norm']
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.config.get('grad_clip_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['grad_clip_norm']
                        )
                    self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                self.global_step += 1
                
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # 限制训练批次数（调试用）
                if (self.config.get('max_train_batches', 0) > 0 and 
                    batch_idx >= self.config['max_train_batches']):
                    break
                    
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    losses = self.compute_loss(batch)
                    val_loss += losses['total_loss'].item()
                    num_batches += 1
                    
                    if (self.config.get('max_val_batches', 0) > 0 and 
                        batch_idx >= self.config['max_val_batches']):
                        break
                        
                except Exception as e:
                    print(f"验证批次 {batch_idx} 出错: {e}")
                    continue
        
        avg_loss = val_loss / max(num_batches, 1)
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，损失: {self.best_val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        print("=" * 60)
        print("开始训练 Diffusion Policy (无仿真评估模式)")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练集: {len(self.train_loader)} 批次")
        print(f"验证集: {len(self.val_loader)} 批次")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])
            
            # 验证
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['val_loss'])
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 检查是否是最佳模型
            is_best = val_metrics['val_loss'] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 计算时间
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            # 打印结果
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']} ({epoch_time:.1f}s)")
            print(f"  训练损失: {train_metrics['train_loss']:.4f}")
            print(f"  验证损失: {val_metrics['val_loss']:.4f}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  耐心计数: {self.patience_counter}/{self.patience}")
            if is_best:
                print("  🎉 新的最佳模型!")
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n早停触发! 验证损失连续 {self.patience} 个epoch没有改善")
                break
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")


def create_simple_config() -> Dict:
    """创建简单训练配置"""
    return {
        # 数据配置
        # 'dataset_path': '/home/alien/Dataset/colosseum_dataset',
        'dataset_path': '/home/alien/simulation/robot-colosseum/dataset/close_box',
        'batch_size': 8,
        'sequence_length': 4,
        'action_horizon': 2,
        'num_workers': 0,
        'image_size': (224, 224),
        'normalize_actions': True,
        'augment_images': True,
        'cameras': ['front_rgb'],
        'image_types': ['rgb'],
        'load_depth': False,
        'load_point_clouds': False,
        'subsample_factor': 1,
        'max_episodes_per_task': None,
        
        # 模型配置
        'action_dim': 7,
        'state_dim': 15,  # 关节位置(7) + 夹爪位姿(7) + 夹爪状态(1) = 15
        'vision_feature_dim': 256,
        'hidden_dim': 256,
        'num_diffusion_steps': 50,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        
        # 训练配置
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        'use_amp': True,
        'patience': 15,
        
        # 保存配置
        'save_dir': '/home/alien/simulation/robot-colosseum/diffusion_policy/my_model',
        
        # 调试配置
        'max_train_batches': 0,  # 0表示不限制
        'max_val_batches': 0,
    }


def main():
    parser = argparse.ArgumentParser(description='简单Diffusion Policy训练（无仿真评估）')
    parser.add_argument('--config', type=str, help='配置文件路径 (JSON)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./my_model', help='保存目录')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_simple_config()
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_path=config['dataset_path'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            action_horizon=config['action_horizon'],
            num_workers=config['num_workers'],
            image_size=config['image_size'],
            normalize_actions=config['normalize_actions'],
            cameras=config['cameras'],
            image_types=config['image_types'],
            load_depth=config['load_depth'],
            load_point_clouds=config['load_point_clouds'],
            subsample_factor=config['subsample_factor'],
            max_episodes_per_task=config['max_episodes_per_task']
        )
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        print(f"错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("创建模型...")
    model = create_diffusion_policy(
        action_dim=config['action_dim'],
        action_horizon=config['action_horizon'],
        state_dim=config['state_dim'],
        vision_feature_dim=config['vision_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_diffusion_steps=config['num_diffusion_steps'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # 开始训练
    trainer.train()



if __name__ == "__main__":
    main()
