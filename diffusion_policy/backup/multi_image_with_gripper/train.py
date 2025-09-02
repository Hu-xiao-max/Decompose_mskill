#!/usr/bin/env python3
"""
改进的Diffusion Policy训练脚本 - 针对小数据集优化
主要改进：
1. 使用改进的Diffusion模型
2. 添加EMA（指数移动平均）
3. 改进的损失函数和正则化
4. 更好的训练监控
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import time
from datetime import datetime
import copy
import wandb

# 设置无头模式
os.environ['QT_QPA_PLATFORM'] = 'minimal'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.pop('DISPLAY', None)

from data_loader import create_data_loaders
# 使用改进的模型
from diffusion_model import create_improved_diffusion_policy


class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class ImprovedTrainer:
    """改进的训练器 - 针对小数据集优化"""
    
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
        
        # 使用AdamW优化器（更好的权重衰减）
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=config.get('betas', (0.9, 0.999)),
            eps=1e-8
        )
        
        # 改进的学习率调度器 - 使用OneCycleLR
        total_steps = len(train_loader) * config['num_epochs']
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,  # 10%用于warmup
            anneal_strategy='cos',
            final_div_factor=100.0
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        # EMA
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema = EMA(self.model, decay=config.get('ema_decay', 0.999))
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 监控指标
        self.metrics = {
            'grad_norm': [],
            'action_range': [],
            'noise_scale': []
        }
        
        # 早停机制
        self.patience = config.get('patience', 30)
        self.patience_counter = 0
        self.min_delta = config.get('min_delta', 1e-4)
        
        # 创建保存目录
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Wandb日志（可选）
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project="diffusion_policy",
                name=config.get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=config
            )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失 - 添加正则化和监控"""
        # 处理多相机图像
        images_list = []
        for camera in self.config['cameras']:
            camera_key = f'images_{camera}'
            if camera_key in batch:
                images_list.append(batch[camera_key])
        
        # 处理图像
        if len(images_list) == 0 and 'images' in batch:
            images = batch['images'].to(self.device)
        elif len(images_list) == 1:
            images = images_list[0].to(self.device)
        else:
            images = torch.stack(images_list, dim=1).to(self.device)
        
        robot_states = batch['robot_states'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        batch_size = actions.shape[0]
        
        # 监控动作范围
        action_min = actions.min().item()
        action_max = actions.max().item()
        self.metrics['action_range'].append((action_min, action_max))
        
        # 检查异常值
        if abs(action_min) > 100 or abs(action_max) > 100:
            print(f"警告: 检测到异常动作值 [{action_min:.2f}, {action_max:.2f}]")
            # 裁剪异常值
            actions = torch.clamp(actions, -10, 10)
        
        # 随机采样时间步 - 使用重要性采样
        if self.config.get('importance_sampling', True):
            # 对早期时间步给予更高权重（它们更难学习）
            weights = 1.0 / (torch.arange(self.model.num_diffusion_steps, device=self.device) + 10)
            weights = weights / weights.sum()
            timesteps = torch.multinomial(weights, batch_size, replacement=True)
        else:
            timesteps = torch.randint(
                0, self.model.num_diffusion_steps, 
                (batch_size,), device=self.device
            ).long()
        
        # 为动作添加噪声
        noisy_actions, noise = self.model.add_noise(actions, timesteps)
        
        # 监控噪声规模
        noise_scale = noise.std().item()
        self.metrics['noise_scale'].append(noise_scale)
        
        # 前向传播
        try:
            if self.use_amp:
                with autocast():
                    predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
                    
                    # 主损失 - MSE
                    mse_loss = nn.functional.mse_loss(predicted_noise, noise)
                    
                    # 添加L2正则化（防止预测过大的噪声）
                    l2_reg = self.config.get('l2_lambda', 0.001) * torch.mean(predicted_noise ** 2)
                    
                    # 添加平滑正则化（鼓励时间上的平滑性）
                    if predicted_noise.shape[1] > 1:
                        smooth_reg = self.config.get('smooth_lambda', 0.001) * torch.mean(
                            (predicted_noise[:, 1:] - predicted_noise[:, :-1]) ** 2
                        )
                    else:
                        smooth_reg = 0.0
                    
                    total_loss = mse_loss + l2_reg + smooth_reg
            else:
                predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
                mse_loss = nn.functional.mse_loss(predicted_noise, noise)
                l2_reg = self.config.get('l2_lambda', 0.001) * torch.mean(predicted_noise ** 2)
                
                if predicted_noise.shape[1] > 1:
                    smooth_reg = self.config.get('smooth_lambda', 0.001) * torch.mean(
                        (predicted_noise[:, 1:] - predicted_noise[:, :-1]) ** 2
                    )
                else:
                    smooth_reg = 0.0
                
                total_loss = mse_loss + l2_reg + smooth_reg
                
        except Exception as e:
            print(f"模型前向传播错误: {e}")
            raise
        
        losses = {
            'mse_loss': mse_loss,
            'l2_reg': l2_reg if isinstance(l2_reg, torch.Tensor) else torch.tensor(l2_reg),
            'smooth_reg': smooth_reg if isinstance(smooth_reg, torch.Tensor) else torch.tensor(smooth_reg),
            'total_loss': total_loss
        }
        
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'mse': 0.0, 'l2': 0.0, 'smooth': 0.0, 'total': 0.0}
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
                    
                    # 梯度裁剪前先unscale
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip_norm']
                    )
                    self.optimizer.step()
                
                # 更新学习率
                self.scheduler.step()
                
                # 更新EMA
                if self.use_ema:
                    self.ema.update()
                
                # 记录梯度范数
                self.metrics['grad_norm'].append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                
                # 累积损失
                epoch_losses['mse'] += losses['mse_loss'].item()
                epoch_losses['l2'] += losses['l2_reg'].item()
                epoch_losses['smooth'] += losses['smooth_reg'].item()
                epoch_losses['total'] += total_loss.item()
                num_batches += 1
                self.global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    'grad': f"{grad_norm:.2f}"
                })
                
                # Wandb日志
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': total_loss.item(),
                        'train/mse_loss': losses['mse_loss'].item(),
                        'train/grad_norm': grad_norm,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'step': self.global_step
                    })
                
                # 限制训练批次数（调试用）
                if (self.config.get('max_train_batches', 0) > 0 and 
                    batch_idx >= self.config['max_train_batches']):
                    break
                    
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证 - 使用EMA模型"""
        # 如果使用EMA，应用EMA权重
        if self.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        val_losses = {'mse': 0.0, 'total': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    losses = self.compute_loss(batch)
                    val_losses['mse'] += losses['mse_loss'].item()
                    val_losses['total'] += losses['total_loss'].item()
                    num_batches += 1
                    
                    if (self.config.get('max_val_batches', 0) > 0 and 
                        batch_idx >= self.config['max_val_batches']):
                        break
                        
                except Exception as e:
                    print(f"验证批次 {batch_idx} 出错: {e}")
                    continue
        
        # 恢复原始权重
        if self.use_ema:
            self.ema.restore()
        
        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': self.metrics,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.use_ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            
            # 如果使用EMA，也保存EMA模型
            if self.use_ema:
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), 
                          os.path.join(self.save_dir, 'best_model_ema.pth'))
                self.ema.restore()
            
            print(f"✅ 保存最佳模型，损失: {self.best_val_loss:.4f}")
    
    def print_diagnostics(self):
        """打印诊断信息"""
        if self.metrics['grad_norm']:
            avg_grad = np.mean(self.metrics['grad_norm'][-100:])
            print(f"  平均梯度范数: {avg_grad:.3f}")
        
        if self.metrics['action_range']:
            recent_ranges = self.metrics['action_range'][-10:]
            min_vals = [r[0] for r in recent_ranges]
            max_vals = [r[1] for r in recent_ranges]
            print(f"  动作范围: [{np.mean(min_vals):.2f}, {np.mean(max_vals):.2f}]")
        
        if self.metrics['noise_scale']:
            avg_noise = np.mean(self.metrics['noise_scale'][-100:])
            print(f"  平均噪声规模: {avg_noise:.3f}")
    
    def train(self):
        """主训练循环"""
        print("=" * 60)
        print("🚀 开始训练改进的Diffusion Policy")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"扩散步数: {self.model.num_diffusion_steps}")
        print(f"使用EMA: {self.use_ema}")
        print(f"训练集: {len(self.train_loader)} 批次")
        print(f"验证集: {len(self.val_loader)} 批次")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['num_epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['total'])
            
            # 验证
            val_losses = self.validate()
            self.val_losses.append(val_losses['total'])
            
            # 检查是否是最佳模型
            is_best = False
            if val_losses['total'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            # 计算时间
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            print(f"\n📊 Epoch {epoch + 1}/{self.config['num_epochs']} ({epoch_time:.1f}s)")
            print(f"  训练损失: {train_losses['total']:.4f} (MSE: {train_losses['mse']:.4f})")
            print(f"  验证损失: {val_losses['total']:.4f} (MSE: {val_losses['mse']:.4f})")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  耐心计数: {self.patience_counter}/{self.patience}")
            
            # 打印诊断信息
            self.print_diagnostics()
            
            if is_best:
                print("  🎉 新的最佳模型!")
            
            # Wandb日志
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_losses['total'],
                    'val/epoch_loss': val_losses['total'],
                    'val/best_loss': self.best_val_loss,
                    'patience': self.patience_counter
                })
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ 早停触发! 验证损失连续 {self.patience} 个epoch没有改善")
                break
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成! 总时间: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()


def create_improved_config() -> Dict:
    """创建改进的训练配置 - 针对小数据集"""
    return {
        # 数据配置
        'dataset_path': '/home/alien/simulation/robot-colosseum/dataset/wipe_desk_poor',
        'batch_size': 4,  # 减小批次大小
        'sequence_length': 4,
        'action_horizon': 2,  # 减小预测长度
        'num_workers': 0,
        'image_size': (224, 224),
        'normalize_actions': True,
        'augment_images': True,
        
        # 多相机配置
        'cameras': ['front_rgb'],  # 先用单相机测试
        'image_types': ['rgb'],
        'require_all_cameras': False,
        
        # 模型配置 - 关键改进
        'action_dim': 8,
        'state_dim': 15,
        'vision_feature_dim': 256,  # 减小
        'hidden_dim': 256,  # 减小
        'num_diffusion_steps': 50,  # 大幅减少！
        'num_layers': 3,  # 减少层数
        'num_heads': 4,
        'dropout': 0.2,  # 增加dropout
        'num_cameras': 1,
        'fusion_method': 'attention',
        'clip_range': (-5.0, 5.0),  # 动作裁剪范围
        'prediction_type': 'epsilon',  # 或 'v_prediction'
        
        # 训练配置 - 优化
        'num_epochs': 100,  # 增加epochs
        'learning_rate': 5e-5,  # 减小学习率
        'weight_decay': 1e-4,
        'grad_clip_norm': 1.0,
        'use_amp': True,
        'patience': 30,
        'min_delta': 1e-4,
        
        # EMA配置
        'use_ema': True,
        'ema_decay': 0.999,
        
        # 正则化
        'l2_lambda': 0.001,
        'smooth_lambda': 0.001,
        'importance_sampling': True,
        
        # 保存配置
        'save_dir': './improved_diffusion_model',
        
        # 日志
        'use_wandb': False,
        'run_name': 'improved_diffusion',
        
        # 调试配置
        'max_train_batches': 0,
        'max_val_batches': 0,
    }


def main():
    parser = argparse.ArgumentParser(description='改进的Diffusion Policy训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集路径')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save_dir', type=str, default='./improved_model')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_improved_config()
    
    # 命令行参数覆盖
    if args.dataset:
        config['dataset_path'] = args.dataset
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    # 调试模式
    if args.debug:
        config['max_train_batches'] = 5
        config['max_val_batches'] = 2
        config['num_epochs'] = 2
        print("⚠️ 调试模式启用")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("📦 创建数据加载器...")
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
            require_all_cameras=config.get('require_all_cameras', False),
            augment_images=config['augment_images']
        )
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        return
    
    # 创建改进的模型
    print("🤖 创建改进的模型...")
    model = create_improved_diffusion_policy(
        action_dim=config['action_dim'],
        action_horizon=config['action_horizon'],
        state_dim=config['state_dim'],
        vision_feature_dim=config['vision_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_diffusion_steps=config['num_diffusion_steps'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_cameras=config.get('num_cameras', 1),
        clip_range=tuple(config.get('clip_range', (-5.0, 5.0))),
        prediction_type=config.get('prediction_type', 'epsilon')
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = ImprovedTrainer(
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