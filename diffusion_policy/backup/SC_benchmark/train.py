#!/usr/bin/env python3
"""
不带仿真评估的Diffusion Policy训练脚本 - 支持多相机
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
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
from diffusion_model import create_improved_diffusion_policy


class SimpleTrainer:
    """简单训练器（无仿真评估）- 支持多相机"""
    
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
        
        # 优化器 - 使用AdamW与增强配置
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=config.get('betas', (0.9, 0.999)),  # 更稳定的beta值
            eps=config.get('eps', 1e-8)
        )
        
        # EMA(指数移动平均)支持
        self.use_ema = config.get('use_ema', True)
        if self.use_ema:
            self.ema_decay = config.get('ema_decay', 0.9999)
            self.ema_model = self._create_ema_model()
        
        # 学习率调度器 - 添加预热
        warmup_steps = config.get('warmup_steps', 1000)
        self.warmup_steps = warmup_steps
        self.base_lr = config['learning_rate']
        
        # 使用CosineAnnealingWarmRestarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 50),  # 重启周期
            T_mult=config.get('T_mult', 2),
            eta_min=config['learning_rate'] * 0.001
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
        
        # 日志配置
        self.log_every_n_steps = config.get('log_every_n_steps', 50)
        self.save_every_n_epochs = config.get('save_every_n_epochs', 10)
    
    def _create_ema_model(self) -> nn.Module:
        """创建EMA模型的深拷贝"""
        import copy
        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model
    
    def _update_ema(self):
        """更新EMA模型参数"""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失 - 支持多相机输入"""
        # 处理多相机图像
        images_list = []
        
        # 收集所有相机的图像
        for camera in self.config['cameras']:
            camera_key = f'images_{camera}'
            if camera_key in batch:
                images_list.append(batch[camera_key])
        
        # 根据相机数量处理图像
        if len(images_list) == 0:
            # 向后兼容：如果没有相机特定的键，使用默认的'images'
            if 'images' in batch:
                images = batch['images'].to(self.device)
                # 确保是5维张量 [B, T, C, H, W]
                if len(images.shape) == 5:
                    pass  # 已经是正确格式
                else:
                    raise ValueError(f"图像维度错误: {images.shape}")
            else:
                raise ValueError("批次中没有找到图像数据")
        elif len(images_list) == 1:
            # 单相机情况
            images = images_list[0].to(self.device)
        else:
            # 多相机情况 - 堆叠成 [B, num_cameras, T, C, H, W]
            try:
                # 确保所有图像具有相同的形状
                for i, img in enumerate(images_list):
                    if img.shape != images_list[0].shape:
                        print(f"警告: 相机 {i} 图像形状 {img.shape} 与第一个相机 {images_list[0].shape} 不匹配")
                
                images = torch.stack(images_list, dim=1).to(self.device)
                # print(f"多相机图像形状: {images.shape}")  # 调试信息
            except Exception as e:
                print(f"堆叠图像时出错: {e}")
                print(f"图像列表长度: {len(images_list)}")
                for i, img in enumerate(images_list):
                    print(f"  相机 {i} 形状: {img.shape}")
                raise
        
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
        try:
            if self.use_amp:
                with autocast():
                    predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
                    mse_loss = nn.functional.mse_loss(predicted_noise, noise)
            else:
                predicted_noise = self.model(noisy_actions, timesteps, images, robot_states)
                mse_loss = nn.functional.mse_loss(predicted_noise, noise)
        except Exception as e:
            print(f"模型前向传播错误: {e}")
            print(f"  噪声动作形状: {noisy_actions.shape}")
            print(f"  时间步形状: {timesteps.shape}")
            print(f"  图像形状: {images.shape}")
            print(f"  机器人状态形状: {robot_states.shape}")
            raise
        
        losses = {'mse_loss': mse_loss, 'total_loss': mse_loss}
        return losses
    
    def _update_learning_rate(self):
        """更新学习率（包括预热）"""
        if self.global_step < self.warmup_steps:
            # 预热阶段：线性增长到目标学习率
            lr = self.base_lr * (self.global_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # 预热后使用调度器
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 更新学习率（包括预热）
            self._update_learning_rate()
            
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
                
                # 更新EMA模型
                if self.use_ema:
                    self._update_ema()
                
                # 更详细的日志
                if self.global_step % self.log_every_n_steps == 0:
                    pbar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                        'step': self.global_step
                    })
                else:
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
        """验证 - 支持EMA模型"""
        # 使用EMA模型进行验证如果可用
        eval_model = self.ema_model if self.use_ema else self.model
        eval_model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                try:
                    # 如果使用EMA，临时替换模型进行验证
                    if self.use_ema:
                        original_model = self.model
                        self.model = eval_model
                        losses = self.compute_loss(batch)
                        self.model = original_model
                    else:
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
            # 如果使用EMA，也保存EMA模型
            if self.use_ema:
                ema_checkpoint = checkpoint.copy()
                ema_checkpoint['model_state_dict'] = self.ema_model.state_dict()
                torch.save(ema_checkpoint, os.path.join(self.save_dir, 'best_ema_model.pth'))
            print(f"保存最佳模型，损失: {self.best_val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        print("=" * 60)
        print("开始训练 Diffusion Policy (多相机支持)")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"相机数量: {self.config.get('num_cameras', 1)}")
        print(f"相机列表: {self.config.get('cameras', ['front_rgb'])}")
        print(f"训练集: {len(self.train_loader)} 批次")
        print(f"验证集: {len(self.val_loader)} 批次")
        if self.use_ema:
            print(f"EMA衰减系数: {self.ema_decay}")
        print(f"图像尺寸: {self.config.get('image_size', (224, 224))}")
        print(f"批次大小: {self.config.get('batch_size', 16)}")
        print(f"使用混合精度: {self.use_amp}")
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
            
            # 更新学习率调度器（仅在预热后）
            if self.scheduler is not None and self.global_step >= self.warmup_steps:
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
            should_save = is_best or (epoch + 1) % self.save_every_n_epochs == 0
            if should_save:
                self.save_checkpoint(is_best)
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n早停触发! 验证损失连续 {self.patience} 个epoch没有改善")
                break
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")


def create_optimized_config() -> Dict:
    """创建24GB GPU优化配置 - 针对RLBench任务"""
    return {
        # 数据配置 - 优化用于更大批次
        'dataset_path': '/home/alien/simulation/robot-colosseum/dataset/wipe_desk',
        'batch_size': 32,  # 利用24GB显存增加batch size
        'sequence_length': 8,  # 增加序列长度以提供更多时序信息
        'action_horizon': 4,  # 增加动作预测范围
        'num_workers': 8,  # 利用更多CPU核心
        'image_size': (256, 256),  # 增加图像分辨率以提供更多细节
        'normalize_actions': True,
        'augment_images': True,
        
        # 多相机配置 - RLBench通常使用多视角
        'cameras': ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb'],
        'image_types': ['rgb'],
        'require_all_cameras': False,  # 允许部分相机缺失
        
        'load_depth': False,
        'load_point_clouds': False,
        'subsample_factor': 1,
        'max_episodes_per_task': None,
        
        # 模型配置 - 大幅增加模型容量
        'action_dim': 8,  # 7个关节 + 1个夹爪
        'state_dim': 15,
        'vision_feature_dim': 1024,  # 增加视觉特征维度
        'hidden_dim': 1024,  # 显著增加隐藏维度
        'num_diffusion_steps': 200,  # 增加扩散步数以提高质量
        'num_layers': 12,  # 显著增加层数
        'num_heads': 16,  # 增加注意力头
        'dropout': 0.05,  # 进一步减少dropout防止欠拟合
        'num_cameras': 4,  # 支持更多相机
        'fusion_method': 'attention',  # 使用注意力融合多视角
        'prediction_type': 'epsilon',  # 扩散预测类型
        'clip_range': [-2.0, 2.0],  # 训练与推理一致的动作范围
        
        # 训练配置 - 优化超参数
        'num_epochs': 300,  # 增加训练轮数
        'learning_rate': 2e-5,  # 较小的学习率用于稳定训练
        'weight_decay': 5e-6,  # 较小的weight decay
        'grad_clip_norm': 1.0,  # 更严格的梯度裁剪
        'use_amp': True,  # 使用混合精度节省显存
        'patience': 80,  # 增加耐心值
        
        # 新增高级训练配置
        'warmup_steps': 2000,  # 增加预热步数
        'T_0': 100,  # 余弦重启周期
        'T_mult': 2,
        'betas': (0.9, 0.999),  # 优化器参数
        'eps': 1e-8,
        
        # EMA配置
        'use_ema': True,
        'ema_decay': 0.9999,
        
        # 保存配置
        'save_dir': '/home/alien/simulation/robot-colosseum/diffusion_policy/enhanced_model',
        'save_every_n_epochs': 10,  # 定期保存
        
        # 调试配置
        'max_train_batches': 0,  # 0表示不限制
        'max_val_batches': 0,
        'log_every_n_steps': 50,  # 更频繁的日志
    }


def main():
    parser = argparse.ArgumentParser(description='Diffusion Policy训练（多相机支持）')
    parser.add_argument('--config', type=str, help='配置文件路径 (JSON)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default=f'/home/alien/simulation/robot-colosseum/diffusion_policy/my_model/{time.strftime("%Y%m%d_%H%M%S")}', 
                        help='保存目录')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_optimized_config()  # 使用优化配置
    
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
            max_episodes_per_task=config['max_episodes_per_task'],
            require_all_cameras=config.get('require_all_cameras', True)
        )
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        print(f"错误详情: {str(e)}")
        print("请检查:")
        print(f"1. 数据集路径: {config['dataset_path']}")
        print(f"2. 相机配置: {config['cameras']}")
        print(f"3. 批次大小: {config['batch_size']}")
        import traceback
        traceback.print_exc()
        return
    
    # 将动作归一化统计写入配置，供推理时反归一化使用
    try:
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'action_mean') and hasattr(train_loader.dataset, 'action_std'):
            action_mean = getattr(train_loader.dataset, 'action_mean', None)
            action_std = getattr(train_loader.dataset, 'action_std', None)
            if action_mean is not None and action_std is not None:
                # 转成可序列化的列表
                config['action_mean'] = [float(x) for x in np.array(action_mean).tolist()]
                config['action_std'] = [float(x) for x in np.array(action_std).tolist()]
                print("已将动作归一化统计保存到配置中: action_mean/std")
    except Exception as stats_err:
        print(f"警告: 无法将动作统计保存到配置: {stats_err}")
    
    # 创建模型
    print("创建增强模型...")
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
        fusion_method=config.get('fusion_method', 'attention'),
        prediction_type='epsilon',  # 使用epsilon预测
        clip_range=(-2.0, 2.0),  # RLBench适合的动作范围
        use_ema=config.get('use_ema', True)
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
