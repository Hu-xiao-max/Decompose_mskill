#!/usr/bin/env python3
"""
脚本用于检查训练数据集的结构和内容
"""

import sys
import os
sys.path.append('/home/alien/simulation/robot-colosseum/diffusion_policy')

from data_loader import create_data_loaders
import torch
import numpy as np

def examine_dataset():
    """检查数据集内容"""
    
    # 使用与train.py相同的配置
    config = {
        'dataset_path': '/home/alien/simulation/robot-colosseum/dataset/close_box_full',
        'batch_size': 2,  # 小批次用于检查
        'sequence_length': 4,
        'action_horizon': 2,
        'num_workers': 0,  # 避免多进程问题
        'image_size': (224, 224),
        'normalize_actions': True,
        'augment_images': False,  # 关闭增强以看原始数据
        'cameras': ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb'],
        'image_types': ['rgb'],
        'require_all_cameras': True,
        'load_depth': False,
        'load_point_clouds': False,
        'subsample_factor': 1,
        'max_episodes_per_task': 2,  # 限制episode数量
    }
    
    print("=" * 60)
    print("数据集结构检查")
    print("=" * 60)
    
    try:
        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader = create_data_loaders(
            dataset_path=config['dataset_path'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            action_horizon=config['action_horizon'],
            num_workers=config['num_workers'],
            cameras=config['cameras'],
            require_all_cameras=config['require_all_cameras'],
            max_episodes_per_task=config['max_episodes_per_task']
        )
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 检查第一个训练批次
        print("\n" + "=" * 40)
        print("训练数据批次内容:")
        print("=" * 40)
        
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n批次 {batch_idx + 1}:")
            print("-" * 30)
            
            # 基本信息
            print(f"任务名称: {batch['task_name']}")
            print(f"Episode索引: {batch['episode_idx']}")
            print(f"起始索引: {batch['start_idx']}")
            
            # 机器人状态
            robot_states = batch['robot_states']
            print(f"\n机器人状态 (robot_states):")
            print(f"  形状: {robot_states.shape}")
            print(f"  数据类型: {robot_states.dtype}")
            print(f"  数值范围: [{robot_states.min():.4f}, {robot_states.max():.4f}]")
            print(f"  第一个样本第一帧状态:")
            print(f"    关节位置(前7维): {robot_states[0, 0, :7].numpy()}")
            print(f"    夹爪姿态(7-14维): {robot_states[0, 0, 7:14].numpy()}")
            print(f"    夹爪开合(第15维): {robot_states[0, 0, 14].item():.4f}")
            
            # 动作
            actions = batch['actions']
            print(f"\n动作 (actions):")
            print(f"  形状: {actions.shape}")
            print(f"  数据类型: {actions.dtype}")
            print(f"  数值范围: [{actions.min():.4f}, {actions.max():.4f}]")
            print(f"  第一个样本动作序列:")
            for t in range(actions.shape[1]):
                action = actions[0, t]
                print(f"    时刻{t}: 关节动作{action[:7].numpy()}, 夹爪{action[7].item():.4f}")
            
            # 相机图像
            cameras = config['cameras']
            for camera in cameras:
                key = f'images_{camera}'
                if key in batch:
                    images = batch[key]
                    print(f"\n{camera} 图像:")
                    print(f"  形状: {images.shape}")  # [batch, seq_len, channels, height, width]
                    print(f"  数据类型: {images.dtype}")
                    print(f"  数值范围: [{images.min():.4f}, {images.max():.4f}]")
                else:
                    print(f"\n{camera} 图像: 不存在")
            
            # 兼容性图像
            if 'images' in batch:
                images = batch['images']
                print(f"\n默认图像 (images):")
                print(f"  形状: {images.shape}")
                print(f"  数据类型: {images.dtype}")
                print(f"  数值范围: [{images.min():.4f}, {images.max():.4f}]")
            
            # 只检查前2个批次
            if batch_idx >= 1:
                break
        
        # 检查验证数据
        print("\n" + "=" * 40)
        print("验证数据批次内容:")
        print("=" * 40)
        
        for batch_idx, batch in enumerate(val_loader):
            print(f"\n验证批次 {batch_idx + 1}:")
            print(f"  任务名称: {batch['task_name']}")
            print(f"  机器人状态形状: {batch['robot_states'].shape}")
            print(f"  动作形状: {batch['actions'].shape}")
            
            # 只检查第一个批次
            if batch_idx >= 0:
                break
        
    except Exception as e:
        print(f"检查数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("数据集检查完成")
    print("=" * 60)

if __name__ == "__main__":
    examine_dataset()