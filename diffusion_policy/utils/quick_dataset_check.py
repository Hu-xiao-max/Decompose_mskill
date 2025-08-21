#!/usr/bin/env python3
"""
快速检查训练数据集输入的脚本
专门用于查看单个批次的详细内容
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# 添加路径以导入模块
sys.path.append('/home/alien/simulation/robot-colosseum/diffusion_policy')

from data_loader import create_data_loaders


def print_detailed_batch(batch, batch_idx=0):
    """详细打印单个批次的所有信息"""
    
    print("=" * 80)
    print(f"批次 {batch_idx} 详细信息")
    print("=" * 80)
    
    # 1. 基本信息
    print(f"\n📋 基本信息:")
    print(f"  批次中的样本数: {len(batch['task_name'])}")
    print(f"  任务名称: {batch['task_name']}")
    if 'episode_idx' in batch:
        print(f"  Episode索引: {batch['episode_idx']}")
    if 'start_idx' in batch:
        print(f"  起始索引: {batch['start_idx']}")
    
    # 2. 图像数据详情
    images = batch['images']
    print(f"\n🖼️  图像数据:")
    print(f"  张量形状: {images.shape}")  # [batch_size, seq_len, channels, height, width]
    print(f"  数据类型: {images.dtype}")
    print(f"  设备: {images.device}")
    print(f"  内存大小: {images.element_size() * images.nelement() / 1024 / 1024:.1f} MB")
    print(f"  数值统计:")
    print(f"    最小值: {images.min().item():.6f}")
    print(f"    最大值: {images.max().item():.6f}")
    print(f"    均值: {images.mean().item():.6f}")
    print(f"    标准差: {images.std().item():.6f}")
    
    # 检查第一个样本的每一帧
    print(f"\n  第一个样本的图像序列 (共{images.shape[1]}帧):")
    for frame_idx in range(images.shape[1]):
        frame = images[0, frame_idx]  # [C, H, W]
        print(f"    帧{frame_idx}: 形状={frame.shape}, 范围=[{frame.min():.3f}, {frame.max():.3f}], 均值={frame.mean():.3f}")
    
    # 3. 机器人状态详情
    robot_states = batch['robot_states']
    print(f"\n🤖 机器人状态:")
    print(f"  张量形状: {robot_states.shape}")  # [batch_size, seq_len, state_dim]
    print(f"  数据类型: {robot_states.dtype}")
    print(f"  设备: {robot_states.device}")
    print(f"  数值统计:")
    print(f"    最小值: {robot_states.min().item():.6f}")
    print(f"    最大值: {robot_states.max().item():.6f}")
    print(f"    均值: {robot_states.mean().item():.6f}")
    print(f"    标准差: {robot_states.std().item():.6f}")
    
    # 分解状态向量的各个部分
    if robot_states.shape[-1] >= 15:
        first_sample_last_frame = robot_states[0, -1, :]  # 第一个样本的最后一帧
        
        print(f"\n  第一个样本最后一帧的状态分解:")
        print(f"    关节位置 (前7维): {first_sample_last_frame[:7].cpu().numpy()}")
        print(f"    末端位置 (8-10维): {first_sample_last_frame[7:10].cpu().numpy()}")
        print(f"    末端姿态 (11-14维): {first_sample_last_frame[10:14].cpu().numpy()}")
        print(f"    夹爪状态 (15维): {first_sample_last_frame[14].item():.6f}")
        
        # 显示整个序列的变化
        print(f"\n  序列中状态的变化 (第一个样本):")
        for t in range(robot_states.shape[1]):
            state_t = robot_states[0, t, :]
            joint_pos_t = state_t[:7].cpu().numpy()
            ee_pos_t = state_t[7:10].cpu().numpy()
            gripper_t = state_t[14].item()
            print(f"    t={t}: 关节范围=[{joint_pos_t.min():.3f}, {joint_pos_t.max():.3f}], "
                  f"末端位置={ee_pos_t}, 夹爪={gripper_t:.3f}")
    
    # 4. 动作数据详情
    actions = batch['actions']
    print(f"\n🎮 动作数据:")
    print(f"  张量形状: {actions.shape}")  # [batch_size, action_horizon, action_dim]
    print(f"  数据类型: {actions.dtype}")
    print(f"  设备: {actions.device}")
    print(f"  数值统计:")
    print(f"    最小值: {actions.min().item():.6f}")
    print(f"    最大值: {actions.max().item():.6f}")
    print(f"    均值: {actions.mean().item():.6f}")
    print(f"    标准差: {actions.std().item():.6f}")
    
    # 显示每个样本的动作序列
    print(f"\n  所有样本的动作序列:")
    for sample_idx in range(actions.shape[0]):
        print(f"    样本{sample_idx} (任务: {batch['task_name'][sample_idx]}):")
        for action_idx in range(actions.shape[1]):
            action = actions[sample_idx, action_idx].cpu().numpy()
            print(f"      动作步骤{action_idx}: {action}")
    
    # 5. 动作在各维度上的分布
    print(f"\n  动作各维度统计:")
    for dim in range(actions.shape[-1]):
        dim_values = actions[:, :, dim].flatten()
        print(f"    维度{dim}: 范围=[{dim_values.min():.6f}, {dim_values.max():.6f}], "
              f"均值={dim_values.mean():.6f}, 标准差={dim_values.std():.6f}")
    
    # 6. 批次中的其他键
    print(f"\n📦 批次中的所有键:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: 张量 {value.shape} {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: 列表长度={len(value)}, 内容={value}")
        else:
            print(f"  {key}: {type(value)} = {value}")
    
    print("\n" + "=" * 80)


def save_sample_image(batch, save_path="./sample_image.png"):
    """保存一张样本图像用于查看"""
    
    images = batch['images']  # [batch_size, seq_len, channels, height, width]
    
    # 取第一个样本的最后一帧
    sample_img = images[0, -1]  # [C, H, W]
    
    # 反归一化 (ImageNet标准)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 移到CPU并反归一化
    img_cpu = sample_img.cpu()
    img_denorm = img_cpu * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    # 转换为PIL图像
    img_np = img_denorm.numpy().transpose(1, 2, 0)  # CHW -> HWC
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    
    # 保存
    img_pil.save(save_path)
    print(f"✅ 样本图像已保存到: {save_path}")


def main():
    """主函数"""
    
    # 配置参数 - 使用与train.py相同的配置
    config = {
        'dataset_path': '/home/alien/simulation/robot-colosseum/dataset/basketball_in_hoop',
        'batch_size': 2,  # 小批次方便查看
        'sequence_length': 4,
        'action_horizon': 2,
        'num_workers': 0,
        'image_size': (224, 224),
        'normalize_actions': True,
        'cameras': ['front_rgb'],
        'max_episodes_per_task': 2  # 限制episode数量
    }
    
    print("🔍 训练数据集快速检查工具")
    print("=" * 50)
    print(f"数据集路径: {config['dataset_path']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"序列长度: {config['sequence_length']}")
    print(f"动作步数: {config['action_horizon']}")
    
    try:
        # 创建数据加载器
        print("\n📂 创建数据加载器...")
        train_loader, val_loader = create_data_loaders(**config)
        
        print(f"✅ 训练数据: {len(train_loader)} 批次")
        print(f"✅ 验证数据: {len(val_loader)} 批次")
        
        # 获取第一个批次
        print("\n📋 获取第一个训练批次...")
        first_batch = next(iter(train_loader))
        
        # 详细检查这个批次
        print_detailed_batch(first_batch, 0)
        
        # 保存样本图像
        save_sample_image(first_batch, "./quick_check_sample.png")
        
        # 如果有验证数据，也检查一个验证批次
        if len(val_loader) > 0:
            print("\n" + "🔎 检查第一个验证批次:")
            val_batch = next(iter(val_loader))
            print_detailed_batch(val_batch, 0)
        
        print("\n✅ 数据检查完成!")
        
    except Exception as e:
        print(f"\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
