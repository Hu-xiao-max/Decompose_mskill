#!/usr/bin/env python3
"""
Basketball in Hoop 推理脚本 - 使用JointVelocity控制
基于visualize_task.py的配置
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from PIL import Image
from typing import Tuple
from omegaconf import DictConfig

# RLBench导入 - 与visualize_task.py完全相同
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

# Colosseum导入
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt

# from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop
from colosseum.rlbench.tasks import *

# 添加diffusion_policy路径
import sys
sys.path.append('/home/alien/simulation/robot-colosseum/diffusion_policy')
from diffusion_model import create_diffusion_policy


class DiffusionInference:
    """扩散策略推理器"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 创建模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 参数
        self.sequence_length = self.config.get('sequence_length', 4)
        self.action_horizon = self.config.get('action_horizon', 2)
        
        # 图像变换
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 缓存
        self.image_buffer = []
        self.state_buffer = []
        
        print(f"✓ 推理器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  动作维度: {self.config.get('action_dim', 7)}")
    
    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = create_diffusion_policy(
            action_dim=self.config.get('action_dim', 7),
            action_horizon=self.config.get('action_horizon', 2),
            state_dim=self.config.get('state_dim', 15),
            vision_feature_dim=self.config.get('vision_feature_dim', 512),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_diffusion_steps=self.config.get('num_diffusion_steps', 100),
            num_layers=self.config.get('num_layers', 4),
            num_heads=self.config.get('num_heads', 8),
            dropout=self.config.get('dropout', 0.1)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✓ 模型加载成功 (Epoch {checkpoint.get('epoch', 'unknown')})")
        return model
    
    def process_observation(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理观察"""
        # 获取图像
        image = None
        if hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
            image = obs.front_rgb
        elif hasattr(obs, 'wrist_rgb') and obs.wrist_rgb is not None:
            image = obs.wrist_rgb
        else:
            image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # 转换图像
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        # transform返回的是tensor，移动到设备
        tensor_result = self.transform(image_pil)
        image_tensor = tensor_result.to(self.device)  # type: ignore
        
        # 获取状态
        state = []
        
        # 关节位置
        if hasattr(obs, 'joint_positions'):
            state.extend(list(obs.joint_positions[:7]))
        else:
            state.extend([0.0] * 7)
        
        # 末端执行器位姿
        if hasattr(obs, 'gripper_pose'):
            pose = obs.gripper_pose
            if len(pose) >= 3:
                state.extend(list(pose[:3]))
                if len(pose) >= 7:
                    state.extend(list(pose[3:7]))
                else:
                    state.extend([0.0, 0.0, 0.0, 1.0])
            else:
                state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        else:
            state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        
        # 夹爪状态
        if hasattr(obs, 'gripper_open'):
            state.append(float(obs.gripper_open))
        else:
            state.append(1.0)
        
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        
        return image_tensor, state_tensor
    
    def update_buffers(self, image: torch.Tensor, state: torch.Tensor):
        """更新缓存"""
        self.image_buffer.append(image)
        self.state_buffer.append(state)
        
        # 保持长度
        while len(self.image_buffer) > self.sequence_length:
            self.image_buffer.pop(0)
            self.state_buffer.pop(0)
    
    def predict_action(self) -> np.ndarray:
        """预测动作"""
        # 填充缓存
        while len(self.image_buffer) < self.sequence_length:
            if len(self.image_buffer) > 0:
                self.image_buffer.insert(0, self.image_buffer[0])
                self.state_buffer.insert(0, self.state_buffer[0])
            else:
                default_img = torch.zeros(3, 224, 224, device=self.device)
                default_state = torch.zeros(15, device=self.device)
                self.image_buffer.append(default_img)
                self.state_buffer.append(default_state)
        
        # 准备输入
        images = torch.stack(self.image_buffer).unsqueeze(0).to(self.device)
        states = torch.stack(self.state_buffer).unsqueeze(0).to(self.device)
        
        # 采样
        with torch.no_grad():
            actions = self.model.sample(images, states, num_inference_steps=50)
        
        # 取第一个动作
        action = actions[0, 0].cpu().numpy()  # [7]
        
        # 限制关节速度范围
        action = np.clip(action, -0.5, 0.5)  # 限制速度
        
        # 如果值太小，放大
        if np.abs(action).max() < 0.01:
            action = action * 10
        
        # 添加夹爪
        action = np.append(action, 1.0)  # 8维
        
        return action
    
    def reset(self):
        """重置缓存"""
        self.image_buffer.clear()
        self.state_buffer.clear()


def main():
    class_task_name = 'CloseBox'
    class_task_name_path = 'CloseBox_single_view'
    load_path='/home/alien/simulation/robot-colosseum/diffusion_policy/my_model/'+class_task_name_path

    parser = argparse.ArgumentParser(description='Basketball in Hoop推理')
    parser.add_argument('--model', type=str, 
                       default=load_path + '/best_model.pth',
                       help='模型路径')
    parser.add_argument('--config', type=str,
                       default=load_path + '/config.json',
                       help='配置文件')
    parser.add_argument('--episodes', type=int, default=1, help='测试回合数')
    parser.add_argument('--headless', action='store_true', help='无头模式')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("=" * 50)
    
    # 创建推理器
    inference = DiffusionInference(args.model, args.config, args.device)
    
    # 创建环境 - 与visualize_task.py相同
    data_config = DictConfig({
        'image_size': [256, 256],
        'renderer': 'opengl3',
        'images': {'rgb': True, 'depth': False, 'mask': False, 'point_cloud': False},
        'cameras': {
            'left_shoulder': True,
            'right_shoulder': True,
            'overhead': True,
            'wrist': True,
            'front': True
        },
        'depth_in_meters': False,
        'masks_as_one_channel': True
    })
    
    env_config = DictConfig({
        'seed': 42,
        'scene': {'factors': []},
        'task_name': class_task_name
    })
    
    # 创建环境
    env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),  # 关键：使用JointVelocity
            gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_config),
        headless=args.headless,
        robot_setup="panda",
        use_variations=False,
        env_config=env_config
    )
    
    env.launch()
    
    # 根据任务名获取任务类
    if class_task_name == 'CloseBox':
        from colosseum.rlbench.tasks.close_box import CloseBox
        task = env.get_task(CloseBox)
    elif class_task_name == 'BasketballInHoop':
        from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop
        task = env.get_task(BasketballInHoop)
    else:
        raise ValueError(f"不支持的任务: {class_task_name}")
    
    print("✓ 环境创建成功")
    
    # 运行测试
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")
        
        # 重置
        descriptions, obs = task.reset()
        inference.reset()
        
        print(f"任务: {descriptions[0]}")
        
        success = False
        max_steps = 200
        
        for step in range(max_steps):
            # 处理观察
            image, state = inference.process_observation(obs)
            inference.update_buffers(image, state)
            
            # 预测动作
            action = inference.predict_action()
            
            # 显示进度
            if step % 20 == 0:
                vel_norm = np.linalg.norm(action[:7])
                print(f"  Step {step}: 关节速度范数={vel_norm:.3f}")
            
            # 执行动作

            obs, reward, terminate = task.step(action)
                
            if terminate:
                success = True
                print(f"✓ 任务完成! (Step {step})")
                break
                
                # time.sleep(0.02)
            
            if success:
                break
        
        if not success:
            print(f"✗ 任务未完成")
    
    # 关闭环境
    env.shutdown()
    print("\n环境已关闭")


if __name__ == "__main__":
    main()