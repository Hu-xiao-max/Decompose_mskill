#!/usr/bin/env python3
"""
推理脚本 - 支持多相机输入
基于训练的多相机配置进行推理
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from PIL import Image
from typing import Tuple, List, Dict
from omegaconf import DictConfig

# RLBench导入
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

# Colosseum导入
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks import *

# 添加diffusion_policy路径
import sys
sys.path.append('/home/alien/simulation/robot-colosseum/diffusion_policy')
from diffusion_model import create_diffusion_policy


class MultiCameraDiffusionInference:
    """多相机扩散策略推理器"""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 获取相机配置
        self.cameras = self.config.get('cameras', ['front_rgb'])
        self.num_cameras = self.config.get('num_cameras', len(self.cameras))
        
        # 创建模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 参数
        self.sequence_length = self.config.get('sequence_length', 4)
        self.action_horizon = self.config.get('action_horizon', 2)
        self.image_size = tuple(self.config.get('image_size', [224, 224]))
        
        # 动作统计（从训练数据获取）
        self.action_mean = np.array(self.config.get('action_mean', [0.0] * 7))
        self.action_std = np.array(self.config.get('action_std', [1.0] * 7))
        
        # 图像变换
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 为每个相机创建缓存
        self.image_buffers = {cam: [] for cam in self.cameras}
        self.state_buffer = []
        
        print(f"✓ 多相机推理器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  相机: {self.cameras}")
        print(f"  相机数量: {self.num_cameras}")
        print(f"  动作维度: {self.config.get('action_dim', 7)}")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  融合方法: {self.config.get('fusion_method', 'attention')}")
    
    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 如果配置中有保存的动作统计，使用它们
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if 'action_mean' in saved_config:
                self.config['action_mean'] = saved_config['action_mean']
            if 'action_std' in saved_config:
                self.config['action_std'] = saved_config['action_std']
        
        model = create_diffusion_policy(
            action_dim=self.config.get('action_dim', 7),
            action_horizon=self.config.get('action_horizon', 2),
            state_dim=self.config.get('state_dim', 15),
            vision_feature_dim=self.config.get('vision_feature_dim', 256),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_diffusion_steps=self.config.get('num_diffusion_steps', 50),
            num_layers=self.config.get('num_layers', 2),
            num_heads=self.config.get('num_heads', 4),
            dropout=self.config.get('dropout', 0.1),
            num_cameras=self.num_cameras,
            fusion_method=self.config.get('fusion_method', 'attention')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✓ 模型加载成功 (Epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"  最佳验证损失: {checkpoint.get('best_val_loss', 'unknown')}")
        return model
    
    def _get_camera_image(self, obs, camera_name: str) -> np.ndarray:
        """从观察中获取指定相机的图像"""
        # 相机名称映射（从配置名到观察属性名）
        camera_mapping = {
            'front_rgb': 'front_rgb',
            'left_shoulder_rgb': 'left_shoulder_rgb',
            'right_shoulder_rgb': 'right_shoulder_rgb',
            'wrist_rgb': 'wrist_rgb',
            'overhead_rgb': 'overhead_rgb'
        }
        
        # 获取对应的属性名
        attr_name = camera_mapping.get(camera_name, camera_name)
        
        # 尝试获取图像
        if hasattr(obs, attr_name):
            image = getattr(obs, attr_name)
            if image is not None:
                return image
        
        # 如果找不到，返回默认图像
        print(f"警告: 无法获取相机 {camera_name} 的图像，使用默认图像")
        return np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    def process_observation(self, obs) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """处理多相机观察"""
        camera_tensors = {}
        
        # 处理每个相机的图像
        for camera in self.cameras:
            # 获取图像
            image = self._get_camera_image(obs, camera)
            
            # 转换图像
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 确保是RGB图像
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            image_pil = Image.fromarray(image)
            image_tensor = self.transform(image_pil).to(self.device)
            camera_tensors[camera] = image_tensor
        
        # 获取机器人状态
        state = []
        
        # 关节位置 (7维)
        if hasattr(obs, 'joint_positions'):
            joint_pos = obs.joint_positions
            state.extend(list(joint_pos[:7]))
        else:
            state.extend([0.0] * 7)
        
        # 末端执行器位姿 (7维: x,y,z,qx,qy,qz,qw)
        if hasattr(obs, 'gripper_pose'):
            pose = obs.gripper_pose
            if len(pose) >= 7:
                state.extend(list(pose[:7]))
            elif len(pose) >= 3:
                state.extend(list(pose[:3]))
                state.extend([0.0, 0.0, 0.0, 1.0])
            else:
                state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        else:
            state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        
        # 夹爪状态 (1维)
        if hasattr(obs, 'gripper_open'):
            state.append(float(obs.gripper_open))
        else:
            state.append(1.0)
        
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        
        return camera_tensors, state_tensor
    
    def update_buffers(self, camera_images: Dict[str, torch.Tensor], state: torch.Tensor):
        """更新多相机缓存"""
        # 更新每个相机的缓存
        for camera, image in camera_images.items():
            self.image_buffers[camera].append(image)
            
            # 保持长度
            while len(self.image_buffers[camera]) > self.sequence_length:
                self.image_buffers[camera].pop(0)
        
        # 更新状态缓存
        self.state_buffer.append(state)
        while len(self.state_buffer) > self.sequence_length:
            self.state_buffer.pop(0)
    
    def predict_action(self) -> np.ndarray:
        """预测动作（支持多相机）"""
        # 填充缓存到序列长度
        for camera in self.cameras:
            while len(self.image_buffers[camera]) < self.sequence_length:
                if len(self.image_buffers[camera]) > 0:
                    # 复制第一帧
                    self.image_buffers[camera].insert(0, self.image_buffers[camera][0])
                else:
                    # 使用默认图像
                    default_img = torch.zeros(3, *self.image_size, device=self.device)
                    self.image_buffers[camera].append(default_img)
        
        while len(self.state_buffer) < self.sequence_length:
            if len(self.state_buffer) > 0:
                self.state_buffer.insert(0, self.state_buffer[0])
            else:
                default_state = torch.zeros(15, device=self.device)
                self.state_buffer.append(default_state)
        
        # 准备输入
        if self.num_cameras == 1:
            # 单相机：[1, seq_len, 3, H, W]
            images = torch.stack(self.image_buffers[self.cameras[0]]).unsqueeze(0)
        else:
            # 多相机：[1, num_cameras, seq_len, 3, H, W]
            camera_images = []
            for camera in self.cameras:
                cam_seq = torch.stack(self.image_buffers[camera])  # [seq_len, 3, H, W]
                camera_images.append(cam_seq)
            images = torch.stack(camera_images).unsqueeze(0)  # [1, num_cameras, seq_len, 3, H, W]
        
        states = torch.stack(self.state_buffer).unsqueeze(0)  # [1, seq_len, state_dim]
        
        # 采样动作
        with torch.no_grad():
            actions = self.model.sample(
                images, 
                states, 
                num_inference_steps=self.config.get('num_inference_steps', 50)
            )
        
        # 取第一个动作
        action = actions[0, 0].cpu().numpy()  # [7]
        
        # 反归一化动作（如果训练时进行了归一化）
        if self.config.get('normalize_actions', True):
            action = action * self.action_std + self.action_mean
        
        # 限制关节速度范围
        max_velocity = 0.5
        action = np.clip(action, -max_velocity, max_velocity)
        
        # 如果动作太小，适当放大
        if np.abs(action).max() < 0.01:
            action = action * 10
        
        # 添加夹爪动作（保持打开）
        action = np.append(action, 1.0)  # 8维
        
        return action
    
    def reset(self):
        """重置所有缓存"""
        for camera in self.cameras:
            self.image_buffers[camera].clear()
        self.state_buffer.clear()


def main():
    # 任务配置
    class_task_name = 'CloseBox'
    load_path = '/home/alien/simulation/robot-colosseum/diffusion_policy/my_model/'+class_task_name
    
    parser = argparse.ArgumentParser(description='多相机Diffusion Policy推理')
    parser.add_argument('--model', type=str, 
                       default=f'{load_path}/best_model.pth',
                       help='模型路径')
    parser.add_argument('--config', type=str,
                       default=f'{load_path}/config.json',
                       help='配置文件')
    parser.add_argument('--task', type=str, default=class_task_name,
                       help='任务名称')
    parser.add_argument('--episodes', type=int, default=1, 
                       help='测试回合数')
    parser.add_argument('--headless', action='store_true', 
                       help='无头模式')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='设备')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='每个episode的最大步数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("多相机Diffusion Policy推理")
    print("=" * 60)
    
    # 创建推理器
    inference = MultiCameraDiffusionInference(args.model, args.config, args.device)
    
    # 创建环境配置
    data_config = DictConfig({
        'image_size': [256, 256],
        'renderer': 'opengl3',
        'images': {
            'rgb': True, 
            'depth': False, 
            'mask': False, 
            'point_cloud': False
        },
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
        'task_name': args.task
    })
    
    # 创建环境
    env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),
            gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_config),
        headless=args.headless,
        robot_setup="panda",
        use_variations=False,
        env_config=env_config
    )
    
    env.launch()
    
    # 获取任务
    task_mapping = {
        'CloseBox': 'colosseum.rlbench.tasks.close_box.CloseBox',
        'BasketballInHoop': 'colosseum.rlbench.tasks.basketball_in_hoop.BasketballInHoop',
        'OpenDrawer': 'colosseum.rlbench.tasks.open_drawer.OpenDrawer',
        'StackCups': 'colosseum.rlbench.tasks.stack_cups.StackCups',
    }
    
    if args.task in task_mapping:
        # 动态导入任务类
        module_path, class_name = task_mapping[args.task].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        task_class = getattr(module, class_name)
        task = env.get_task(task_class)
    else:
        raise ValueError(f"不支持的任务: {args.task}")
    
    print(f"✓ 环境创建成功")
    print(f"  任务: {args.task}")
    
    # 统计信息
    total_success = 0
    
    # 运行测试
    for episode in range(args.episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*50}")
        
        # 重置
        descriptions, obs = task.reset()
        inference.reset()
        
        print(f"任务描述: {descriptions[0]}")
        
        success = False
        
        for step in range(args.max_steps):
            # 处理多相机观察
            camera_images, state = inference.process_observation(obs)
            inference.update_buffers(camera_images, state)
            
            # 预测动作
            action = inference.predict_action()
            
            # 显示进度
            if step % 20 == 0:
                vel_norm = np.linalg.norm(action[:7])
                gripper = action[7]
                print(f"  Step {step:3d}: |v|={vel_norm:.3f}, gripper={gripper:.1f}")
            
            # # 执行动作（可以重复几次以增加效果）
            # repeat_action = 5
            # for _ in range(repeat_action):
            #     obs, reward, terminate = task.step(action)
            
            obs, reward, terminate = task.step(action)
                
            if terminate:
                success = True
                print(f"\n✓ 任务完成! (Step {step})")
                total_success += 1
                break
                
                # time.sleep(0.01)  # 小延迟
            
            if success:
                break
        
        if not success:
            print(f"\n✗ 任务未完成 (达到最大步数)")
    
    # 显示统计
    print(f"\n{'='*60}")
    print(f"测试完成")
    print(f"成功率: {total_success}/{args.episodes} ({100*total_success/args.episodes:.1f}%)")
    print(f"{'='*60}")
    
    # 关闭环境
    env.shutdown()
    print("\n环境已关闭")


if __name__ == "__main__":
    main()