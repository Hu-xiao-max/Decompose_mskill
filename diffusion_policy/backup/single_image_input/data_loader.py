"""
完整的Colosseum数据集加载器
包含完整的RLBench支持和高级数据处理功能
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import glob
import cv2
# from scipy.spatial.transform import Rotation  # 暂时注释掉，如果不需要的话


class ColosseumDataset(Dataset):
    """完整的Colosseum数据集类"""
    
    def __init__(
        self, 
        dataset_path: str,
        tasks: Optional[List[str]] = None,
        sequence_length: int = 8,
        action_horizon: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        normalize_actions: bool = True,
        augment_images: bool = True,
        cameras: List[str] = ['front_rgb'],
        image_types: List[str] = ['rgb'],
        load_depth: bool = False,
        load_point_clouds: bool = False,
        subsample_factor: int = 1,
        max_episodes_per_task: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            dataset_path: 数据集根路径
            tasks: 要使用的任务列表，None表示使用所有任务
            sequence_length: 输入序列长度
            action_horizon: 预测的动作步数
            image_size: 图像尺寸
            normalize_actions: 是否归一化动作
            augment_images: 是否进行图像增强
            cameras: 使用的相机列表
            image_types: 图像类型列表 ['rgb', 'depth', 'mask']
            load_depth: 是否加载深度图像
            load_point_clouds: 是否加载点云数据
            subsample_factor: 数据子采样因子
            max_episodes_per_task: 每个任务的最大episode数
        """
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.normalize_actions = normalize_actions
        self.cameras = cameras
        self.image_types = image_types
        self.load_depth = load_depth
        self.load_point_clouds = load_point_clouds
        self.subsample_factor = subsample_factor
        
        # 图像预处理
        self.image_transform = self._create_image_transform(augment_images)
        
        # 深度图像预处理
        if self.load_depth:
            self.depth_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        
        # 收集所有可用的episode数据
        self.episodes = self._collect_episodes(tasks, max_episodes_per_task)
        print(f"加载了 {len(self.episodes)} 个episodes")
        
        # 计算动作统计信息用于归一化
        if self.normalize_actions:
            self._compute_action_stats()
    
    def _create_image_transform(self, augment: bool) -> transforms.Compose:
        """创建图像预处理流水线"""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        if augment:
            # 数据增强
            augmentation_list = [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, 
                    saturation=0.1, hue=0.05
                ),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(
                    self.image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2))
            ]
            
            # 随机应用增强
            for aug in augmentation_list:
                if np.random.rand() < 0.3:  # 30%概率应用每种增强
                    transform_list.insert(-2, aug)
        
        return transforms.Compose(transform_list)
    
    def _collect_episodes(self, tasks: Optional[List[str]] = None, max_per_task: Optional[int] = None) -> List[Dict]:
        """收集所有episode路径和信息"""
        episodes = []
        
        # 获取所有任务目录
        task_dirs = [d for d in os.listdir(self.dataset_path) 
                    if os.path.isdir(os.path.join(self.dataset_path, d)) and '_' in d]
        
        for task_dir in task_dirs:
            # 提取任务名
            task_name = '_'.join(task_dir.split('_')[:-1])
            
            # 如果指定了任务列表，则过滤
            if tasks is not None and task_name not in tasks:
                continue
            
            task_path = os.path.join(self.dataset_path, task_dir)
            task_episodes = []
            
            # 查找episode目录
            for variation_dir in ['variation0', 'episodes']:
                if variation_dir == 'variation0':
                    episodes_base = os.path.join(task_path, variation_dir, 'episodes')
                else:
                    episodes_base = os.path.join(task_path, variation_dir)
                
                if not os.path.exists(episodes_base):
                    continue
                
                # 找到所有episode目录
                episode_dirs = [d for d in os.listdir(episodes_base) 
                              if d.startswith('episode') and 
                              os.path.isdir(os.path.join(episodes_base, d))]
                
                for episode_dir in sorted(episode_dirs):
                    episode_path = os.path.join(episodes_base, episode_dir)
                    
                    # 检查必要的文件是否存在
                    low_dim_file = os.path.join(episode_path, 'low_dim_obs.pkl')
                    if not os.path.exists(low_dim_file):
                        continue
                    
                    # 检查相机数据
                    valid_cameras = []
                    for camera in self.cameras:
                        camera_path = os.path.join(episode_path, camera)
                        if os.path.exists(camera_path):
                            image_files = [f for f in os.listdir(camera_path) 
                                         if f.endswith('.png')]
                            if len(image_files) >= self.sequence_length + self.action_horizon:
                                valid_cameras.append(camera)
                    
                    if valid_cameras:
                        # 计算episode长度
                        camera_path = os.path.join(episode_path, valid_cameras[0])
                        image_files = [f for f in os.listdir(camera_path) 
                                     if f.endswith('.png')]
                        episode_length = len(image_files)
                        
                        if episode_length >= self.sequence_length + self.action_horizon:
                            task_episodes.append({
                                'task_name': task_name,
                                'task_dir': task_dir,
                                'episode_path': episode_path,
                                'length': episode_length,
                                'cameras': valid_cameras,
                                'variation': variation_dir
                            })
            
            # 限制每个任务的episode数量
            if max_per_task and len(task_episodes) > max_per_task:
                task_episodes = task_episodes[:max_per_task]
            
            episodes.extend(task_episodes)
        
        return episodes
    
    def _compute_action_stats(self):
        """计算动作统计信息用于归一化"""
        print("计算动作统计信息...")
        all_actions = []
        
        # 采样一部分episodes来计算统计信息
        sample_episodes = self.episodes[::max(1, len(self.episodes) // 50)]
        
        for episode_info in sample_episodes:
            try:
                with open(os.path.join(episode_info['episode_path'], 'low_dim_obs.pkl'), 'rb') as f:
                    demo_data = pickle.load(f)
                
                for obs in demo_data[::self.subsample_factor]:
                    if hasattr(obs, 'joint_velocities') and obs.joint_velocities is not None:
                        all_actions.append(obs.joint_velocities)
                    elif hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                        # 如果没有速度，计算位置差分作为近似速度
                        all_actions.append(obs.joint_positions)
            except Exception as e:
                print(f"警告: 无法加载 {episode_info['episode_path']}: {e}")
                continue
        
        if all_actions:
            all_actions = np.array(all_actions)
            self.action_mean = np.mean(all_actions, axis=0)
            self.action_std = np.std(all_actions, axis=0) + 1e-6  # 避免除零
        else:
            print("警告: 无法计算动作统计信息，使用默认值")
            self.action_mean = np.zeros(7)
            self.action_std = np.ones(7)
        
        print(f"动作均值: {self.action_mean}")
        print(f"动作标准差: {self.action_std}")
    
    def _load_image(self, image_path: str, is_depth: bool = False) -> np.ndarray:
        """加载单张图像"""
        try:
            if is_depth:
                # 加载深度图像
                image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
                if image is None:
                    raise ValueError(f"无法加载深度图像: {image_path}")
                # 归一化深度值
                image = image.astype(np.float32) / 1000.0  # 假设深度单位是mm
                image = np.clip(image, 0, 10)  # 限制深度范围到10m
                return image
            else:
                # 加载RGB图像
                image = Image.open(image_path).convert('RGB')
                return np.array(image)
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            # 返回默认图像
            if is_depth:
                return np.zeros(self.image_size, dtype=np.float32)
            else:
                return np.zeros((*self.image_size, 3), dtype=np.uint8)
    
    def _load_episode_data(self, episode_info: Dict) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """加载单个episode的数据"""
        episode_path = episode_info['episode_path']
        
        # 加载图像数据
        images_data = {}
        
        for camera in episode_info['cameras']:
            if camera not in self.cameras:
                continue
                
            camera_path = os.path.join(episode_path, camera)
            image_files = sorted([f for f in os.listdir(camera_path) if f.endswith('.png')],
                               key=lambda x: int(x.split('.')[0]))
            
            # 子采样
            if self.subsample_factor > 1:
                image_files = image_files[::self.subsample_factor]
            
            camera_images = []
            for img_file in image_files:
                img_path = os.path.join(camera_path, img_file)
                is_depth = 'depth' in camera or 'depth' in img_file
                image = self._load_image(img_path, is_depth)
                camera_images.append(image)
            
            images_data[camera] = np.array(camera_images)
        
        # 加载低维观察数据
        with open(os.path.join(episode_path, 'low_dim_obs.pkl'), 'rb') as f:
            demo_data = pickle.load(f)
        
        # 子采样
        if self.subsample_factor > 1:
            demo_data = demo_data[::self.subsample_factor]
        
        # 提取机器人状态和动作
        robot_states = []
        actions = []
        
        for obs in demo_data:
            # 机器人状态：关节位置 + 末端执行器位置和朝向 + 夹爪状态
            state = []
            
            if hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                state.extend(obs.joint_positions)
            else:
                state.extend([0.0] * 7)
            
            if hasattr(obs, 'gripper_pose') and obs.gripper_pose is not None:
                state.extend(obs.gripper_pose)  # [x, y, z, qx, qy, qz, qw]
            else:
                state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
            
            if hasattr(obs, 'gripper_open') and obs.gripper_open is not None:
                state.append(float(obs.gripper_open))
            else:
                state.append(0.5)
            
            robot_states.append(state)
            
            # 动作：优先使用关节速度，否则使用关节位置
            action = None
            if hasattr(obs, 'joint_velocities') and obs.joint_velocities is not None:
                action = obs.joint_velocities
            elif hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                action = obs.joint_positions
            
            if action is not None:
                actions.append(action[:7] if len(action) >= 7 else np.pad(action, (0, 7-len(action))))
            else:
                actions.append(np.zeros(7))
        
        robot_states = np.array(robot_states)
        actions = np.array(actions)
        
        return images_data, robot_states, actions
    
    def _augment_robot_state(self, state: np.ndarray) -> np.ndarray:
        """对机器人状态进行数据增强"""
        augmented_state = state.copy()
        
        # 添加少量噪声到关节位置
        joint_noise = np.random.normal(0, 0.01, 7)
        augmented_state[:7] += joint_noise
        
        # 添加少量噪声到末端执行器位置
        pos_noise = np.random.normal(0, 0.005, 3)
        augmented_state[7:10] += pos_noise
        
        # 保持四元数归一化
        quat = augmented_state[10:14]
        quat_noise = np.random.normal(0, 0.01, 4)
        quat += quat_noise
        quat = quat / np.linalg.norm(quat)
        augmented_state[10:14] = quat
        
        return augmented_state
    
    def __len__(self) -> int:
        """返回数据集大小"""
        total_samples = 0
        for episode_info in self.episodes:
            episode_length = episode_info['length'] // self.subsample_factor
            total_samples += max(0, episode_length - self.sequence_length - self.action_horizon + 1)
        return total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 找到对应的episode和起始位置
        current_idx = idx
        episode_idx = 0
        
        while episode_idx < len(self.episodes):
            episode_info = self.episodes[episode_idx]
            episode_length = episode_info['length'] // self.subsample_factor
            max_start_idx = episode_length - self.sequence_length - self.action_horizon + 1
            
            if current_idx < max_start_idx:
                start_idx = current_idx
                break
            
            current_idx -= max_start_idx
            episode_idx += 1
        
        if episode_idx >= len(self.episodes):
            episode_idx = len(self.episodes) - 1
            episode_info = self.episodes[episode_idx]
            episode_length = episode_info['length'] // self.subsample_factor
            start_idx = max(0, episode_length - self.sequence_length - self.action_horizon)
        else:
            episode_info = self.episodes[episode_idx]
            start_idx = current_idx
        
        # 加载episode数据
        images_data, robot_states, actions = self._load_episode_data(episode_info)
        
        # 提取序列
        end_obs_idx = start_idx + self.sequence_length
        end_action_idx = end_obs_idx + self.action_horizon
        
        # 处理图像序列
        processed_images = {}
        for camera, camera_images in images_data.items():
            if camera not in self.cameras:
                continue
                
            image_sequence = camera_images[start_idx:end_obs_idx]
            image_tensors = []
            
            for img in image_sequence:
                if len(img.shape) == 2:  # 深度图像
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                    img_tensor = self.depth_transform(img_pil)
                else:  # RGB图像
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_tensor = self.image_transform(img_pil)
                image_tensors.append(img_tensor)
            
            processed_images[camera] = torch.stack(image_tensors)
        
        # 机器人状态序列
        robot_state_sequence = robot_states[start_idx:end_obs_idx]
        
        # 数据增强（如果启用）
        if self.image_transform.transforms and hasattr(self.image_transform.transforms[0], 'brightness'):
            augmented_states = []
            for state in robot_state_sequence:
                augmented_states.append(self._augment_robot_state(state))
            robot_state_sequence = np.array(augmented_states)
        
        # 动作序列（预测目标）
        action_sequence = actions[end_obs_idx-1:end_action_idx-1]
        
        # 归一化动作
        if self.normalize_actions:
            action_sequence = (action_sequence - self.action_mean) / self.action_std
        
        result = {
            'robot_states': torch.FloatTensor(robot_state_sequence),
            'actions': torch.FloatTensor(action_sequence),
            'task_name': episode_info['task_name'],
            'episode_idx': episode_idx,
            'start_idx': start_idx
        }
        
        # 添加图像数据
        for camera, img_tensor in processed_images.items():
            result[f'images_{camera}'] = img_tensor.float()
        
        # 主要图像（用于向后兼容）
        if 'front_rgb' in processed_images:
            result['images'] = processed_images['front_rgb'].float()
        else:
            # 使用第一个可用的相机
            first_camera = list(processed_images.keys())[0]
            result['images'] = processed_images[first_camera].float()
        
        return result


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    train_tasks: Optional[List[str]] = None,
    val_tasks: Optional[List[str]] = None,
    sequence_length: int = 8,
    action_horizon: int = 4,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 如果没有指定训练和验证任务，则自动分割
    if train_tasks is None and val_tasks is None:
        all_tasks = [
            'basketball_in_hoop', 'close_box', 'close_laptop_lid', 'empty_dishwasher',
            'get_ice_from_fridge', 'hockey', 'insert_onto_square_peg', 'meat_on_grill',
            'move_hanger', 'open_drawer', 'place_wine_at_rack_location', 'put_money_in_safe',
            'reach_and_drag', 'scoop_with_spatula', 'setup_chess', 'slide_block_to_target',
            'stack_cups', 'straighten_rope', 'turn_oven_on', 'wipe_desk'
        ]
        
        # 80-20分割
        split_idx = int(len(all_tasks) * 0.8)
        train_tasks = all_tasks[:split_idx]
        val_tasks = all_tasks[split_idx:]
    
    print(f"训练任务: {train_tasks}")
    print(f"验证任务: {val_tasks}")
    
    # 创建数据集
    train_dataset = ColosseumDataset(
        dataset_path=dataset_path,
        tasks=train_tasks,
        sequence_length=sequence_length,
        action_horizon=action_horizon,
        augment_images=True,
        **kwargs
    )
    
    val_dataset = ColosseumDataset(
        dataset_path=dataset_path,
        tasks=val_tasks,
        sequence_length=sequence_length,
        action_horizon=action_horizon,
        augment_images=False,
        **kwargs
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据加载器
    dataset_path = "/home/alien/Dataset/colosseum_dataset"
    
    print("测试完整数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=4,
            sequence_length=4,
            action_horizon=2,
            num_workers=0,
            cameras=['front_rgb'],
            max_episodes_per_task=5
        )
        
        print(f"训练数据loader: {len(train_loader)} 批次")
        print(f"验证数据loader: {len(val_loader)} 批次")
        
        # 加载一个batch看看数据格式
        for batch in train_loader:
            print(f"图像形状: {batch['images'].shape}")
            print(f"机器人状态形状: {batch['robot_states'].shape}")
            print(f"动作形状: {batch['actions'].shape}")
            print(f"任务: {batch['task_name']}")
            break
            
    except Exception as e:
        print(f"测试失败，可能需要完整的RLBench环境: {e}")
        print("请使用 simple_data_loader.py 作为替代")
