"""
完整的Colosseum数据集加载器 - 支持多相机视角
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
import warnings
warnings.filterwarnings('ignore')


class ColosseumDataset(Dataset):
    """完整的Colosseum数据集类 - 支持多相机视角"""
    
    def __init__(
        self, 
        dataset_path: str,
        tasks: Optional[List[str]] = None,
        sequence_length: int = 8,
        action_horizon: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        normalize_actions: bool = True,
        augment_images: bool = True,
        cameras: List[str] = ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb'],
        image_types: List[str] = ['rgb'],
        load_depth: bool = False,
        load_point_clouds: bool = False,
        subsample_factor: int = 1,
        max_episodes_per_task: Optional[int] = None,
        require_all_cameras: bool = True  # 是否要求所有相机都存在
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
            require_all_cameras: 是否要求所有指定的相机都存在
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
        self.augment_images = augment_images
        self.require_all_cameras = require_all_cameras
        
        print(f"初始化数据集，使用相机: {self.cameras}")
        
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
        if augment:
            # 训练时的数据增强
            transform_list = [
                transforms.Resize(self.image_size),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, 
                    saturation=0.1, hue=0.05
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2))
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]
        else:
            # 验证时的简单预处理
            transform_list = [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]
        
        return transforms.Compose(transform_list)
    
    def _collect_episodes(self, tasks: Optional[List[str]] = None, max_per_task: Optional[int] = None) -> List[Dict]:
        """收集所有episode路径和信息"""
        episodes = []
        
        # 获取所有任务目录
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
        
        task_dirs = [d for d in os.listdir(self.dataset_path) 
                    if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # 统计相机可用性
        camera_stats = {cam: 0 for cam in self.cameras}
        skipped_episodes = 0
        
        for task_dir in sorted(task_dirs):
            # 提取任务名
            if '_' in task_dir:
                task_name = '_'.join(task_dir.split('_')[:-1])
            else:
                task_name = task_dir
            
            # 如果指定了任务列表，则过滤
            if tasks is not None and task_name not in tasks:
                continue
            
            task_path = os.path.join(self.dataset_path, task_dir)
            task_episodes = []
            
            # 查找episode目录 - 支持多种目录结构
            episode_paths = []
            
            # 检查 variation0/episodes 结构
            variation0_path = os.path.join(task_path, 'variation0', 'episodes')
            if os.path.exists(variation0_path):
                episode_dirs = [d for d in os.listdir(variation0_path) 
                              if d.startswith('episode') and 
                              os.path.isdir(os.path.join(variation0_path, d))]
                episode_paths.extend([os.path.join(variation0_path, d) for d in episode_dirs])
            
            # 检查 episodes 结构
            episodes_path = os.path.join(task_path, 'episodes')
            if os.path.exists(episodes_path):
                episode_dirs = [d for d in os.listdir(episodes_path) 
                              if d.startswith('episode') and 
                              os.path.isdir(os.path.join(episodes_path, d))]
                episode_paths.extend([os.path.join(episodes_path, d) for d in episode_dirs])
            
            # 检查直接的 episode 目录
            direct_episodes = [d for d in os.listdir(task_path) 
                             if d.startswith('episode') and 
                             os.path.isdir(os.path.join(task_path, d))]
            episode_paths.extend([os.path.join(task_path, d) for d in direct_episodes])
            
            # 去重
            episode_paths = list(set(episode_paths))
            
            for episode_path in sorted(episode_paths):
                # 检查必要的文件是否存在
                low_dim_file = os.path.join(episode_path, 'low_dim_obs.pkl')
                if not os.path.exists(low_dim_file):
                    continue
                
                # 检查相机数据
                valid_cameras = []
                missing_cameras = []
                
                for camera in self.cameras:
                    camera_path = os.path.join(episode_path, camera)
                    if os.path.exists(camera_path):
                        image_files = [f for f in os.listdir(camera_path) 
                                     if f.endswith('.png')]
                        if len(image_files) >= self.sequence_length + self.action_horizon:
                            valid_cameras.append(camera)
                            camera_stats[camera] += 1
                        else:
                            missing_cameras.append(camera)
                    else:
                        missing_cameras.append(camera)
                
                # 决定是否使用这个episode
                if self.require_all_cameras:
                    # 严格模式：要求所有相机都存在
                    if len(valid_cameras) == len(self.cameras):
                        episode_info = self._create_episode_info(
                            episode_path, task_name, task_dir, valid_cameras
                        )
                        if episode_info:
                            task_episodes.append(episode_info)
                    else:
                        skipped_episodes += 1
                        if skipped_episodes <= 5:  # 只打印前几个警告
                            print(f"跳过 {episode_path}: 缺少相机 {missing_cameras}")
                else:
                    # 宽松模式：只要有至少一个相机就可以
                    if valid_cameras:
                        episode_info = self._create_episode_info(
                            episode_path, task_name, task_dir, valid_cameras
                        )
                        if episode_info:
                            task_episodes.append(episode_info)
                    else:
                        skipped_episodes += 1
            
            # 限制每个任务的episode数量
            if max_per_task and len(task_episodes) > max_per_task:
                task_episodes = task_episodes[:max_per_task]
            
            episodes.extend(task_episodes)
            
            if task_episodes:
                print(f"任务 {task_name}: 加载了 {len(task_episodes)} 个episodes")
        
        # 打印相机统计
        print(f"\n相机可用性统计:")
        for cam, count in camera_stats.items():
            print(f"  {cam}: {count} episodes")
        
        if skipped_episodes > 0:
            print(f"\n总共跳过了 {skipped_episodes} 个episodes (相机不完整)")
        
        return episodes
    
    def _create_episode_info(self, episode_path: str, task_name: str, 
                           task_dir: str, valid_cameras: List[str]) -> Optional[Dict]:
        """创建episode信息字典"""
        # 计算episode长度
        camera_path = os.path.join(episode_path, valid_cameras[0])
        image_files = [f for f in os.listdir(camera_path) if f.endswith('.png')]
        episode_length = len(image_files)
        
        if episode_length >= self.sequence_length + self.action_horizon:
            return {
                'task_name': task_name,
                'task_dir': task_dir,
                'episode_path': episode_path,
                'length': episode_length,
                'cameras': valid_cameras,
                'variation': 'unknown'  # 可以从路径推断
            }
        return None
    
    def _compute_action_stats(self):
        """计算动作统计信息用于归一化"""
        print("计算动作统计信息...")
        
        # 如果没有episodes，使用默认值
        if not self.episodes:
            print("警告: 没有episodes可用，使用默认动作统计")
            self.action_mean = np.zeros(7)
            self.action_std = np.ones(7)
            return
        
        all_actions = []
        
        # 采样一部分episodes来计算统计信息
        sample_size = min(50, len(self.episodes))
        if sample_size == 0:
            print("警告: 没有episodes可用于计算统计信息")
            self.action_mean = np.zeros(7)
            self.action_std = np.ones(7)
            return
            
        sample_episodes = self.episodes[::max(1, len(self.episodes) // sample_size)] if sample_size > 0 else []
        
        for episode_info in sample_episodes:
            try:
                with open(os.path.join(episode_info['episode_path'], 'low_dim_obs.pkl'), 'rb') as f:
                    demo_data = pickle.load(f)
                
                for obs in demo_data[::self.subsample_factor]:
                    if hasattr(obs, 'joint_velocities') and obs.joint_velocities is not None:
                        all_actions.append(obs.joint_velocities[:7])
                    elif hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                        all_actions.append(obs.joint_positions[:7])
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
            
            # 关节位置 (7维)
            if hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                joint_pos = obs.joint_positions
                state.extend(joint_pos[:7] if len(joint_pos) >= 7 else np.pad(joint_pos, (0, 7-len(joint_pos))))
            else:
                state.extend([0.0] * 7)
            
            # 夹爪位姿 (7维: x,y,z,qx,qy,qz,qw)
            if hasattr(obs, 'gripper_pose') and obs.gripper_pose is not None:
                gripper_pose = obs.gripper_pose
                if len(gripper_pose) >= 7:
                    state.extend(gripper_pose[:7])
                else:
                    # 如果格式不对，使用默认值
                    state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
            else:
                state.extend([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
            
            # 夹爪开合状态 (1维)
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
        
        robot_states = np.array(robot_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        
        return images_data, robot_states, actions
    
    def _augment_robot_state(self, state: np.ndarray) -> np.ndarray:
        """对机器人状态进行数据增强"""
        if not self.augment_images:
            return state
            
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
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        augmented_state[10:14] = quat
        
        return augmented_state
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if not self.episodes:
            return 0
            
        total_samples = 0
        for episode_info in self.episodes:
            episode_length = episode_info['length'] // self.subsample_factor
            total_samples += max(0, episode_length - self.sequence_length - self.action_horizon + 1)
        
        # 确保至少返回1，避免空数据集问题
        return max(1, total_samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 找到对应的episode和起始位置
        current_idx = idx
        episode_idx = 0
        
        while episode_idx < len(self.episodes):
            episode_info = self.episodes[episode_idx]
            episode_length = episode_info['length'] // self.subsample_factor
            max_start_idx = episode_length - self.sequence_length - self.action_horizon + 1
            
            if max_start_idx > 0 and current_idx < max_start_idx:
                start_idx = current_idx
                break
            
            if max_start_idx > 0:
                current_idx -= max_start_idx
            episode_idx += 1
        
        if episode_idx >= len(self.episodes):
            # 如果索引超出范围，使用最后一个episode
            episode_idx = len(self.episodes) - 1
            episode_info = self.episodes[episode_idx]
            episode_length = episode_info['length'] // self.subsample_factor
            start_idx = max(0, episode_length - self.sequence_length - self.action_horizon)
        else:
            episode_info = self.episodes[episode_idx]
        
        # 加载episode数据
        images_data, robot_states, actions = self._load_episode_data(episode_info)
        
        # 提取序列
        end_obs_idx = start_idx + self.sequence_length
        end_action_idx = end_obs_idx + self.action_horizon
        
        # 处理所有相机的图像序列
        processed_images = {}
        
        # 只处理实际存在的相机
        available_cameras = [cam for cam in self.cameras if cam in images_data]
        
        for camera in available_cameras:
            camera_images = images_data[camera]
            image_sequence = camera_images[start_idx:end_obs_idx]
            image_tensors = []
            
            for img in image_sequence:
                if len(img.shape) == 2:  # 深度图像
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                    img_tensor = self.depth_transform(img_pil) if self.load_depth else torch.zeros(1, *self.image_size)
                else:  # RGB图像
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_tensor = self.image_transform(img_pil)
                image_tensors.append(img_tensor)
            
            processed_images[camera] = torch.stack(image_tensors)
        
        # 如果某些相机缺失，用零填充（仅在非严格模式下）
        if not self.require_all_cameras:
            for camera in self.cameras:
                if camera not in processed_images:
                    # 创建零张量作为占位符
                    zero_images = torch.zeros(self.sequence_length, 3, *self.image_size)
                    processed_images[camera] = zero_images
        
        # 机器人状态序列
        robot_state_sequence = robot_states[start_idx:end_obs_idx]
        
        # 数据增强
        if self.augment_images:
            augmented_states = []
            for state in robot_state_sequence:
                augmented_states.append(self._augment_robot_state(state))
            robot_state_sequence = np.array(augmented_states)
        
        # 动作序列（预测目标）
        action_sequence = actions[end_obs_idx-1:end_action_idx-1]
        
        # 归一化动作
        if self.normalize_actions:
            action_sequence = (action_sequence - self.action_mean) / self.action_std
        
        # 构建返回字典
        result = {
            'robot_states': torch.FloatTensor(robot_state_sequence),
            'actions': torch.FloatTensor(action_sequence),
            'task_name': episode_info['task_name'],
            'episode_idx': episode_idx,
            'start_idx': start_idx
        }
        
        # 添加每个相机的图像数据
        for camera in self.cameras:
            if camera in processed_images:
                result[f'images_{camera}'] = processed_images[camera].float()
        
        # 为了向后兼容，保留'images'键指向front_rgb
        if 'front_rgb' in processed_images:
            result['images'] = processed_images['front_rgb'].float()
        elif available_cameras:
            # 如果没有front_rgb，使用第一个可用的相机
            result['images'] = processed_images[available_cameras[0]].float()
        
        return result


def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    train_tasks: Optional[List[str]] = None,
    val_tasks: Optional[List[str]] = None,
    sequence_length: int = 8,
    action_horizon: int = 4,
    num_workers: int = 4,
    cameras: List[str] = ['front_rgb'],
    require_all_cameras: bool = True,
    val_split: float = 0.2,  # 新增验证集比例参数
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    
    # 如果没有指定训练和验证任务，则自动分割
    if train_tasks is None and val_tasks is None:
        # 默认任务列表
        all_tasks = [
            'basketball_in_hoop', 'close_box', 'close_laptop_lid', 'empty_dishwasher',
            'get_ice_from_fridge', 'hockey', 'insert_onto_square_peg', 'meat_on_grill',
            'move_hanger', 'open_drawer', 'place_wine_at_rack_location', 'put_money_in_safe',
            'reach_and_drag', 'scoop_with_spatula', 'setup_chess', 'slide_block_to_target',
            'stack_cups', 'straighten_rope', 'turn_oven_on', 'wipe_desk'
        ]
        
        # 检查数据集中实际存在的任务
        available_tasks = []
        if os.path.exists(dataset_path):
            for task_dir in os.listdir(dataset_path):
                if '_' in task_dir:
                    task_name = '_'.join(task_dir.split('_')[:-1])
                    if task_name in all_tasks and task_name not in available_tasks:
                        available_tasks.append(task_name)
        
        if not available_tasks:
            print(f"警告: 在 {dataset_path} 中没有找到标准任务，使用所有目录")
            available_tasks = None
        else:
            print(f"找到 {len(available_tasks)} 个可用任务")
            
            # 特殊处理：如果只有一个任务，训练和验证使用同一个任务
            if len(available_tasks) == 1:
                print(f"只有一个任务 '{available_tasks[0]}'，训练和验证将使用同一任务的不同episodes")
                train_tasks = available_tasks
                val_tasks = available_tasks
            else:
                # 多任务时进行80-20分割
                split_idx = max(1, int(len(available_tasks) * (1 - val_split)))
                train_tasks = available_tasks[:split_idx]
                val_tasks = available_tasks[split_idx:]
    
    print(f"训练任务: {train_tasks if train_tasks else '所有'}")
    print(f"验证任务: {val_tasks if val_tasks else '所有'}")
    
    # 当训练和验证使用相同任务时，需要在episode级别分割
    if train_tasks == val_tasks and train_tasks is not None:
        print("训练和验证使用相同任务，将在episode级别进行分割")
        
        # 先加载所有数据获取episode信息
        temp_dataset = ColosseumDataset(
            dataset_path=dataset_path,
            tasks=train_tasks,
            sequence_length=sequence_length,
            action_horizon=action_horizon,
            augment_images=False,
            cameras=cameras,
            require_all_cameras=require_all_cameras,
            **kwargs
        )
        
        # 获取所有episodes并分割
        all_episodes = temp_dataset.episodes
        if len(all_episodes) > 1:
            split_idx = max(1, int(len(all_episodes) * (1 - val_split)))
            train_episodes = all_episodes[:split_idx]
            val_episodes = all_episodes[split_idx:]
            
            print(f"Episode级别分割: {len(train_episodes)} 训练, {len(val_episodes)} 验证")
            
            # 创建训练数据集（使用部分episodes）
            train_dataset = ColosseumDataset(
                dataset_path=dataset_path,
                tasks=train_tasks,
                sequence_length=sequence_length,
                action_horizon=action_horizon,
                augment_images=True,
                cameras=cameras,
                require_all_cameras=require_all_cameras,
                **kwargs
            )
            train_dataset.episodes = train_episodes
            
            # 创建验证数据集（使用另一部分episodes）
            val_dataset = ColosseumDataset(
                dataset_path=dataset_path,
                tasks=val_tasks,
                sequence_length=sequence_length,
                action_horizon=action_horizon,
                augment_images=False,
                cameras=cameras,
                require_all_cameras=require_all_cameras,
                **kwargs
            )
            val_dataset.episodes = val_episodes
            
        else:
            print("警告: 只有一个episode，训练和验证将使用相同数据")
            train_dataset = temp_dataset
            val_dataset = temp_dataset
    else:
        # 不同任务的正常处理
        print("\n创建训练数据集...")
        train_dataset = ColosseumDataset(
            dataset_path=dataset_path,
            tasks=train_tasks,
            sequence_length=sequence_length,
            action_horizon=action_horizon,
            augment_images=True,
            cameras=cameras,
            require_all_cameras=require_all_cameras,
            **kwargs
        )
        
        print("\n创建验证数据集...")
        val_dataset = ColosseumDataset(
            dataset_path=dataset_path,
            tasks=val_tasks,
            sequence_length=sequence_length,
            action_horizon=action_horizon,
            augment_images=False,
            cameras=cameras,
            require_all_cameras=require_all_cameras,
            **kwargs
        )
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查数据路径和配置")
    
    if len(val_dataset) == 0:
        print("警告: 验证数据集为空，将使用训练数据集的一部分作为验证")
        val_dataset = train_dataset
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
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


def test_multi_camera_loading():
    """测试多相机加载功能"""
    # 修改为你的数据集路径
    dataset_path = "/home/alien/simulation/robot-colosseum/dataset/close_box"
    
    print("=" * 60)
    print("测试多相机数据加载")
    print("=" * 60)
    
    # 测试多相机加载
    cameras = ['front_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb']
    
    try:
        train_loader, val_loader = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=2,
            sequence_length=4,
            action_horizon=2,
            num_workers=0,
            cameras=cameras,
            image_size=(224, 224),
            normalize_actions=True,
            max_episodes_per_task=2,
            require_all_cameras=False  # 设置为False以处理相机缺失的情况
        )
        
        print(f"\n成功创建数据加载器:")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        
        # 测试加载一个批次
        print("\n加载一个训练批次...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"\n批次 {batch_idx + 1} 内容:")
            print(f"  - robot_states: {batch['robot_states'].shape}")
            print(f"  - actions: {batch['actions'].shape}")
            print(f"  - task_name: {batch['task_name']}")
            
            # 检查每个相机的图像
            for camera in cameras:
                key = f'images_{camera}'
                if key in batch:
                    print(f"  - {key}: {batch[key].shape}")
                    # 检查数值范围
                    print(f"      范围: [{batch[key].min():.2f}, {batch[key].max():.2f}]")
                else:
                    print(f"  - {key}: 不存在")
            
            # 检查兼容性键
            if 'images' in batch:
                print(f"  - images (兼容): {batch['images'].shape}")
            
            # 只测试第一个批次
            break
        
        print("\n测试成功!")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行测试
    test_multi_camera_loading()