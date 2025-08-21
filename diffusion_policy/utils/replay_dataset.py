#!/usr/bin/env python3
"""
回放数据集中的真实动作
从low_dim_obs.pkl读取joint_velocities并在basketball_in_hoop中执行
"""

import os
import pickle
import argparse
import time
import numpy as np
from typing import List, Optional
from omegaconf import DictConfig
from tqdm import tqdm

# RLBench导入
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete

# Colosseum导入
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop


class DatasetActionReplay:
    """数据集动作回放器"""
    
    def __init__(self, dataset_path: str, task_name: str = "basketball_in_hoop"):
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.episodes_data = []
        
        # 加载数据
        self._load_episodes()
    
    def _load_episodes(self):
        """加载所有episode数据"""
        print(f"从数据集加载 {self.task_name} episodes...")
        
        # 查找任务目录
        task_dirs = []
        for dir_name in os.listdir(self.dataset_path):
            if self.task_name in dir_name.lower():
                task_dirs.append(os.path.join(self.dataset_path, dir_name))
        
        if not task_dirs:
            raise ValueError(f"未找到任务 {self.task_name} 的数据")
        
        print(f"找到 {len(task_dirs)} 个任务目录")
        
        # 遍历任务目录
        for task_dir in task_dirs:
            # 查找variation目录
            for variation_dir in ['variation0', 'variation1', 'episodes']:
                if variation_dir == 'variation0' or variation_dir == 'variation1':
                    episodes_base = os.path.join(task_dir, variation_dir, 'episodes')
                else:
                    episodes_base = os.path.join(task_dir, variation_dir)
                
                if not os.path.exists(episodes_base):
                    continue
                
                # 查找episode目录
                episode_dirs = [d for d in os.listdir(episodes_base) 
                              if d.startswith('episode') and 
                              os.path.isdir(os.path.join(episodes_base, d))]
                
                for episode_dir in sorted(episode_dirs):
                    episode_path = os.path.join(episodes_base, episode_dir)
                    low_dim_file = os.path.join(episode_path, 'low_dim_obs.pkl')
                    
                    if os.path.exists(low_dim_file):
                        try:
                            with open(low_dim_file, 'rb') as f:
                                demo_data = pickle.load(f)
                            
                            # 提取动作序列
                            actions = self._extract_actions(demo_data)
                            if actions:
                                self.episodes_data.append({
                                    'path': episode_path,
                                    'actions': actions,
                                    'length': len(actions),
                                    'variation': variation_dir
                                })
                                print(f"  加载 {episode_dir}: {len(actions)} 步")
                        except Exception as e:
                            print(f"  加载 {episode_dir} 失败: {e}")
        
        print(f"总共加载了 {len(self.episodes_data)} 个episodes")
    
    def _extract_actions(self, demo_data: List) -> List[np.ndarray]:
        """从演示数据中提取动作序列"""
        actions = []
        
        for i, obs in enumerate(demo_data):
            action = None
            
            # 优先使用joint_velocities
            if hasattr(obs, 'joint_velocities') and obs.joint_velocities is not None:
                velocities = obs.joint_velocities
                if len(velocities) >= 7:
                    action = velocities[:7]
                else:
                    action = np.pad(velocities, (0, 7 - len(velocities)), 'constant')
            
            # 如果没有速度，计算位置差分
            elif hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                if i > 0:
                    prev_obs = demo_data[i-1]
                    if hasattr(prev_obs, 'joint_positions') and prev_obs.joint_positions is not None:
                        current_pos = obs.joint_positions[:7]
                        prev_pos = prev_obs.joint_positions[:7]
                        # 简单的差分作为速度估计
                        action = (current_pos - prev_pos) * 10  # 放大因子
                else:
                    action = np.zeros(7)
            
            # 添加夹爪状态
            if action is not None:
                gripper = 1.0  # 默认关闭
                if hasattr(obs, 'gripper_open') and obs.gripper_open is not None:
                    gripper = float(obs.gripper_open)
                
                # 组合成8维动作 (7个关节 + 1个夹爪)
                full_action = np.append(action, gripper)
                actions.append(full_action)
        
        return actions
    
    def get_episode(self, index: int) -> Optional[dict]:
        """获取指定索引的episode"""
        if 0 <= index < len(self.episodes_data):
            return self.episodes_data[index]
        return None
    
    def get_random_episode(self) -> Optional[dict]:
        """获取随机episode"""
        if self.episodes_data:
            import random
            return random.choice(self.episodes_data)
        return None


def create_environment(headless: bool = False):
    """创建环境"""
    print("创建环境...")
    
    # 数据配置
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
    
    # 环境配置
    env_config = DictConfig({
        'seed': 42,
        'scene': {'factors': []},
        'task_name': 'basketball_in_hoop'
    })
    
    # 创建环境
    env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(),  # 使用JointVelocity
            gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_config),
        headless=headless,
        robot_setup="panda",
        use_variations=False,
        env_config=env_config
    )
    
    env.launch()
    task = env.get_task(BasketballInHoop)
    
    print("✓ 环境创建成功!")
    return env, task


def replay_episode(task, episode_data: dict, speed_scale: float = 1.0, 
                   action_repeat: int = 1, visualize: bool = True):
    """回放单个episode"""
    actions = episode_data['actions']
    print(f"\n回放episode: {episode_data['path']}")
    print(f"  动作数: {len(actions)}")
    print(f"  速度缩放: {speed_scale}")
    print(f"  动作重复: {action_repeat}")
    
    # 重置环境
    descriptions, obs = task.reset()
    print(f"  任务: {descriptions[0]}")
    
    # 显示初始状态
    if hasattr(obs, 'joint_positions'):
        print(f"  初始关节: {obs.joint_positions[:3]}...")
    if hasattr(obs, 'gripper_pose'):
        print(f"  初始夹爪: {obs.gripper_pose[:3]}")
    
    success = False
    total_steps = 0
    
    # 执行动作序列
    pbar = tqdm(actions, desc="执行动作")
    for i, action in enumerate(pbar):
        # 缩放速度
        scaled_action = action.copy()
        scaled_action[:7] *= speed_scale
        
        # 限制速度范围
        scaled_action[:7] = np.clip(scaled_action[:7], -1.0, 1.0)
        
        # 显示进度
        if i % 20 == 0 and visualize:
            vel_norm = np.linalg.norm(scaled_action[:7])
            max_vel = np.abs(scaled_action[:7]).max()
            pbar.set_postfix({
                'vel_norm': f'{vel_norm:.3f}',
                'max_vel': f'{max_vel:.3f}',
                'gripper': f'{scaled_action[7]:.1f}'
            })
        
        # 重复执行动作
        for _ in range(action_repeat):
            obs, reward, terminate = task.step(scaled_action)
            total_steps += 1
            
            if terminate:
                success = True
                print(f"\n✓ 任务成功完成! (步骤 {i}/{len(actions)}, 总步数 {total_steps})")
                return True
            
            time.sleep(0.01)  # 小延迟
    
    if not success:
        print(f"\n✗ 任务未完成 (执行了 {len(actions)} 个动作, {total_steps} 总步数)")
    
    return success


def analyze_episode_actions(episode_data: dict):
    """分析episode的动作统计"""
    actions = np.array(episode_data['actions'])
    
    print(f"\nEpisode动作分析:")
    print(f"  路径: {episode_data['path']}")
    print(f"  动作数: {len(actions)}")
    
    # 关节速度统计
    joint_velocities = actions[:, :7]
    print(f"\n关节速度统计:")
    print(f"  均值: {np.mean(joint_velocities, axis=0)}")
    print(f"  标准差: {np.std(joint_velocities, axis=0)}")
    print(f"  最小值: {np.min(joint_velocities, axis=0)}")
    print(f"  最大值: {np.max(joint_velocities, axis=0)}")
    print(f"  范数均值: {np.mean(np.linalg.norm(joint_velocities, axis=1)):.3f}")
    
    # 夹爪统计
    gripper_states = actions[:, 7]
    print(f"\n夹爪统计:")
    print(f"  均值: {np.mean(gripper_states):.3f}")
    print(f"  打开比例: {np.mean(gripper_states > 0.5):.2%}")
    
    # 动作变化
    velocity_changes = np.diff(joint_velocities, axis=0)
    print(f"\n动作变化:")
    print(f"  变化均值: {np.mean(np.abs(velocity_changes)):.3f}")
    print(f"  最大变化: {np.max(np.abs(velocity_changes)):.3f}")


def main():
    parser = argparse.ArgumentParser(description='回放数据集动作')
    parser.add_argument('--dataset', type=str, 
                       default='/home/alien/Dataset/colosseum_dataset',
                       help='数据集路径')
    parser.add_argument('--task', type=str, 
                       default='basketball_in_hoop',
                       help='任务名称')
    parser.add_argument('--episode', type=int, default=0,
                       help='要回放的episode索引 (-1表示随机)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='速度缩放因子')
    parser.add_argument('--repeat', type=int, default=1,
                       help='每个动作重复次数')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式')
    parser.add_argument('--analyze', action='store_true',
                       help='只分析动作，不执行')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='回放的episode数量')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据集动作回放系统")
    print("=" * 60)
    
    # 加载数据集
    replay = DatasetActionReplay(args.dataset, args.task)
    
    if len(replay.episodes_data) == 0:
        print("错误：没有找到可用的episodes")
        return
    
    # 如果只是分析
    if args.analyze:
        for i in range(min(args.num_episodes, len(replay.episodes_data))):
            if args.episode >= 0:
                episode = replay.get_episode(args.episode)
            else:
                episode = replay.get_random_episode()
            
            if episode:
                analyze_episode_actions(episode)
        return
    
    # 创建环境
    env, task = create_environment(args.headless)
    
    try:
        # 回放episodes
        success_count = 0
        for i in range(args.num_episodes):
            print(f"\n--- Episode {i+1}/{args.num_episodes} ---")
            
            # 选择episode
            if args.episode >= 0:
                episode = replay.get_episode(args.episode)
            else:
                episode = replay.get_random_episode()
            
            if episode:
                # 分析动作
                if i == 0:  # 只分析第一个
                    analyze_episode_actions(episode)
                
                # 回放
                success = replay_episode(
                    task, 
                    episode,
                    speed_scale=args.speed,
                    action_repeat=args.repeat,
                    visualize=not args.headless
                )
                
                if success:
                    success_count += 1
                
                # 短暂暂停
                time.sleep(2)
        
        # 显示结果
        print(f"\n" + "=" * 60)
        print(f"回放完成!")
        print(f"成功: {success_count}/{args.num_episodes}")
        print(f"成功率: {success_count/args.num_episodes*100:.1f}%")
        print("=" * 60)
        
    finally:
        # 关闭环境
        env.shutdown()
        print("\n环境已关闭")


if __name__ == "__main__":
    main()