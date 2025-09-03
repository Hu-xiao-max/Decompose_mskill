#!/usr/bin/env python3
"""
从数据集读取关节位置并在仿真中执行回放
"""

import os
import sys
import pickle
import numpy as np
import time
from omegaconf import DictConfig
from typing import List, Optional

# RLBench imports
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

# Colosseum imports
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.close_box import CloseBox


class DatasetReplaySimulator:
    """从数据集读取并在仿真中回放机械臂动作"""
    
    def __init__(self, dataset_path: str, task_name: str = "close_box", headless: bool = False):
        """
        初始化回放器
        
        Args:
            dataset_path: 数据集路径
            task_name: 任务名称
            headless: 是否无头模式
        """
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.headless = headless
        self.env = None
        self.task_env = None
        
        # 创建环境
        self._setup_environment()
    
    def _setup_environment(self):
        """设置仿真环境"""
        print("初始化仿真环境...")
        
        # 创建环境配置
        obs_config = ObservationConfigExt(DictConfig({
            'image_size': [128, 128],
            'images': {
                'rgb': True,  # 启用RGB以便观察
                'depth': False, 
                'mask': False, 
                'point_cloud': False
            },
            'cameras': {
                'front': False,
                'left_shoulder': False,
                'right_shoulder': False,
                'overhead': False,
                'wrist': False
            },
            'depth_in_meters': False,
            'masks_as_one_channel': True,
            'renderer': 'opengl3'
        }))
        
        # 创建环境
        self.env = EnvironmentExt(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointPosition(),
                gripper_action_mode=Discrete()
            ),
            obs_config=obs_config,
            headless=self.headless,
            robot_setup="panda"
        )
        
        # 启动环境
        self.env.launch()
        
        # 根据任务名称选择任务
        if self.task_name.lower() == "close_box":
            self.task_env = self.env.get_task(CloseBox)
        else:
            # 默认使用CloseBox，可以根据需要添加更多任务
            print(f"警告: 任务 {self.task_name} 不支持，使用 CloseBox")
            self.task_env = self.env.get_task(CloseBox)
        
        print("环境初始化完成!")
    
    def _find_episodes(self) -> List[str]:
        """查找数据集中的episodes"""
        episodes = []
        
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"数据集路径不存在: {self.dataset_path}")
        
        # 查找episode目录
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            
            # 检查多种可能的目录结构
            episode_paths = []
            
            # 直接的episode目录
            if item.startswith('episode') and os.path.isdir(item_path):
                episode_paths.append(item_path)
            
            # variation0/episodes结构
            variation_path = os.path.join(item_path, 'variation0', 'episodes')
            if os.path.exists(variation_path):
                for ep in os.listdir(variation_path):
                    if ep.startswith('episode'):
                        episode_paths.append(os.path.join(variation_path, ep))
            
            # episodes子目录
            episodes_path = os.path.join(item_path, 'episodes')
            if os.path.exists(episodes_path):
                for ep in os.listdir(episodes_path):
                    if ep.startswith('episode'):
                        episode_paths.append(os.path.join(episodes_path, ep))
            
            episodes.extend(episode_paths)
        
        return sorted(episodes)
    
    def _load_episode_data(self, episode_path: str) -> Optional[List]:
        """加载episode的低维观察数据"""
        low_dim_file = os.path.join(episode_path, 'low_dim_obs.pkl')
        
        if not os.path.exists(low_dim_file):
            print(f"警告: {episode_path} 中没有找到 low_dim_obs.pkl")
            return None
        
        try:
            with open(low_dim_file, 'rb') as f:
                demo_data = pickle.load(f)
            return demo_data
        except Exception as e:
            print(f"加载 {episode_path} 失败: {e}")
            return None
    
    def _extract_joint_positions(self, demo_data: List) -> List[np.ndarray]:
        """从演示数据中提取关节位置序列"""
        joint_positions = []
        gripper_states = []
        
        for obs in demo_data:
            # 提取关节位置 (前7个关节)
            if hasattr(obs, 'joint_positions') and obs.joint_positions is not None:
                joints = obs.joint_positions[:7]
                if len(joints) < 7:
                    # 如果关节数不足，用零填充
                    joints = np.pad(joints, (0, 7-len(joints)))
                joint_positions.append(joints)
            else:
                # 如果没有关节位置，使用零位置
                joint_positions.append(np.zeros(7))
            
            # 提取夹爪状态
            if hasattr(obs, 'gripper_open') and obs.gripper_open is not None:
                # gripper_open是布尔值，转换为动作值
                gripper_action = 1.0 if obs.gripper_open else 0.0
                gripper_states.append(gripper_action)
            else:
                gripper_states.append(1.0)  # 默认打开
        
        # 合并关节位置和夹爪状态
        actions = []
        for joints, gripper in zip(joint_positions, gripper_states):
            action = np.concatenate([joints, [gripper]])
            actions.append(action)
        
        return actions
    
    def replay_episode(self, episode_idx: int = 0, step_delay: float = 0.1, 
                      skip_steps: int = 1) -> bool:
        """
        回放指定的episode
        
        Args:
            episode_idx: episode索引
            step_delay: 每步之间的延迟时间
            skip_steps: 跳过的步数（用于加快回放）
            
        Returns:
            成功回放返回True，否则返回False
        """
        # 查找episodes
        episodes = self._find_episodes()
        
        if not episodes:
            print("没有找到任何episodes")
            return False
        
        if episode_idx >= len(episodes):
            print(f"Episode索引 {episode_idx} 超出范围，总共有 {len(episodes)} 个episodes")
            return False
        
        episode_path = episodes[episode_idx]
        print(f"回放Episode: {episode_path}")
        
        # 加载episode数据
        demo_data = self._load_episode_data(episode_path)
        if demo_data is None:
            return False
        
        # 提取动作序列
        actions = self._extract_joint_positions(demo_data)
        print(f"提取到 {len(actions)} 个动作步骤")
        
        # 重置环境
        print("重置环境...")
        descriptions, obs = self.task_env.reset()
        print(f"任务描述: {descriptions}")
        
        # 等待用户确认
        input("按Enter开始回放...")
        
        # 执行动作序列
        success_count = 0
        for i, action in enumerate(actions[::skip_steps]):
            try:
                print(f"步骤 {i+1}/{len(actions)//skip_steps}: "
                      f"关节位置 {action[:7]}, 夹爪 {action[7]:.1f}")
                
                # 执行动作
                obs, reward, terminate = self.task_env.step(action)
                success_count += 1
                
                # 检查任务是否完成
                if terminate:
                    print(f"任务完成! 奖励: {reward}")
                    break
                
                # 延迟
                if step_delay > 0:
                    time.sleep(step_delay)
                
            except Exception as e:
                print(f"执行步骤 {i} 时出错: {e}")
                continue
        
        print(f"回放完成，成功执行 {success_count} 个步骤")
        return True
    
    def list_episodes(self):
        """列出所有可用的episodes"""
        episodes = self._find_episodes()
        print(f"找到 {len(episodes)} 个episodes:")
        for i, ep in enumerate(episodes):
            # 获取episode信息
            demo_data = self._load_episode_data(ep)
            if demo_data:
                print(f"  {i}: {ep} ({len(demo_data)} 步骤)")
            else:
                print(f"  {i}: {ep} (无法加载)")
    
    def interactive_replay(self):
        """交互式回放模式"""
        print("\n=== 交互式Episode回放 ===")
        
        while True:
            try:
                # 列出episodes
                self.list_episodes()
                
                # 获取用户输入
                user_input = input("\n输入episode索引 (q退出): ").strip()
                
                if user_input.lower() == 'q':
                    break
                
                episode_idx = int(user_input)
                
                # 获取回放参数
                delay_input = input("步骤延迟秒数 (默认0.1): ").strip()
                step_delay = float(delay_input) if delay_input else 0.1
                
                skip_input = input("跳过步数 (默认1): ").strip()
                skip_steps = int(skip_input) if skip_input else 1
                
                # 执行回放
                success = self.replay_episode(episode_idx, step_delay, skip_steps)
                
                if not success:
                    print("回放失败!")
                
                # 询问是否继续
                continue_input = input("\n继续回放其他episode? (y/n): ").strip().lower()
                if continue_input != 'y':
                    break
                    
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    def shutdown(self):
        """关闭环境"""
        if self.env:
            print("关闭环境...")
            self.env.shutdown()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从数据集回放机械臂动作')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/alien/simulation/robot-colosseum/dataset/close_box_full',
                       help='数据集路径')
    parser.add_argument('--task', type=str, default='close_box',
                       help='任务名称')
    parser.add_argument('--episode', type=int, default=0,
                       help='要回放的episode索引（不指定则进入交互模式）')
    parser.add_argument('--headless', action='store_true',
                       help='无头模式运行')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='步骤间延迟时间')
    parser.add_argument('--skip', type=int, default=1,
                       help='跳过的步数')
    
    args = parser.parse_args()
    
    # 创建回放器
    simulator = None
    try:
        simulator = DatasetReplaySimulator(
            dataset_path=args.dataset_path,
            task_name=args.task,
            headless=args.headless
        )
        
        if args.episode is not None:
            # 单个episode回放
            success = simulator.replay_episode(
                episode_idx=args.episode,
                step_delay=args.delay,
                skip_steps=args.skip
            )
            if not success:
                print("回放失败!")
        else:
            # 交互式模式
            simulator.interactive_replay()
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if simulator:
            simulator.shutdown()


if __name__ == "__main__":
    main()
    #  python dataset_replay_sim.py --episode 0 --delay 0.1 --skip 1