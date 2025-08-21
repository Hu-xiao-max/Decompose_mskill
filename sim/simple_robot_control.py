#!/usr/bin/env python3
"""
简化的机械臂控制示例
演示如何使用Colosseum创建最基本的机械臂控制环境
"""

import numpy as np
from omegaconf import DictConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop


def create_simple_robot_environment():
    """创建简单的机械臂环境"""
    
    # 基础数据配置
    data_config = DictConfig({
        'image_size': [128, 128],
        'renderer': 'opengl3',
        'images': {'rgb': True, 'depth': False, 'mask': False, 'point_cloud': False},
        'cameras': {'left_shoulder': True, 'right_shoulder': False, 'overhead': True, 'wrist': True, 'front': False},
        'depth_in_meters': False,
        'masks_as_one_channel': True
    })
    
    # 环境配置
    env_config = DictConfig({
        'seed': 42,
        'scene': {'factors': []}  # 不使用环境变化因子
    })
    
    # 创建环境
    env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_config),
        headless=False,  # 显示图形界面
        robot_setup="panda",  # 使用Panda机械臂
        use_variations=False,
        env_config=env_config
    )
    
    return env


def main():
    """主函数"""
    print("创建机械臂仿真环境...")
    
    # 创建环境
    env = create_simple_robot_environment()
    
    try:
        # 启动环境
        print("启动环境...")
        env.launch()
        
        # 获取任务
        print("加载篮球任务...")
        task_env = env.get_task(BasketballInHoop)
        
        # 重置环境
        print("重置环境...")
        descriptions, obs = task_env.reset()
        print(f"任务描述: {descriptions[0]}")
        
        # 获取当前机械臂状态
        print("\n当前机械臂状态:")
        print(f"夹爪位置: {obs.gripper_pose[:3]}")
        print(f"夹爪朝向: {obs.gripper_pose[3:]}")
        print(f"关节位置: {obs.joint_positions}")
        
        # 定义目标位置（相对于当前位置向上移动10cm）
        current_pos = obs.gripper_pose[:3]
        current_quat = obs.gripper_pose[3:]
        target_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.1]
        
        print(f"\n目标位置: {target_pos}")
        print("开始移动机械臂...")
        
        # 构造动作: [x, y, z, qx, qy, qz, qw, gripper]
        action = np.array(target_pos + current_quat.tolist() + [1])
        
        # 执行动作
        max_steps = 50
        for step in range(max_steps):
            obs, reward, terminate = task_env.step(action)
            
            # 检查距离
            current_pos = obs.gripper_pose[:3]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            
            if step % 10 == 0:
                print(f"步数: {step}, 当前位置: {current_pos}, 距离目标: {distance:.3f}m")
            
            if distance < 0.05:  # 5cm容错
                print(f"成功到达目标位置！最终距离: {distance:.3f}m")
                break
                
            if terminate:
                print("任务终止")
                break
        
        print("\n按Enter键继续...")
        input()
        
    except Exception as e:
        print(f"错误: {e}")
        
    finally:
        print("关闭环境...")
        env.shutdown()


if __name__ == "__main__":
    main()

