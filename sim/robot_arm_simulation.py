#!/usr/bin/env python3
"""
机械臂仿真环境示例
使用Colosseum框架创建包含机械臂的仿真环境，并提供控制接口让机械臂移动到指定位置

"""

import numpy as np
import time
from typing import List, Tuple, Optional
from omegaconf import DictConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop


class RobotArmController:
    """机械臂控制器类"""
    
    def __init__(self, headless: bool = False, robot_setup: str = "panda"):
        """
        初始化机械臂控制器
        
        Args:
            headless: 是否无头模式运行（不显示界面）
            robot_setup: 机器人类型，默认为"panda"
        """
        self.headless = headless
        self.robot_setup = robot_setup
        self.env = None
        self.task_env = None
        
        # 创建基础配置
        self._create_config()
        
    def _create_config(self):
        """创建环境配置"""
        # 创建观察配置
        data_config = DictConfig({
            'image_size': [128, 128],
            'renderer': 'opengl3',
            'images': {
                'rgb': True,
                'depth': True,
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
        
        # 创建环境配置
        env_config = DictConfig({
            'seed': 42,
            'scene': {
                'factors': []  # 不使用变化因子
            }
        })
        
        # 初始化环境
        self.env = EnvironmentExt(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),  # 使用末端执行器位置控制
                gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfigExt(data_config),
            headless=self.headless,
            robot_setup=self.robot_setup,
            use_variations=False,  # 不使用环境变化
            env_config=env_config
        )
        
    def launch(self):
        """启动仿真环境"""
        if self.env is None:
            raise RuntimeError("环境未初始化！")
        
        print(f"正在启动机械臂仿真环境... (机器人类型: {self.robot_setup})")
        self.env.launch()
        print("环境启动成功！")
        
        # 获取一个基础任务环境用于控制
        self.task_env = self.env.get_task(BasketballInHoop)
        print("任务环境已加载")
        
    def get_current_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前机械臂末端执行器的位置和朝向
        
        Returns:
            位置(x,y,z)和朝向(四元数)
        """
        if self.task_env is None:
            raise RuntimeError("任务环境未初始化！")
            
        obs = self.task_env.get_observation()
        
        # 从观察数据中获取机械臂状态
        gripper_pose = obs.gripper_pose  # [x, y, z, qx, qy, qz, qw]
        position = gripper_pose[:3]
        quaternion = gripper_pose[3:]
        
        return position, quaternion
    
    def move_to_position(self, 
                        target_position: List[float], 
                        target_quaternion: Optional[List[float]] = None,
                        max_steps: int = 100) -> bool:
        """
        移动机械臂到指定位置
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_quaternion: 目标朝向四元数 [qx, qy, qz, qw]，如果为None则保持当前朝向
            max_steps: 最大执行步数
            
        Returns:
            是否成功到达目标位置
        """
        if self.task_env is None:
            raise RuntimeError("任务环境未初始化！")
            
        current_pos, current_quat = self.get_current_pose()
        
        if target_quaternion is None:
            target_quaternion = current_quat.tolist()
            
        print(f"当前位置: {current_pos}")
        print(f"目标位置: {target_position}")
        print(f"开始移动机械臂...")
        
        # 构造动作：[x, y, z, qx, qy, qz, qw, gripper_action]
        action = np.array(target_position + target_quaternion + [1])  # gripper_action=1表示保持当前状态
        
        success = False
        for step in range(max_steps):
            try:
                # 执行动作
                obs, reward, terminate = self.task_env.step(action)
                
                # 检查是否到达目标位置（容错范围内）
                current_pos, _ = self.get_current_pose()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_position))
                
                if distance < 0.05:  # 5cm容错范围
                    print(f"成功到达目标位置！步数: {step + 1}, 最终距离: {distance:.3f}m")
                    success = True
                    break
                    
                if step % 10 == 0:
                    print(f"步数: {step}, 当前距离: {distance:.3f}m")
                    
                if terminate:
                    print("任务结束")
                    break
                    
            except Exception as e:
                print(f"执行动作时出错: {e}")
                break
                
        if not success:
            print(f"未能在{max_steps}步内到达目标位置")
            
        return success
    
    def move_joints(self, joint_positions: List[float], max_steps: int = 100) -> bool:
        """
        通过关节角度控制机械臂
        
        Args:
            joint_positions: 目标关节角度列表
            max_steps: 最大执行步数
            
        Returns:
            是否成功到达目标关节位置
        """
        # 需要重新配置动作模式为关节控制
        print("切换到关节控制模式...")
        
        # 重新创建环境以使用关节控制
        env_joint = EnvironmentExt(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointPosition(),
                gripper_action_mode=Discrete()
            ),
            obs_config=self.env._obs_config,
            headless=self.headless,
            robot_setup=self.robot_setup,
            use_variations=False
        )
        
        env_joint.launch()
        task_env_joint = env_joint.get_task(BasketballInHoop)
        task_env_joint.reset()
        
        print(f"目标关节角度: {joint_positions}")
        print("开始移动关节...")
        
        # 构造关节动作
        action = np.array(joint_positions + [1])  # 加上夹爪动作
        
        success = False
        for step in range(max_steps):
            try:
                obs, reward, terminate = task_env_joint.step(action)
                
                # 获取当前关节位置
                current_joints = obs.joint_positions
                
                # 检查是否到达目标关节位置
                joint_diff = np.abs(np.array(current_joints) - np.array(joint_positions))
                max_diff = np.max(joint_diff)
                
                if max_diff < 0.1:  # 0.1弧度容错
                    print(f"成功到达目标关节位置！步数: {step + 1}, 最大关节差: {max_diff:.3f}rad")
                    success = True
                    break
                    
                if step % 10 == 0:
                    print(f"步数: {step}, 最大关节差: {max_diff:.3f}rad")
                    
                if terminate:
                    print("任务结束")
                    break
                    
            except Exception as e:
                print(f"执行关节动作时出错: {e}")
                break
                
        env_joint.shutdown()
        
        if not success:
            print(f"未能在{max_steps}步内到达目标关节位置")
            
        return success
    
    def get_observation(self) -> Observation:
        """获取当前观察数据"""
        if self.task_env is None:
            raise RuntimeError("任务环境未初始化！")
        return self.task_env.get_observation()
    
    def reset_environment(self):
        """重置环境"""
        if self.task_env is None:
            raise RuntimeError("任务环境未初始化！")
        print("重置环境...")
        self.task_env.reset()
        print("环境重置完成")
    
    def shutdown(self):
        """关闭环境"""
        if self.env is not None:
            print("关闭仿真环境...")
            self.env.shutdown()
            print("环境已关闭")


def demo_basic_movement():
    """基础移动演示"""
    print("=== 基础移动演示 ===")
    
    # 创建控制器（使用图形界面）
    controller = RobotArmController(headless=False)
    
    try:
        # 启动环境
        controller.launch()
        
        # 重置环境
        controller.reset_environment()
        
        # 获取初始位置
        initial_pos, initial_quat = controller.get_current_pose()
        print(f"初始位置: {initial_pos}")
        
        # 移动到几个不同的位置
        target_positions = [
            [initial_pos[0] + 0.1, initial_pos[1], initial_pos[2]],     # 向右移动10cm
            [initial_pos[0], initial_pos[1] + 0.1, initial_pos[2]],     # 向前移动10cm
            [initial_pos[0], initial_pos[1], initial_pos[2] + 0.1],     # 向上移动10cm
            initial_pos.tolist()  # 返回初始位置
        ]
        
        for i, target_pos in enumerate(target_positions):
            print(f"\n--- 移动到位置 {i+1}: {target_pos} ---")
            success = controller.move_to_position(target_pos)
            
            if success:
                print("移动成功！")
                time.sleep(1)  # 暂停1秒观察
            else:
                print("移动失败！")
                
    finally:
        controller.shutdown()


def demo_joint_control():
    """关节控制演示"""
    print("=== 关节控制演示 ===")
    
    controller = RobotArmController(headless=False)
    
    try:
        controller.launch()
        controller.reset_environment()
        
        # 获取当前关节位置
        obs = controller.get_observation()
        initial_joints = obs.joint_positions
        print(f"初始关节位置: {initial_joints}")
        
        # 定义几个目标关节位置
        target_joint_sets = [
            [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785],  # 预定义姿态1
            [0.5, -0.3, 0.2, -1.2, 0.1, 0.8, 0.785],  # 预定义姿态2
            initial_joints  # 返回初始位置
        ]
        
        for i, target_joints in enumerate(target_joint_sets):
            print(f"\n--- 移动到关节位置 {i+1} ---")
            success = controller.move_joints(target_joints)
            
            if success:
                print("关节移动成功！")
                time.sleep(2)
            else:
                print("关节移动失败！")
                
    finally:
        controller.shutdown()


if __name__ == "__main__":
    print("Colosseum 机械臂仿真环境控制示例")
    print("=" * 50)
    
    while True:
        print("\n请选择演示模式:")
        print("1. 基础位置控制演示")
        print("2. 关节控制演示")
        print("3. 退出")
        
        choice = input("请输入选择 (1-3): ").strip()
        
        if choice == "1":
            demo_basic_movement()
        elif choice == "2":
            demo_joint_control()
        elif choice == "3":
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")

