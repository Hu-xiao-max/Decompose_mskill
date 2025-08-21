#!/usr/bin/env python3
"""
机械臂任务演示脚本
展示如何使用Colosseum框架创建和执行各种机械臂操作任务
"""

import numpy as np
import time
from typing import List
from omegaconf import DictConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop
from colosseum.rlbench.tasks.reach_and_drag import ReachAndDrag
from colosseum.rlbench.tasks.stack_cups import StackCups


class RobotTaskDemo:
    """机械臂任务演示类"""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.env = None
        
    def create_environment(self):
        """创建环境"""
        data_config = DictConfig({
            'image_size': [256, 256],
            'renderer': 'opengl3',
            'images': {'rgb': True, 'depth': True, 'mask': False, 'point_cloud': False},
            'cameras': {'left_shoulder': True, 'right_shoulder': True, 'overhead': True, 'wrist': True, 'front': True},
            'depth_in_meters': False,
            'masks_as_one_channel': True
        })
        
        env_config = DictConfig({
            'seed': 42,
            'scene': {'factors': []}
        })
        
        self.env = EnvironmentExt(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaPlanning(),
                gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfigExt(data_config),
            headless=self.headless,
            robot_setup="panda",
            use_variations=False,
            env_config=env_config
        )
        
        self.env.launch()
        print("环境创建成功!")
    
    def demo_basketball_task(self):
        """演示篮球投篮任务"""
        print("\n=== 篮球投篮任务演示 ===")
        
        task_env = self.env.get_task(BasketballInHoop)
        descriptions, obs = task_env.reset()
        
        print(f"任务描述: {descriptions[0]}")
        print("开始执行投篮动作...")
        
        # 获取球的位置和篮筐位置
        ball_pos = obs.gripper_pose[:3]  # 假设夹爪初始在球的位置
        print(f"球的位置: {ball_pos}")
        
        # 执行抓取和投篮动作序列
        actions_sequence = [
            # 1. 移动到球的位置
            np.array([ball_pos[0], ball_pos[1], ball_pos[2], 0, 0, 0, 1, 0]),  # 打开夹爪
            # 2. 抓住球
            np.array([ball_pos[0], ball_pos[1], ball_pos[2], 0, 0, 0, 1, 1]),  # 关闭夹爪
            # 3. 举起球
            np.array([ball_pos[0], ball_pos[1], ball_pos[2] + 0.2, 0, 0, 0, 1, 1]),
            # 4. 向篮筐方向移动
            np.array([ball_pos[0] + 0.3, ball_pos[1], ball_pos[2] + 0.3, 0, 0, 0, 1, 1]),
            # 5. 投篮（释放球）
            np.array([ball_pos[0] + 0.3, ball_pos[1], ball_pos[2] + 0.3, 0, 0, 0, 1, 0]),
        ]
        
        for i, action in enumerate(actions_sequence):
            print(f"执行动作 {i+1}/{len(actions_sequence)}")
            
            for _ in range(20):  # 每个动作执行20步
                obs, reward, terminate = task_env.step(action)
                
                if terminate:
                    print("任务完成!")
                    return
                    
                time.sleep(0.05)
        
        print("篮球任务演示完成")
    
    def demo_reach_and_drag_task(self):
        """演示伸手拖拽任务"""
        print("\n=== 伸手拖拽任务演示 ===")
        
        task_env = self.env.get_task(ReachAndDrag)
        descriptions, obs = task_env.reset()
        
        print(f"任务描述: {descriptions[0]}")
        print("开始执行拖拽动作...")
        
        # 获取初始位置
        start_pos = obs.gripper_pose[:3]
        
        # 执行拖拽动作序列
        actions_sequence = [
            # 1. 移动到目标物体
            np.array([start_pos[0] + 0.1, start_pos[1] + 0.1, start_pos[2], 0, 0, 0, 1, 0]),
            # 2. 抓住物体
            np.array([start_pos[0] + 0.1, start_pos[1] + 0.1, start_pos[2], 0, 0, 0, 1, 1]),
            # 3. 拖拽到目标位置
            np.array([start_pos[0] - 0.2, start_pos[1] - 0.1, start_pos[2], 0, 0, 0, 1, 1]),
            # 4. 释放物体
            np.array([start_pos[0] - 0.2, start_pos[1] - 0.1, start_pos[2], 0, 0, 0, 1, 0]),
        ]
        
        for i, action in enumerate(actions_sequence):
            print(f"执行拖拽动作 {i+1}/{len(actions_sequence)}")
            
            for _ in range(25):
                obs, reward, terminate = task_env.step(action)
                
                if terminate:
                    print("拖拽任务完成!")
                    return
                    
                time.sleep(0.04)
        
        print("拖拽任务演示完成")
    
    def demo_custom_movement_pattern(self):
        """演示自定义运动模式"""
        print("\n=== 自定义运动模式演示 ===")
        
        task_env = self.env.get_task(BasketballInHoop)
        descriptions, obs = task_env.reset()
        
        # 获取起始位置
        center_pos = obs.gripper_pose[:3]
        print(f"中心位置: {center_pos}")
        
        # 生成圆形轨迹
        radius = 0.15
        num_points = 20
        
        print("执行圆形轨迹运动...")
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            target_x = center_pos[0] + radius * np.cos(angle)
            target_y = center_pos[1] + radius * np.sin(angle)
            target_z = center_pos[2] + 0.1 * np.sin(2 * angle)  # 添加垂直波动
            
            action = np.array([target_x, target_y, target_z, 0, 0, 0, 1, 1])
            
            print(f"移动到点 {i+1}/{num_points}: [{target_x:.3f}, {target_y:.3f}, {target_z:.3f}]")
            
            for _ in range(15):
                obs, reward, terminate = task_env.step(action)
                time.sleep(0.02)
        
        # 返回中心位置
        action = np.array([center_pos[0], center_pos[1], center_pos[2], 0, 0, 0, 1, 1])
        for _ in range(20):
            obs, reward, terminate = task_env.step(action)
            time.sleep(0.02)
        
        print("圆形轨迹运动完成")
    
    def demo_precise_manipulation(self):
        """演示精确操作"""
        print("\n=== 精确操作演示 ===")
        
        task_env = self.env.get_task(BasketballInHoop)
        descriptions, obs = task_env.reset()
        
        start_pos = obs.gripper_pose[:3]
        
        # 执行精确的定点操作
        print("执行精确定点操作...")
        
        # 定义精确的控制点
        precision_points = [
            [start_pos[0] + 0.05, start_pos[1], start_pos[2]],      # 右移5cm
            [start_pos[0] + 0.05, start_pos[1] + 0.05, start_pos[2]], # 前移5cm
            [start_pos[0], start_pos[1] + 0.05, start_pos[2]],      # 左移5cm
            [start_pos[0], start_pos[1], start_pos[2] + 0.05],      # 上移5cm
            [start_pos[0], start_pos[1], start_pos[2]],             # 回到起始位置
        ]
        
        for i, point in enumerate(precision_points):
            print(f"精确移动到点 {i+1}: {point}")
            
            action = np.array(point + [0, 0, 0, 1, 1])
            
            # 慢速精确移动
            for step in range(30):
                obs, reward, terminate = task_env.step(action)
                
                # 计算距离目标的距离
                current_pos = obs.gripper_pose[:3]
                distance = np.linalg.norm(np.array(current_pos) - np.array(point))
                
                if step % 10 == 0:
                    print(f"  步数: {step}, 距离目标: {distance:.4f}m")
                
                if distance < 0.01:  # 1cm精度
                    print(f"  精确到达! 最终距离: {distance:.4f}m")
                    break
                    
                time.sleep(0.05)
            
            time.sleep(0.5)  # 在每个点停留0.5秒
        
        print("精确操作演示完成")
    
    def run_all_demos(self):
        """运行所有演示"""
        try:
            self.create_environment()
            
            # 运行各种演示
            self.demo_basketball_task()
            time.sleep(2)
            
            self.demo_reach_and_drag_task()
            time.sleep(2)
            
            self.demo_custom_movement_pattern()
            time.sleep(2)
            
            self.demo_precise_manipulation()
            
        except Exception as e:
            print(f"演示过程中出错: {e}")
        
        finally:
            if self.env:
                self.env.shutdown()
                print("环境已关闭")


def main():
    """主函数"""
    print("Colosseum 机械臂任务演示")
    print("=" * 50)
    
    while True:
        print("\n请选择演示:")
        print("1. 篮球投篮任务")
        print("2. 伸手拖拽任务")
        print("3. 自定义运动模式")
        print("4. 精确操作演示")
        print("5. 运行所有演示")
        print("6. 退出")
        
        choice = input("请输入选择 (1-6): ").strip()
        
        demo = RobotTaskDemo(headless=False)
        
        try:
            if choice == "1":
                demo.create_environment()
                demo.demo_basketball_task()
            elif choice == "2":
                demo.create_environment()
                demo.demo_reach_and_drag_task()
            elif choice == "3":
                demo.create_environment()
                demo.demo_custom_movement_pattern()
            elif choice == "4":
                demo.create_environment()
                demo.demo_precise_manipulation()
            elif choice == "5":
                demo.run_all_demos()
            elif choice == "6":
                print("退出程序")
                break
            else:
                print("无效选择，请重新输入")
                continue
                
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            if demo.env:
                demo.env.shutdown()


if __name__ == "__main__":
    # main()
    demo = RobotTaskDemo(headless=False)
    demo.create_environment()
    demo.demo_basketball_task()

