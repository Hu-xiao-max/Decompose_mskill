#!/usr/bin/env python3
"""
获取Panda机械臂的关节限制
"""

import numpy as np
from omegaconf import DictConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop


def get_joint_limits():
    """获取关节限制"""
    
    # 创建环境
    env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(),
            gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(DictConfig({
            'image_size': [128, 128],
            'images': {'rgb': False, 'depth': False, 'mask': False, 'point_cloud': False},
            'cameras': {},
            'depth_in_meters': False,
            'masks_as_one_channel': True,
            'renderer': 'opengl3'
        })),
        headless=True,
        robot_setup="panda"
    )
    
    try:
        env.launch()
        task_env = env.get_task(BasketballInHoop)
        descriptions, obs = task_env.reset()
        
        # 获取机器人
        robot = task_env._scene.robot
        arm = robot.arm
        
        print("=== Panda机械臂关节限制 ===")
        print(f"关节数量: {len(arm.joints)}")
        
        # 获取当前关节位置
        current_joints = arm.get_joint_positions()
        print("\n当前关节位置:")
        for i, pos in enumerate(current_joints):
            print(f"关节{i}: {pos:.4f} rad ({pos*180/np.pi:.1f}°)")
        
        # 获取关节限制
        print("\n关节运动范围:")
        try:
            # 方法1: 尝试使用get_joint_intervals()
            joint_ranges = arm.get_joint_intervals()
            print(f"获取到 {len(joint_ranges)} 个关节范围")
            
            for i, range_info in enumerate(joint_ranges):
                if isinstance(range_info, (list, tuple)) and len(range_info) == 2:
                    min_val, max_val = range_info
                    print(f"关节{i}: [{min_val:.4f}, {max_val:.4f}] rad = [{min_val*180/np.pi:.1f}°, {max_val*180/np.pi:.1f}°]")
                else:
                    print(f"关节{i}: 范围格式异常 - {range_info}")
                    
        except Exception as e:
            print(f"无法使用get_joint_intervals(): {e}")
            
            # 方法2: 逐个关节获取
            print("尝试逐个关节获取限制:")
            for i, joint in enumerate(arm.joints):
                try:
                    interval = joint.get_joint_interval()
                    if interval and len(interval) == 2:
                        min_val, max_val = interval
                        print(f"关节{i}: [{min_val:.4f}, {max_val:.4f}] rad = [{min_val*180/np.pi:.1f}°, {max_val*180/np.pi:.1f}°]")
                    else:
                        print(f"关节{i}: 无有效范围信息")
                except Exception as joint_e:
                    print(f"关节{i}: 获取失败 - {joint_e}")
        
        # 检查零位
        print("\n=== 零位可达性检查 ===")
        zero_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 尝试设置零位看是否报错
        try:
            # 保存当前位置
            original_pos = arm.get_joint_positions()
            
            # 尝试设置零位
            arm.set_joint_target_positions(zero_joints)
            print("设置零位目标位置: 成功")
            
            # 恢复原位置
            arm.set_joint_target_positions(original_pos)
            
        except Exception as e:
            print(f"设置零位目标位置: 失败 - {e}")
        
        # 显示Panda的标准关节限制（从文档获得）
        print("\n=== Panda标准关节限制（参考） ===")
        panda_standard_limits = [
            (-2.8973, 2.8973),   # joint1: ±166°
            (-1.7628, 1.7628),   # joint2: ±101°  
            (-2.8973, 2.8973),   # joint3: ±166°
            (-3.0718, -0.0698),  # joint4: -176° to -4° (注意：上限为负数!)
            (-2.8973, 2.8973),   # joint5: ±166°
            (-0.0175, 3.7525),   # joint6: -1° to +215°
            (-2.8973, 2.8973)    # joint7: ±166°
        ]
        
        for i, (min_val, max_val) in enumerate(panda_standard_limits):
            print(f"关节{i}: [{min_val:.4f}, {max_val:.4f}] rad = [{min_val*180/np.pi:.1f}°, {max_val*180/np.pi:.1f}°]")
            
            # 检查零位是否在范围内
            zero_in_range = min_val <= 0.0 <= max_val
            status = "✓" if zero_in_range else "✗"
            print(f"   零位检查: {status} {'可达' if zero_in_range else '不可达'}")
        
        print(f"\n关节4的限制说明: 上限为 {panda_standard_limits[3][1]:.4f} rad = {panda_standard_limits[3][1]*180/np.pi:.1f}°")
        print("这就是为什么 [0,0,0,0,0,0,0] 不可达的原因 - 关节4不能为0!")
        
        # 建议一个可达的位置
        print("\n=== 建议的可达位置 ===")
        safe_joints = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]  # 关节4设为-1.5，关节6设为1.5
        print(f"建议位置: {safe_joints}")
        print("对应角度:", [f"{j*180/np.pi:.1f}°" for j in safe_joints])
        
    finally:
        env.shutdown()


if __name__ == "__main__":
    get_joint_limits()