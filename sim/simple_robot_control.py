#!/usr/bin/env python3
"""
简化的机械臂控制 - 仅发送关节位置
"""

import numpy as np
from omegaconf import DictConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import ObservationConfigExt
from colosseum.rlbench.tasks.basketball_in_hoop import BasketballInHoop
import time


def main():
    """发送关节位置控制机械臂"""
    
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
        headless=False,
        robot_setup="panda"
    )
    
    env.launch()
    task_env = env.get_task(BasketballInHoop)
    descriptions, obs = task_env.reset()
    
    # 第一个目标关节位置
    target_joints_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # target_joints_1 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    action_1 = np.concatenate([target_joints_1, [1.0]])
    
    # 执行第一个动作
    print("执行第一个动作...")
    input("按Enter执行第一个动作...")
    for i in range(100):
        task_env.step(action_1)
        #time.sleep(0.01)
    

    env.shutdown()


if __name__ == "__main__":
    main()

