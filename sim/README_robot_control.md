# 机械臂仿真环境控制指南

本指南演示如何使用Colosseum框架创建包含机械臂的仿真环境，并提供控制接口让机械臂移动到指定位置。

## 📋 文件说明

- `robot_arm_simulation.py` - 完整的机械臂控制类和演示程序
- `simple_robot_control.py` - 简化版本的基础控制示例
- `simple_arm_config.yaml` - 环境配置文件

## 🚀 快速开始

### 1. 安装依赖

确保您已经安装了Colosseum及其依赖：

```bash
pip install -e .
```

### 2. 运行简单示例

```bash
# 运行基础控制示例
python simple_robot_control.py

# 或运行完整的交互式演示
python robot_arm_simulation.py
```

## 🎯 核心功能

### 机械臂控制器类

`RobotArmController` 类提供以下主要功能：

#### 1. 环境初始化
```python
controller = RobotArmController(headless=False, robot_setup="panda")
controller.launch()
```

#### 2. 位置控制
```python
# 移动到指定的笛卡尔坐标位置
target_position = [0.5, 0.2, 0.8]  # [x, y, z]
success = controller.move_to_position(target_position)
```

#### 3. 关节控制
```python
# 通过关节角度控制机械臂
joint_angles = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785]
success = controller.move_joints(joint_angles)
```

#### 4. 状态获取
```python
# 获取当前末端执行器位置和朝向
position, quaternion = controller.get_current_pose()

# 获取完整观察数据
observation = controller.get_observation()
```

## 🛠️ 技术细节

### 动作模式

该框架支持多种动作模式：

1. **末端执行器位置控制**: `EndEffectorPoseViaPlanning()`
   - 通过指定目标位置和朝向来控制机械臂
   - 动作格式: `[x, y, z, qx, qy, qz, qw, gripper_action]`

2. **关节位置控制**: `JointPosition()`
   - 直接控制每个关节的角度
   - 动作格式: `[joint1, joint2, ..., joint7, gripper_action]`

3. **关节速度控制**: `JointVelocity()`
   - 控制每个关节的运动速度

### 机器人类型

支持的机器人类型（通过`robot_setup`参数）：
- `"panda"` - Franka Panda机械臂（默认）
- 其他RLBench支持的机器人类型

### 观察空间

系统提供丰富的观察信息：
- 机械臂关节位置和速度
- 末端执行器位置和朝向
- 多视角相机图像（RGB/深度）
- 任务相关的状态信息

## 📝 使用示例

### 示例1：基础移动

```python
from robot_arm_simulation import RobotArmController

# 创建控制器
controller = RobotArmController(headless=False)
controller.launch()

# 获取当前位置
current_pos, _ = controller.get_current_pose()

# 向右移动10cm
target_pos = [current_pos[0] + 0.1, current_pos[1], current_pos[2]]
success = controller.move_to_position(target_pos)

if success:
    print("移动成功!")

controller.shutdown()
```

### 示例2：序列动作

```python
# 定义一系列目标位置
waypoints = [
    [0.5, 0.2, 0.8],
    [0.6, 0.3, 0.9],
    [0.4, 0.1, 0.7]
]

controller = RobotArmController()
controller.launch()

# 依次移动到各个点
for i, waypoint in enumerate(waypoints):
    print(f"移动到waypoint {i+1}: {waypoint}")
    success = controller.move_to_position(waypoint)
    if not success:
        print(f"移动到waypoint {i+1}失败")
        break
    time.sleep(1)  # 暂停1秒

controller.shutdown()
```

### 示例3：关节空间控制

```python
# 预定义的关节配置
home_position = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]
ready_position = [0.5, -0.5, 0.2, -1.5, 0.1, 1.0, 0.785]

controller = RobotArmController()
controller.launch()

# 移动到准备位置
controller.move_joints(ready_position)
time.sleep(2)

# 返回初始位置
controller.move_joints(home_position)

controller.shutdown()
```

## ⚙️ 配置选项

### 环境配置

在`simple_arm_config.yaml`中可以配置：

- 图像分辨率
- 相机设置
- 渲染选项
- 传感器数据类型

### 高级配置

- **变化因子**: 通过`scene.factors`添加环境随机化
- **多机器人**: 通过`robot_setup`选择不同机器人
- **自定义任务**: 继承`Task`类创建自定义操作任务

## 🔧 故障排除

### 常见问题

1. **环境启动失败**
   - 检查CoppeliaSim是否正确安装
   - 确认依赖包版本兼容

2. **机械臂不移动**
   - 检查目标位置是否在工作空间内
   - 验证动作格式是否正确

3. **图形界面不显示**
   - 确认`headless=False`
   - 检查显示器配置

### 调试技巧

```python
# 启用详细输出
controller = RobotArmController(headless=False)
controller.launch()

# 检查当前状态
obs = controller.get_observation()
print(f"关节位置: {obs.joint_positions}")
print(f"末端位置: {obs.gripper_pose}")

# 小步长移动测试
current_pos, quat = controller.get_current_pose()
small_step = [current_pos[0] + 0.01, current_pos[1], current_pos[2]]
controller.move_to_position(small_step)
```

## 📚 扩展阅读

- [RLBench文档](https://github.com/stepjam/RLBench)
- [PyRep文档](https://github.com/stepjam/PyRep)
- [Colosseum论文](https://arxiv.org/abs/2402.08191)

## 🤝 贡献

欢迎提交问题和改进建议！

