#!/usr/bin/env python3
"""
测试扩散策略推理系统的基本功能
"""

import os
import torch
import numpy as np
from PIL import Image

def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试模型加载...")
    
    # 检查是否有训练好的模型
    model_paths = [
        '.diffusion_policy/my_model/best_model.pth',
        './diffusion_policy/checkpoints/best_model.pth'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练脚本生成模型")
        return False
    
    print(f"✓ 找到模型文件: {model_path}")
    
    try:
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✓ 模型加载成功")
        print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  - 最佳验证损失: {checkpoint.get('best_val_loss', 'unknown')}")
        print(f"  - 模型参数数量: {len(checkpoint['model_state_dict'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_diffusion_inference():
    """测试扩散推理"""
    print("\n" + "=" * 50)
    print("测试扩散推理逻辑...")
    
    try:
        from diffusion_model import create_diffusion_policy
        
        # 创建测试模型
        model = create_diffusion_policy(
            action_dim=7,
            action_horizon=2,
            state_dim=15,
            vision_feature_dim=256,
            hidden_dim=256,
            num_diffusion_steps=50,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        model.eval()
        print("✓ 扩散模型创建成功")
        
        # 创建测试输入
        batch_size = 1
        sequence_length = 4
        
        # 模拟图像输入
        images = torch.randn(batch_size, sequence_length, 3, 224, 224)
        
        # 模拟机器人状态
        robot_states = torch.randn(batch_size, sequence_length, 15)
        
        print(f"✓ 测试输入创建完成")
        print(f"  - 图像形状: {images.shape}")
        print(f"  - 状态形状: {robot_states.shape}")
        
        # 测试采样
        with torch.no_grad():
            actions = model.sample(images, robot_states, num_inference_steps=10)
        
        print(f"✓ 扩散采样成功")
        print(f"  - 动作形状: {actions.shape}")
        print(f"  - 动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 扩散推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_preprocessing():
    """测试图像预处理"""
    print("\n" + "=" * 50)
    print("测试图像预处理...")
    
    try:
        import torchvision.transforms as transforms
        
        # 创建图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        image_pil = Image.fromarray(test_image)
        
        # 预处理
        processed = transform(image_pil)
        
        print(f"✓ 图像预处理成功")
        print(f"  - 原始图像形状: {test_image.shape}")
        print(f"  - 处理后形状: {processed.shape}")
        print(f"  - 数值范围: [{processed.min():.3f}, {processed.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像预处理测试失败: {e}")
        return False

def test_colosseum_imports():
    """测试Colosseum相关导入"""
    print("\n" + "=" * 50)
    print("测试Colosseum导入...")
    
    try:
        # 测试基础导入
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        print("✓ RLBench基础模块导入成功")
        
        # 测试Colosseum扩展
        from colosseum import ASSETS_CONFIGS_FOLDER, TASKS_PY_FOLDER, TASKS_TTM_FOLDER
        from colosseum.rlbench.extensions.environment import EnvironmentExt
        from colosseum.rlbench.utils import ObservationConfigExt, name_to_class
        print("✓ Colosseum扩展模块导入成功")
        
        # 测试任务类查找
        task_class = name_to_class("basketball_in_hoop", TASKS_PY_FOLDER)
        if task_class is not None:
            print("✓ basketball_in_hoop任务类找到")
        else:
            print("❌ basketball_in_hoop任务类未找到")
            return False
        
        print(f"  - 配置文件夹: {ASSETS_CONFIGS_FOLDER}")
        print(f"  - 任务Python文件夹: {TASKS_PY_FOLDER}")
        print(f"  - 任务TTM文件夹: {TASKS_TTM_FOLDER}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Colosseum导入失败: {e}")
        print("请确保已正确安装Colosseum和RLBench")
        return False
    except Exception as e:
        print(f"❌ Colosseum测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🧪 Diffusion Policy推理系统测试")
    print("=" * 60)
    
    tests = [
        ("模型加载", test_model_loading),
        ("扩散推理", test_diffusion_inference),
        ("图像预处理", test_image_preprocessing),
        ("Colosseum导入", test_colosseum_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15} : {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！可以运行推理脚本")
    else:
        print("⚠️  存在失败的测试，请检查环境配置")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
