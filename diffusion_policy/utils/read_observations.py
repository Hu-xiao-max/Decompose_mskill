#!/usr/bin/env python3
"""
读取pkl文件中_observations的属性
"""

import os
import pickle


def read_observations_attributes(file_path: str):
    """读取_observations中的属性"""
    
    print(f"读取文件: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在")
        return
    
    try:
        with open(file_path, 'rb') as f:
            demo = pickle.load(f)
        
        print(f"✅ 文件读取成功")
        print(f"Demo类型: {type(demo)}")
        print(f"Demo长度: {len(demo)}")
        
        # 检查_observations属性
        if hasattr(demo, '_observations'):
            observations = demo._observations
            print(f"\n📊 _observations信息:")
            print(f"类型: {type(observations)}")
            print(f"长度: {len(observations)}")
            
            # 查看第一个观察对象的属性
            if len(observations) > 0:
                first_obs = observations[0]
                print(f"\n🔍 第一个观察对象:")
                print(f"类型: {type(first_obs)}")
                print(f"属性数量: {len(dir(first_obs))}")
                print(f"\n属性列表:")
                
                # 列出所有属性
                attrs = [attr for attr in dir(first_obs) if not attr.startswith('__')]
                for i, attr in enumerate(attrs, 1):
                    print(f"  {i:2d}. {attr}")
                
                # 显示所有属性的值
                print(f"\n🔧 所有属性值:")
                for attr in attrs:
                    if hasattr(first_obs, attr):
                        try:
                            value = getattr(first_obs, attr)
                            
                            if callable(value):
                                print(f"  {attr}: [方法/函数]")
                            elif value is None:
                                print(f"  {attr}: None")
                            elif hasattr(value, 'shape'):
                                # numpy数组
                                if value.size <= 20:  # 小数组显示完整内容
                                    print(f"  {attr}: 形状={value.shape}, 值={value}")
                                else:
                                    print(f"  {attr}: 形状={value.shape}, 范围=[{value.min():.3f}, {value.max():.3f}]")
                            elif isinstance(value, dict):
                                print(f"  {attr}: 字典, 键数量={len(value)}, 键={list(value.keys())}")
                            elif isinstance(value, (list, tuple)):
                                print(f"  {attr}: {type(value).__name__}, 长度={len(value)}")
                            elif isinstance(value, (int, float, bool, str)):
                                print(f"  {attr}: {value}")
                            else:
                                print(f"  {attr}: {type(value)}, 值={str(value)[:100]}...")
                        except Exception as e:
                            print(f"  {attr}: [获取失败: {e}]")
                    else:
                        print(f"  {attr}: 不存在")
        else:
            print("❌ 没有找到_observations属性")
            
    except Exception as e:
        print(f"❌ 读取失败: {e}")


def main():
    """主函数"""
    
    print("🔍 读取_observations属性工具")
    print("=" * 60)
    
    # 默认文件路径
    file_path = "/home/alien/simulation/robot-colosseum/dataset/basketball_in_hoop/basketball_in_hoop_0/variation0/episodes/episode0/low_dim_obs.pkl"
    
    read_observations_attributes(file_path)


if __name__ == "__main__":
    main()
