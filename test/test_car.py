"""
测试车辆移动功能
"""
import time
from modules.hardware import init_arduino

def test_car_movement():
    print("开始测试车辆移动功能")
    
    # 初始化Arduino
    car_move_func, _ = init_arduino()
    
    # 测试不同方向和速度
    test_cases = [
        ("front", 0.5),
        ("stop", 0),
        ("back", 0.5),
        ("stop", 0),
        ("left", 0.5),
        ("stop", 0),
        ("right", 0.5),
        ("stop", 0)
    ]
    
    for direction, speed in test_cases:
        print(f"\n测试: 方向={direction}, 速度={speed}")
        print(f"发送命令: {direction}, {speed}")
        
        # 发送移动命令
        result = car_move_func(direction, speed)
        print(f"命令结果: {result}")
        
        # 等待观察
        time.sleep(2)
    
    print("\n测试完成")

if __name__ == "__main__":
    test_car_movement()
