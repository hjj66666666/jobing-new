import time
import argparse
from config.config import Config

"""
旋转时间测试脚本

这个脚本用于测试和优化车辆旋转时间参数，通过不同的速度和角度设置，
帮助用户找到最佳的旋转时间计算公式。

使用方法:
1. 基本使用: python test_rotation_time.py
2. 指定速度和角度: python test_rotation_time.py --speed 0.3 --angle 90
3. 多次测试取平均: python test_rotation_time.py --speed 0.3 --angle 90 --times 5
"""

def calculate_rotation_time(speed, angle):
    """
    根据当前配置计算旋转时间
    
    Args:
        speed: 旋转速度 (0-1)
        angle: 旋转角度 (度)
    
    Returns:
        float: 计算的旋转时间（秒）
    """
    # 当前使用的计算公式
    rotation_time = angle / (speed * 100)
    # 限制在合理范围内
    rotation_time = max(0.5, min(2.0, rotation_time))
    return rotation_time

def simulate_rotation(speed, angle, test_times=1):
    """
    模拟旋转测试并返回建议的旋转时间
    
    Args:
        speed: 旋转速度 (0-1)
        angle: 旋转角度 (度)
        test_times: 测试次数，取平均值
    
    Returns:
        dict: 包含测试结果的字典
    """
    print(f"\n开始测试旋转参数 - 速度: {speed}, 角度: {angle}度")
    print(f"将进行 {test_times} 次测试并取平均值")
    
    # 计算理论旋转时间
    theoretical_time = calculate_rotation_time(speed, angle)
    print(f"理论旋转时间: {theoretical_time:.2f} 秒 (使用公式: 角度/(速度*100))")
    
    # 收集实际旋转时间建议（这里使用模拟值，实际应用中应替换为真实测量）
    print("\n请手动执行旋转操作并计时:")
    print("1. 执行旋转命令")
    print("2. 测量从开始到旋转完成所需的实际时间")
    print("3. 输入您测量的实际时间")
    
    total_actual_time = 0
    
    for i in range(test_times):
        try:
            actual_time = float(input(f"请输入第 {i+1} 次测试的实际旋转时间（秒）: "))
            total_actual_time += actual_time
        except ValueError:
            print("输入无效，请输入数字。使用默认值 1.0 秒。")
            total_actual_time += 1.0
    
    average_actual_time = total_actual_time / test_times
    
    # 计算误差
    error_percentage = ((average_actual_time - theoretical_time) / theoretical_time) * 100
    
    # 根据误差提供建议的参数调整
    suggested_coefficient = (angle / average_actual_time) / speed if speed > 0 else 100
    
    results = {
        "speed": speed,
        "angle": angle,
        "test_times": test_times,
        "theoretical_time": theoretical_time,
        "actual_time": average_actual_time,
        "error_percentage": error_percentage,
        "suggested_coefficient": suggested_coefficient
    }
    
    # 打印测试结果和建议
    print("\n===== 测试结果 =====")
    print(f"测试速度: {speed}")
    print(f"测试角度: {angle}度")
    print(f"理论计算时间: {theoretical_time:.2f} 秒")
    print(f"实际测量时间: {average_actual_time:.2f} 秒")
    print(f"误差百分比: {error_percentage:+.2f}%")
    print(f"建议的系数: {suggested_coefficient:.2f}")
    print("\n===== 建议调整 =====")
    
    if abs(error_percentage) < 10:
        print("当前公式比较准确，不需要调整。")
    else:
        print(f"建议将公式中的系数 100 调整为: {suggested_coefficient:.2f}")
        print(f"修改 config.py 中的 rotation_time 计算逻辑为:")
        print(f"rotation_time = angle / (speed * {suggested_coefficient:.2f})")
    
    return results

def find_optimal_parameters():
    """
    帮助用户找到最优的旋转参数
    """
    print("\n===== 旋转参数优化向导 =====")
    print("这个向导将帮助您找到最佳的旋转时间计算参数")
    
    speeds = [0.2, 0.3, 0.4, 0.5]
    angles = [45, 90, 180]
    
    all_results = []
    
    for speed in speeds:
        for angle in angles:
            print(f"\n\n=== 测试组合: 速度={speed}, 角度={angle}度 ===")
            result = simulate_rotation(speed, angle, test_times=3)
            all_results.append(result)
    
    # 计算平均建议系数
    valid_coefficients = [r['suggested_coefficient'] for r in all_results]
    avg_coefficient = sum(valid_coefficients) / len(valid_coefficients)
    
    print("\n\n===== 参数优化总结 =====")
    print(f"基于所有测试的平均建议系数: {avg_coefficient:.2f}")
    print(f"建议的最终计算公式: rotation_time = angle / (speed * {avg_coefficient:.2f})")
    print(f"修改 config.py 中的 rotation_time 计算逻辑以获得更准确的旋转时间。")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试和优化车辆旋转时间参数')
    parser.add_argument('--speed', type=float, default=Config.search_speed, 
                        help=f'旋转速度 (默认: {Config.search_speed})')
    parser.add_argument('--angle', type=float, default=Config.search_rotation_angle, 
                        help=f'旋转角度 (默认: {Config.search_rotation_angle}度)')
    parser.add_argument('--times', type=int, default=1, 
                        help='测试次数，取平均值 (默认: 1)')
    parser.add_argument('--optimize', action='store_true', 
                        help='运行完整的参数优化向导')
    
    args = parser.parse_args()
    
    # 打印当前配置信息
    print("===== 旋转时间测试工具 =====")
    print(f"当前配置: 搜索速度 = {Config.search_speed}, 搜索旋转角度 = {Config.search_rotation_angle}度")
    
    if args.optimize:
        # 运行参数优化向导
        find_optimal_parameters()
    else:
        # 运行单次测试
        simulate_rotation(args.speed, args.angle, args.times)
    
    print("\n测试完成! 根据测试结果调整 controller.py 中的旋转时间计算公式以获得最佳性能。")