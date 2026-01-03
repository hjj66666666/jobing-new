#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLAM 可视化测试脚本
用于测试 SLAM 功能并可视化生成的地图

使用方法：
    1. 使用真实激光雷达：python test_slam_visualization.py --lidar-port /dev/ttyUSB0
    2. 使用模拟数据：python test_slam_visualization.py --mock

需要：
    - BreezySLAM 已安装
    - 如果使用真实激光雷达：rplidar-roboticia 库和硬件连接
    - OpenCV 用于可视化
"""

import sys
import os
import time
import argparse
import threading
import math

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from config.config import Config

# 设置 map_path（如果配置中没有）
if not hasattr(Config, 'map_path'):
    Config.map_path = './slam_map.png'

print("=" * 60)
print("SLAM 可视化测试")
print("=" * 60)

def test_with_mock_data():
    """使用模拟数据测试 SLAM"""
    print("\n使用模拟数据测试 SLAM...")
    
    from modules.slam import SlamSystem
    
    slam = SlamSystem()
    if not slam.running:
        print("✗ SLAM 系统未初始化")
        return
    
    print("✓ SLAM 系统已初始化")
    print(f"  地图大小: {slam.MAP_SIZE_PIXELS}x{slam.MAP_SIZE_PIXELS} 像素")
    print(f"  地图尺寸: {slam.MAP_SIZE_PIXELS * slam.MAP_METERS_PER_PIXEL:.1f}m x {slam.MAP_SIZE_PIXELS * slam.MAP_METERS_PER_PIXEL:.1f}m")
    
    # 创建模拟环境（一个方形房间，10m x 10m）
    print("\n生成模拟激光雷达数据...")
    
    # 模拟一个方形房间：中心在原点，边长 10m
    room_size = 10000  # 10m in mm
    num_scans = 50
    
    for scan_num in range(num_scans):
        # 模拟机器人在房间内移动（圆形路径）
        t = scan_num / num_scans * 2 * math.pi
        robot_x = 2000 * math.cos(t)  # 半径2m的圆
        robot_y = 2000 * math.sin(t)
        robot_theta = t * 180 / math.pi
        
        # 生成360度扫描数据
        scan_data = []
        for angle_deg in range(0, 360, 1):  # 每1度一个点
            angle_rad = math.radians(angle_deg)
            
            # 计算从机器人位置出发的射线
            ray_x = robot_x
            ray_y = robot_y
            ray_dir_x = math.cos(angle_rad)
            ray_dir_y = math.sin(angle_rad)
            
            # 检测与房间边界的交点
            distance = 0
            max_dist = 12000  # 12m max range
            
            # 简化的边界检测（方形房间）
            for step in range(0, max_dist, 50):  # 每50mm检测一次
                test_x = ray_x + ray_dir_x * step
                test_y = ray_y + ray_dir_y * step
                
                # 检查是否超出房间边界（5m from center = 5000mm）
                if abs(test_x) > room_size/2 or abs(test_y) > room_size/2:
                    distance = step
                    break
            
            if distance == 0:
                distance = max_dist  # 未检测到障碍物
            
            # RPLidar 数据格式: (angle, distance, quality)
            quality = 255 if distance < max_dist else 0
            scan_data.append((angle_deg, distance, quality))
        
        # 更新 SLAM
        slam.update(scan_data)
        
        if (scan_num + 1) % 10 == 0:
            pose = slam.get_pose()
            print(f"  扫描 {scan_num + 1}/{num_scans}: 机器人位置 x={pose[0]:.2f}m, y={pose[1]:.2f}m, theta={pose[2]:.1f}°")
        
        time.sleep(0.1)  # 模拟扫描间隔
    
    print("\n模拟数据测试完成")
    return slam

def test_with_real_lidar(lidar_port):
    """使用真实激光雷达测试 SLAM"""
    print(f"\n使用真实激光雷达测试 SLAM (端口: {lidar_port})...")
    
    # 检查 RPLidar 库
    try:
        from rplidar import RPLidar
    except ImportError:
        print("✗ RPLidar 库未安装")
        print("  安装方法: pip install rplidar-roboticia")
        return None
    
    from modules.slam import SlamSystem
    
    slam = SlamSystem()
    if not slam.running:
        print("✗ SLAM 系统未初始化")
        return None
    
    print("✓ SLAM 系统已初始化")
    
    # 连接激光雷达
    try:
        lidar = RPLidar(lidar_port)
        info = lidar.get_info()
        print(f"✓ 激光雷达连接成功: {info}")
    except Exception as e:
        print(f"✗ 激光雷达连接失败: {e}")
        return None
    
    # 启动电机
    try:
        lidar.start_motor()
        time.sleep(2)  # 等待电机启动
        print("✓ 激光雷达电机已启动")
    except Exception as e:
        print(f"✗ 启动电机失败: {e}")
        lidar.disconnect()
        return None
    
    # 读取数据并更新 SLAM
    scan_count = 0
    max_scans = 100  # 最多读取100次扫描
    
    print(f"\n开始读取数据（最多 {max_scans} 次扫描，按 Ctrl+C 提前停止）...")
    
    try:
        for scan in lidar.iter_scans(max_buf_meas=500):
            scan_count += 1
            
            # RPLidar.iter_scans() 返回格式: [(quality, angle, distance), ...]
            # SLAM.update() 期望格式: [(angle, distance, quality), ...]
            # 需要转换顺序
            converted_scan = [(angle, distance, quality) for quality, angle, distance in scan]
            
            slam.update(converted_scan)
            
            if scan_count % 10 == 0:
                pose = slam.get_pose()
                print(f"  扫描 {scan_count}: 机器人位置 x={pose[0]:.2f}m, y={pose[1]:.2f}m, theta={pose[2]:.1f}°")
            
            if scan_count >= max_scans:
                break
                
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n读取数据失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
        print(f"✓ 完成，共读取 {scan_count} 次扫描")
    
    return slam

def visualize_map(slam, window_name="SLAM Map", save_path=None):
    """可视化 SLAM 地图"""
    print("\n生成可视化地图...")
    
    map_img = slam.get_map()
    if map_img is None:
        print("⚠️  地图尚未生成，等待更新...")
        time.sleep(2)
        map_img = slam.get_map()
    
    if map_img is None:
        print("✗ 无法获取地图图像")
        return
    
    # 获取机器人位姿
    pose = slam.get_pose()
    
    # 在地图上添加信息文本
    display_img = map_img.copy()
    
    # 添加文本信息
    info_text = [
        f"Robot Pose: x={pose[0]:.2f}m, y={pose[1]:.2f}m, theta={pose[2]:.1f}°",
        f"Map Size: {slam.MAP_SIZE_PIXELS}x{slam.MAP_SIZE_PIXELS} pixels",
        f"Map Scale: {slam.MAP_METERS_PER_PIXEL*100:.1f} cm/pixel",
        "Press 'q' to quit, 's' to save"
    ]
    
    y_offset = 20
    for i, text in enumerate(info_text):
        cv2.putText(display_img, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_img, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # 显示地图
    print("✓ 地图可视化窗口已打开")
    print("  按键说明:")
    print("    'q' - 退出")
    print("    's' - 保存地图")
    print("    'r' - 刷新地图")
    
    while True:
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = save_path or Config.map_path
            cv2.imwrite(save_path, map_img)
            print(f"✓ 地图已保存到: {save_path}")
        elif key == ord('r'):
            # 刷新地图
            map_img = slam.get_map()
            if map_img is not None:
                display_img = map_img.copy()
                # 重新添加文本
                y_offset = 20
                for i, text in enumerate(info_text):
                    cv2.putText(display_img, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_img, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                print("✓ 地图已刷新")
    
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='SLAM 可视化测试')
    parser.add_argument('--lidar-port', type=str, default=None,
                       help='激光雷达串口（如 /dev/ttyUSB0），如果指定则使用真实雷达')
    parser.add_argument('--mock', action='store_true',
                       help='使用模拟数据（默认）')
    parser.add_argument('--save', type=str, default=None,
                       help='保存地图的文件路径（默认: ./slam_map.png）')
    parser.add_argument('--scans', type=int, default=50,
                       help='模拟数据的扫描次数（仅 mock 模式）')
    
    args = parser.parse_args()
    
    slam = None
    
    # 选择测试模式
    if args.lidar_port:
        # 使用真实激光雷达
        if not os.path.exists(args.lidar_port):
            print(f"✗ 串口设备不存在: {args.lidar_port}")
            return
        slam = test_with_real_lidar(args.lidar_port)
    else:
        # 使用模拟数据
        slam = test_with_mock_data()
    
    if slam is None:
        print("\n✗ SLAM 测试失败")
        return
    
    # 可视化地图
    visualize_map(slam, save_path=args.save)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

