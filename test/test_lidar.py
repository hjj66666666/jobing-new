#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPLidar A1 激光雷达测试脚本
用于验证激光雷达硬件连接和基本功能

使用方法：
    python test_lidar.py

需要：
    - RPLidar A1 硬件通过 USB 连接
    - 可能需要安装 rplidar-roboticia 库: pip install rplidar-roboticia
"""

import sys
import os
import time
import math

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("RPLidar A1 激光雷达测试")
print("=" * 60)

# 1. 检查串口设备
print("\n步骤 1: 检查串口设备...")
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
print(f"可用串口设备：")
if ports:
    for port in ports:
        print(f"  - {port.device}: {port.description}")
else:
    print("  未找到串口设备")

# 常见 RPLidar 串口名称
possible_lidar_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
print(f"\n常见 RPLidar 串口位置：")
for port_name in possible_lidar_ports:
    if os.path.exists(port_name):
        print(f"  ✓ {port_name} 存在")
    else:
        print(f"  ✗ {port_name} 不存在")

# 2. 尝试导入 RPLidar 库
print("\n步骤 2: 检查 RPLidar 驱动库...")
RPLIDAR_AVAILABLE = False
try:
    from rplidar import RPLidar
    RPLIDAR_AVAILABLE = True
    print("✓ rplidar-roboticia 库已安装")
except ImportError:
    try:
        # 尝试其他可能的导入方式
        import rplidar_roboticia
        RPLIDAR_AVAILABLE = True
        print("✓ rplidar 库已安装（其他版本）")
    except ImportError:
        print("✗ RPLidar 驱动库未安装")
        print("  安装方法: pip install rplidar-roboticia")
        print("  或从源码安装: https://github.com/SkoltechRobotics/rplidar")

# 3. 如果库可用，尝试连接激光雷达
if RPLIDAR_AVAILABLE:
    print("\n步骤 3: 尝试连接激光雷达...")
    
    # 尝试常见的串口
    lidar_port = None
    for port_name in possible_lidar_ports:
        if os.path.exists(port_name):
            try:
                print(f"尝试连接 {port_name}...")
                lidar = RPLidar(port_name)
                info = lidar.get_info()
                print(f"✓ 成功连接到 {port_name}")
                print(f"  设备信息: {info}")
                lidar_port = port_name
                lidar.stop()
                lidar.disconnect()
                break
            except Exception as e:
                print(f"  ✗ {port_name} 连接失败: {e}")
                continue
    
    if lidar_port:
        print("\n步骤 4: 测试数据读取（5秒）...")
        try:
            lidar = RPLidar(lidar_port)
            lidar.start_motor()
            time.sleep(2)  # 等待电机启动
            
            print("开始读取数据...")
            scan_count = 0
            start_time = time.time()
            
            try:
                for scan in lidar.iter_scans(max_buf_meas=500):
                    scan_count += 1
                    if scan:
                        # 显示第一组扫描数据
                        print(f"\n扫描 #{scan_count}:")
                        print(f"  数据点数: {len(scan)}")
                        if len(scan) > 0:
                            # 显示前5个点
                            print("  前5个点 (quality, angle_deg, distance_mm):")
                            for i, (quality, angle, distance) in enumerate(scan[:5]):
                                print(f"    {i+1}. quality={quality}, angle={angle:.1f}°, distance={distance:.0f}mm")
                        
                        # 检查数据范围
                        if len(scan) > 10:
                            distances = [d for _, _, d in scan]
                            angles = [a for _, a, _ in scan]
                            print(f"  距离范围: {min(distances):.0f}mm - {max(distances):.0f}mm")
                            print(f"  角度范围: {min(angles):.1f}° - {max(angles):.1f}°")
                    
                    # 运行5秒
                    if time.time() - start_time > 5:
                        break
                        
            except KeyboardInterrupt:
                print("\n用户中断")
            finally:
                lidar.stop_motor()
                lidar.stop()
                lidar.disconnect()
                print(f"\n✓ 测试完成，共读取 {scan_count} 次扫描")
                
        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  未找到可用的激光雷达串口")
        print("请检查：")
        print("  1. RPLidar A1 是否通过 USB 连接")
        print("  2. 串口权限是否正确（运行: sudo usermod -a -G dialout $USER）")
        print("  3. 串口号是否正确")
else:
    print("\n⚠️  RPLidar 驱动库未安装，无法测试硬件")
    print("\n安装步骤：")
    print("  1. pip install rplidar-roboticia")
    print("  2. 重新运行此测试脚本")

# 4. 测试 SLAM 模块集成
print("\n步骤 5: 测试 SLAM 模块兼容性...")
try:
    from modules.slam import SlamSystem
    slam = SlamSystem()
    if slam.running:
        print("✓ SLAM 模块已初始化")
        print(f"  地图大小: {slam.MAP_SIZE_PIXELS}x{slam.MAP_SIZE_PIXELS} 像素")
        print(f"  地图尺寸: {slam.MAP_SIZE_PIXELS * slam.MAP_METERS_PER_PIXEL:.1f}m x {slam.MAP_SIZE_PIXELS * slam.MAP_METERS_PER_PIXEL:.1f}m")
        print(f"  扫描点数: {slam.SCAN_SIZE}")
        
        # 测试数据格式
        print("\n  测试数据格式兼容性...")
        # 创建模拟数据
        mock_scan = [(255, i, 1000 + i * 10) for i in range(0, 360, 10)]  # 每10度一个点
        try:
            slam.update(mock_scan)
            print("  ✓ 数据格式兼容（模拟数据测试成功）")
        except Exception as e:
            print(f"  ✗ 数据格式不兼容: {e}")
    else:
        print("✗ SLAM 模块未运行（可能 BreezySLAM 未安装）")
except Exception as e:
    print(f"✗ SLAM 模块测试失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

