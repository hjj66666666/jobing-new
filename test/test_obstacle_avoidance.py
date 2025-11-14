#!/usr/bin/env python3
"""
障碍物避障系统测试脚本
用于测试基于OpenCV的障碍物检测和智能避障功能
"""

import cv2
import numpy as np
import time
from config.config import Config
from modules.vision import VisionSystem
from modules.controller import Controller

def test_obstacle_detection():
    """测试障碍物检测功能（改进版 - 深度信息验证）"""
    print("=== 测试障碍物检测功能（深度信息验证） ===")
    
    try:
        # 初始化视觉系统
        vision_system = VisionSystem()
        print("视觉系统初始化成功")
        
        # 测试障碍物检测
        for i in range(10):
            print(f"第 {i+1} 次测试...")
            
            # 获取图像
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = vision_system.vision_process()
            
            # 检测障碍物
            obstacles = vision_system.detect_obstacles_opencv(rgb, depth, aligned_depth_frame, depth_intrin)
            
            # 统计不同类型和关键性的障碍物
            depth_obstacles = [obs for obs in obstacles if obs['type'] == 'depth_obstacle']
            color_obstacles = [obs for obs in obstacles if obs['type'] == 'color_obstacle']
            edge_obstacles = [obs for obs in obstacles if obs['type'] == 'edge_obstacle']
            critical_obstacles = [obs for obs in obstacles if obs.get('is_critical', False)]
            
            print(f"检测到 {len(obstacles)} 个障碍物:")
            print(f"  - 深度障碍物: {len(depth_obstacles)} 个")
            print(f"  - 颜色障碍物: {len(color_obstacles)} 个")
            print(f"  - 边缘障碍物: {len(edge_obstacles)} 个")
            print(f"  - 关键障碍物: {len(critical_obstacles)} 个")
            
            for j, obstacle in enumerate(obstacles):
                critical_text = " [关键]" if obstacle.get('is_critical', False) else ""
                print(f"  障碍物 {j+1}: {obstacle['type']}, 距离={obstacle['distance']:.2f}m, "
                      f"置信度={obstacle['confidence']:.2f}{critical_text}")
            
            # 绘制障碍物
            if obstacles:
                rgb_display = vision_system.draw_obstacles(rgb_display, obstacles)
            
            # 显示图像和统计信息
            cv2.putText(rgb_display, f"Total: {len(obstacles)} | Critical: {len(critical_obstacles)}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(rgb_display, f"Depth: {len(depth_obstacles)} | Color: {len(color_obstacles)} | Edge: {len(edge_obstacles)}", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("障碍物检测测试（深度验证）", rgb_display)
            
            if cv2.waitKey(1000) & 0xFF == 27:  # ESC键退出
                break
                
        print("障碍物检测测试完成")
        
    except Exception as e:
        print(f"障碍物检测测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()
        cv2.destroyAllWindows()

def test_depth_based_avoidance():
    """测试基于深度的避障决策"""
    print("=== 测试基于深度的避障决策 ===")
    
    try:
        # 初始化系统
        vision_system = VisionSystem()
        
        # 模拟车辆移动函数
        def mock_car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
        
        # 初始化控制器
        controller = Controller(vision_system, mock_car_move, None)
        
        # 测试不同距离的障碍物
        test_cases = [
            {
                'name': '近距离关键障碍物',
                'obstacles': [
                    {'world_position': [0.1, 0.5, 0], 'distance': 0.4, 'is_critical': True, 'confidence': 0.9},
                ],
                'pingpang': [0, 1.0, 0]
            },
            {
                'name': '远距离非关键障碍物',
                'obstacles': [
                    {'world_position': [0.2, 0.8, 0], 'distance': 2.0, 'is_critical': False, 'confidence': 0.3},
                ],
                'pingpang': [0, 1.0, 0]
            },
            {
                'name': '中距离障碍物',
                'obstacles': [
                    {'world_position': [0.15, 0.6, 0], 'distance': 1.0, 'is_critical': False, 'confidence': 0.6},
                ],
                'pingpang': [0, 1.0, 0]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n测试案例 {i+1}: {test_case['name']}")
            
            # 测试路径检查
            has_obstacle = controller.check_obstacles_in_path(test_case['pingpang'], test_case['obstacles'])
            print(f"路径障碍物检查结果: {'需要避障' if has_obstacle else '无需避障'}")
            
            # 测试避障方向计算
            direction = controller.calculate_avoidance_direction(test_case['obstacles'], test_case['pingpang'])
            print(f"建议避障方向: {direction}")
        
        print("\n基于深度的避障决策测试完成")
        
    except Exception as e:
        print(f"基于深度的避障决策测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def test_360_search():
    """测试360度搜索功能"""
    print("=== 测试360度搜索功能 ===")
    
    try:
        # 初始化系统
        vision_system = VisionSystem()
        
        # 模拟车辆移动函数
        def mock_car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
        
        # 初始化控制器
        controller = Controller(vision_system, mock_car_move, None)
        
        print("开始360度搜索测试...")
        found = controller.search_360_degrees(max_search_time=10)  # 测试10秒
        
        if found:
            print("360度搜索测试成功：找到了乒乓球")
        else:
            print("360度搜索测试完成：未找到乒乓球")
            
    except Exception as e:
        print(f"360度搜索测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def test_avoidance_logic():
    """测试避障逻辑"""
    print("=== 测试避障逻辑 ===")
    
    try:
        # 初始化系统
        vision_system = VisionSystem()
        
        # 模拟车辆移动函数
        def mock_car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
        
        # 初始化控制器
        controller = Controller(vision_system, mock_car_move, None)
        
        # 测试避障方向计算
        test_obstacles = [
            {'world_position': [0.2, 0.5, 0], 'distance': 0.5},  # 右侧障碍物
            {'world_position': [-0.1, 0.3, 0], 'distance': 0.3},  # 左侧障碍物
        ]
        
        test_pingpang = [0, 1.0, 0]  # 前方的乒乓球
        
        direction = controller.calculate_avoidance_direction(test_obstacles, test_pingpang)
        print(f"避障方向计算结果: {direction}")
        
        # 测试路径障碍物检查
        has_obstacle = controller.check_obstacles_in_path(test_pingpang, test_obstacles)
        print(f"路径障碍物检查结果: {has_obstacle}")
        
        print("避障逻辑测试完成")
        
    except Exception as e:
        print(f"避障逻辑测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def main():
    """主测试函数"""
    print("乒乓球捡取系统 - 障碍物避障功能测试（深度信息验证版）")
    print("=" * 60)
    
    while True:
        print("\n请选择测试项目:")
        print("1. 测试障碍物检测功能（深度验证）")
        print("2. 测试基于深度的避障决策")
        print("3. 测试360度搜索功能")
        print("4. 测试传统避障逻辑")
        print("5. 运行所有测试")
        print("0. 退出")
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("退出测试")
            break
        elif choice == '1':
            test_obstacle_detection()
        elif choice == '2':
            test_depth_based_avoidance()
        elif choice == '3':
            test_360_search()
        elif choice == '4':
            test_avoidance_logic()
        elif choice == '5':
            print("运行所有测试...")
            test_obstacle_detection()
            time.sleep(2)
            test_depth_based_avoidance()
            time.sleep(2)
            test_360_search()
            time.sleep(2)
            test_avoidance_logic()
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
