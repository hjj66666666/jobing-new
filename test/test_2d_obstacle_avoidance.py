#!/usr/bin/env python3
"""
2D避障系统测试脚本
专门用于测试基于传统CV的轻量级障碍物检测和避障功能
适用于算力有限且远距离深度数据不可靠的情况
"""

import cv2
import numpy as np
import time
from config.config import Config
from modules.vision import VisionSystem
from modules.controller import Controller

def test_2d_obstacle_detection():
    """测试2D障碍物检测功能"""
    print("=== 测试2D障碍物检测功能（轻量级版本） ===")
    
    try:
        # 初始化视觉系统
        vision_system = VisionSystem()
        print("视觉系统初始化成功")
        
        # 测试2D障碍物检测
        for i in range(15):
            print(f"第 {i+1} 次测试...")
            
            # 获取图像
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = vision_system.vision_process()
            
            # 使用2D障碍物检测
            obstacles_2d = vision_system.detect_obstacles_2d_only(rgb, results_boxes)
            
            # 统计不同类型和路径状态的障碍物
            color_obstacles = [obs for obs in obstacles_2d if obs['type'] == 'color_obstacle_2d']
            edge_obstacles = [obs for obs in obstacles_2d if obs['type'] == 'edge_obstacle_2d']
            motion_obstacles = [obs for obs in obstacles_2d if obs['type'] == 'motion_obstacle_2d']
            path_obstacles = [obs for obs in obstacles_2d if obs.get('is_in_path', False)]
            critical_obstacles = [obs for obs in obstacles_2d if obs.get('is_critical', False)]
            
            print(f"检测到 {len(obstacles_2d)} 个2D障碍物:")
            print(f"  - 颜色障碍物: {len(color_obstacles)} 个")
            print(f"  - 边缘障碍物: {len(edge_obstacles)} 个")
            print(f"  - 运动障碍物: {len(motion_obstacles)} 个")
            print(f"  - 路径障碍物: {len(path_obstacles)} 个")
            print(f"  - 关键障碍物: {len(critical_obstacles)} 个")
            
            for j, obstacle in enumerate(obstacles_2d):
                path_text = " [路径]" if obstacle.get('is_in_path', False) else ""
                critical_text = " [关键]" if obstacle.get('is_critical', False) else ""
                threat_text = f" 威胁:{obstacle.get('threat_level', 0):.2f}"
                distance_text = f" 距离:{obstacle.get('estimated_distance', 0):.1f}m"
                
                print(f"  障碍物 {j+1}: {obstacle['type']}, "
                      f"位置={obstacle['pixel_position']}, "
                      f"面积={obstacle.get('area', 0):.0f}{path_text}{critical_text}{threat_text}{distance_text}")
            
            # 绘制2D障碍物
            if obstacles_2d:
                rgb_display = vision_system.draw_2d_obstacles(rgb_display, obstacles_2d)
            
            # 显示图像和统计信息
            cv2.putText(rgb_display, f"2D Obstacles: {len(obstacles_2d)} | Path: {len(path_obstacles)}", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(rgb_display, f"Color: {len(color_obstacles)} | Edge: {len(edge_obstacles)} | Motion: {len(motion_obstacles)}", 
                       (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("2D障碍物检测测试（轻量级）", rgb_display)
            
            if cv2.waitKey(800) & 0xFF == 27:  # ESC键退出
                break
                
        print("2D障碍物检测测试完成")
        
    except Exception as e:
        print(f"2D障碍物检测测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()
        cv2.destroyAllWindows()

def test_2d_path_analysis():
    """测试2D路径分析功能"""
    print("=== 测试2D路径分析功能 ===")
    
    try:
        # 初始化系统
        vision_system = VisionSystem()
        
        # 模拟车辆移动函数
        def mock_car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
        
        # 初始化控制器
        controller = Controller(vision_system, mock_car_move, None)
        
        # 测试不同场景的2D路径分析
        test_scenarios = [
            {
                'name': '无障碍物场景',
                'pingpang_boxes': [type('Box', (), {'xyxy': np.array([[300, 200, 350, 250]])})],
                'obstacles_2d': []
            },
            {
                'name': '路径中有障碍物',
                'pingpang_boxes': [type('Box', (), {'xyxy': np.array([[300, 200, 350, 250]])})],
                'obstacles_2d': [
                    {
                        'pixel_position': (320, 300),
                        'is_in_path': True,
                        'threat_level': 0.8,
                        'type': 'color_obstacle_2d'
                    }
                ]
            },
            {
                'name': '路径外有障碍物',
                'pingpang_boxes': [type('Box', (), {'xyxy': np.array([[300, 200, 350, 250]])})],
                'obstacles_2d': [
                    {
                        'pixel_position': (100, 300),
                        'is_in_path': False,
                        'threat_level': 0.2,
                        'type': 'edge_obstacle_2d'
                    }
                ]
            },
            {
                'name': '低威胁障碍物',
                'pingpang_boxes': [type('Box', (), {'xyxy': np.array([[300, 200, 350, 250]])})],
                'obstacles_2d': [
                    {
                        'pixel_position': (310, 250),
                        'is_in_path': True,
                        'threat_level': 0.2,
                        'type': 'motion_obstacle_2d'
                    }
                ]
            }
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\n测试场景 {i+1}: {scenario['name']}")
            
            # 测试路径检查
            has_obstacle = controller.check_obstacles_in_path_2d(
                scenario['pingpang_boxes'], 
                scenario['obstacles_2d']
            )
            print(f"路径障碍物检查结果: {'需要避障' if has_obstacle else '无需避障'}")
            
            # 测试避障方向计算
            direction = controller.calculate_avoidance_direction_2d(scenario['obstacles_2d'])
            print(f"建议避障方向: {direction}")
        
        print("\n2D路径分析测试完成")
        
    except Exception as e:
        print(f"2D路径分析测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def test_2d_avoidance_integration():
    """测试2D避障集成功能"""
    print("=== 测试2D避障集成功能 ===")
    
    try:
        # 初始化系统
        vision_system = VisionSystem()
        
        # 模拟车辆移动函数
        def mock_car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
            time.sleep(0.1)  # 模拟移动时间
        
        # 初始化控制器
        controller = Controller(vision_system, mock_car_move, None)
        
        print("开始2D避障集成测试...")
        
        # 模拟几次检测和避障决策
        for i in range(5):
            print(f"\n第 {i+1} 轮检测...")
            
            # 获取图像
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = vision_system.vision_process()
            
            # 使用2D障碍物检测
            obstacles_2d = vision_system.detect_obstacles_2d_only(rgb, results_boxes)
            
            # 检查是否需要避障
            has_obstacle = controller.check_obstacles_in_path_2d(results_boxes, obstacles_2d)
            
            if has_obstacle:
                print("检测到需要避障的情况")
                direction = controller.calculate_avoidance_direction_2d(obstacles_2d)
                print(f"执行避障动作: {direction}")
                # 这里不实际执行移动，只是模拟
            else:
                print("无障碍物，继续前进")
            
            time.sleep(0.5)
        
        print("2D避障集成测试完成")
        
    except Exception as e:
        print(f"2D避障集成测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def test_performance_comparison():
    """测试性能对比（2D vs 深度检测）"""
    print("=== 测试性能对比（2D vs 深度检测） ===")
    
    try:
        # 初始化视觉系统
        vision_system = VisionSystem()
        
        # 测试2D检测性能
        print("测试2D障碍物检测性能...")
        start_time = time.time()
        
        for i in range(10):
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = vision_system.vision_process()
            obstacles_2d = vision_system.detect_obstacles_2d_only(rgb, results_boxes)
        
        end_time = time.time()
        avg_time_2d = (end_time - start_time) / 10
        
        # 测试深度检测性能
        print("测试深度障碍物检测性能...")
        start_time = time.time()
        
        for i in range(10):
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = vision_system.vision_process()
            obstacles_3d = vision_system.detect_obstacles_opencv(rgb, depth, aligned_depth_frame, depth_intrin)
        
        end_time = time.time()
        avg_time_3d = (end_time - start_time) / 10
        
        # 输出性能对比结果
        print(f"\n性能对比结果:")
        print(f"2D检测平均耗时: {avg_time_2d:.3f}秒")
        print(f"深度检测平均耗时: {avg_time_3d:.3f}秒")
        print(f"性能提升: {((avg_time_3d - avg_time_2d) / avg_time_3d * 100):.1f}%")
        
        if avg_time_2d < avg_time_3d:
            print("✅ 2D检测性能更优，适合算力有限的场景")
        else:
            print("⚠️ 深度检测性能更好，但2D检测更适合远距离场景")
        
    except Exception as e:
        print(f"性能对比测试失败: {e}")
    finally:
        if 'vision_system' in locals():
            vision_system.close()

def main():
    """主测试函数"""
    print("乒乓球捡取系统 - 2D避障功能测试（轻量级版本）")
    print("=" * 60)
    print("适用于算力有限且远距离深度数据不可靠的情况")
    print("=" * 60)
    
    while True:
        print("\n请选择测试项目:")
        print("1. 测试2D障碍物检测功能")
        print("2. 测试2D路径分析功能")
        print("3. 测试2D避障集成功能")
        print("4. 性能对比测试（2D vs 深度）")
        print("5. 运行所有2D测试")
        print("0. 退出")
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == '0':
            print("退出测试")
            break
        elif choice == '1':
            test_2d_obstacle_detection()
        elif choice == '2':
            test_2d_path_analysis()
        elif choice == '3':
            test_2d_avoidance_integration()
        elif choice == '4':
            test_performance_comparison()
        elif choice == '5':
            print("运行所有2D测试...")
            test_2d_obstacle_detection()
            time.sleep(2)
            test_2d_path_analysis()
            time.sleep(2)
            test_2d_avoidance_integration()
            time.sleep(2)
            test_performance_comparison()
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
