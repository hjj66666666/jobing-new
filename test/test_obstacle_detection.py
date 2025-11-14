#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
障碍物检测算法测试脚本
可以在笔记本上直接运行，测试detect_obstacles_opencv函数的功能
"""

import time
import sys
import os

# 检查必要的依赖
print("正在检查必要的依赖...")

try:
    import cv2
    print("✓ OpenCV 已安装")
except ImportError:
    print("✗ OpenCV 未安装，请使用 'pip install opencv-python' 安装")
    # 继续执行，后续会有更详细的错误处理

try:
    import numpy as np
    print("✓ NumPy 已安装")
except ImportError:
    print("✗ NumPy 未安装，请使用 'pip install numpy' 安装")
    # 继续执行

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
    print("✓ pyrealsense2 已安装")
except ImportError:
    HAS_REALSENSE = False
    print("✗ pyrealsense2 未安装，RealSense相机模式不可用")
    print("  如需使用RealSense相机，请安装Intel RealSense SDK和对应的Python绑定")

print()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入配置
from config.config import Config

class ObstacleDetector:
    """简化版的障碍物检测器，只包含我们需要测试的功能"""
    
    def __init__(self):
        # 配置参数
        self.min_detection_distance = Config.min_detection_distance
        self.max_detection_distance = Config.max_detection_distance
        self.min_safe_distance = Config.min_safe_distance
        self.obstacle_size_threshold = Config.obstacle_size_threshold
        
    def detect_obstacles_opencv(self, rgb_image, depth_image, aligned_depth_frame, depth_intrin):
        """
        使用OpenCV进行障碍物检测（直接从vision.py复制的实现）
        :param rgb_image: RGB图像
        :param depth_image: 深度图像
        :param aligned_depth_frame: 对齐的深度帧
        :param depth_intrin: 深度相机内参
        :return: 障碍物列表，每个障碍物包含位置、距离等信息
        """
        obstacles = []
        
        try:
            # 获取图像尺寸
            height, width = rgb_image.shape[:2]
            
            # 设置深度阈值（检测合理距离内的障碍物）
            min_depth = self.min_detection_distance * 1000  # 最小深度（毫米）
            max_depth = self.max_detection_distance * 1000  # 最大深度（毫米）
            critical_depth = self.min_safe_distance * 1000  # 关键距离（毫米）
            
            print(f"深度检测范围: {min_depth/1000:.1f}m - {max_depth/1000:.1f}m, 关键距离: {critical_depth/1000:.1f}m")
            
            # 1. 深度障碍物检测（近距离关键障碍物）
            depth_mask = (depth_image > 0) & (depth_image < critical_depth)
            
            if np.any(depth_mask):
                depth_contours, _ = cv2.findContours(
                    depth_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in depth_contours:
                    area = cv2.contourArea(contour)
                    if area > self.obstacle_size_threshold:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # 获取该点的深度值
                            depth_value = depth_image[cy, cx]
                            if depth_value > 0:
                                # 计算世界坐标
                                world_pos = rs.rs2_deproject_pixel_to_point(
                                    depth_intrin, [cx, cy], depth_value
                                )
                                
                                obstacles.append({
                                    'type': 'depth_obstacle',
                                    'pixel_position': (cx, cy),
                                    'world_position': world_pos,
                                    'distance': depth_value / 1000.0,  # 转换为米
                                    'area': area,
                                    'confidence': 0.9,  # 深度检测置信度最高
                                    'is_critical': True  # 标记为关键障碍物
                                })
            
            # 2. 基于深度的颜色障碍物检测
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            
            # 定义障碍物的颜色范围
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            
            # 创建颜色掩码
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            color_mask = red_mask | blue_mask | green_mask
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            color_contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in color_contours:
                area = cv2.contourArea(contour)
                if area > self.obstacle_size_threshold:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 获取该点的深度值
                        depth_value = depth_image[cy, cx]
                        
                        # 只考虑在合理深度范围内的障碍物
                        if depth_value > 0 and min_depth < depth_value < max_depth:
                            # 计算世界坐标
                            world_pos = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [cx, cy], depth_value
                            )
                            
                            # 根据距离调整置信度
                            distance_m = depth_value / 1000.0
                            if distance_m < self.min_safe_distance:
                                confidence = 0.8  # 近距离高置信度
                                is_critical = True
                            elif distance_m < 1.5:
                                confidence = 0.6  # 中距离中等置信度
                                is_critical = False
                            else:
                                confidence = 0.3  # 远距离低置信度
                                is_critical = False
                            
                            obstacles.append({
                                'type': 'color_obstacle',
                                'pixel_position': (cx, cy),
                                'world_position': world_pos,
                                'distance': distance_m,  # 转换为米
                                'area': area,
                                'confidence': confidence,
                                'is_critical': is_critical
                            })
            
            # 3. 基于深度的边缘检测障碍物
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            edge_contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in edge_contours:
                area = cv2.contourArea(contour)
                if area > self.obstacle_size_threshold * 2:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 获取该点的深度值
                        depth_value = depth_image[cy, cx]
                        
                        # 只考虑在合理深度范围内的障碍物
                        if depth_value > 0 and min_depth < depth_value < max_depth:
                            # 计算世界坐标
                            world_pos = rs.rs2_deproject_pixel_to_point(
                                depth_intrin, [cx, cy], depth_value
                            )
                            
                            # 根据距离调整置信度
                            distance_m = depth_value / 1000.0
                            if distance_m < self.min_safe_distance:
                                confidence = 0.6  # 近距离中等置信度
                                is_critical = True
                            elif distance_m < 1.5:
                                confidence = 0.4  # 中距离低置信度
                                is_critical = False
                            else:
                                confidence = 0.2  # 远距离很低置信度
                                is_critical = False
                            
                            obstacles.append({
                                'type': 'edge_obstacle',
                                'pixel_position': (cx, cy),
                                'world_position': world_pos,
                                'distance': distance_m,  # 转换为米
                                'area': area,
                                'confidence': confidence,
                                'is_critical': is_critical
                            })
            
            # 4. 过滤和合并重复的障碍物
            obstacles = self._filter_duplicate_obstacles(obstacles)
            
            # 5. 按距离和重要性排序
            obstacles.sort(key=lambda x: (x['is_critical'], -x['distance']))  # 关键障碍物优先，距离近的优先
            
            # 6. 输出检测结果统计
            critical_count = sum(1 for obs in obstacles if obs['is_critical'])
            print(f"障碍物检测完成: 总计 {len(obstacles)} 个，关键障碍物 {critical_count} 个")
            
            return obstacles
            
        except Exception as e:
            print(f"障碍物检测过程中发生错误: {e}")
            return []
    
    def _filter_duplicate_obstacles(self, obstacles):
        """
        过滤重复的障碍物
        :param obstacles: 原始障碍物列表
        :return: 过滤后的障碍物列表
        """
        if len(obstacles) <= 1:
            return obstacles
        
        filtered_obstacles = []
        min_distance_threshold = 0.1  # 最小距离阈值（米）
        
        for obstacle in obstacles:
            is_duplicate = False
            obstacle_pos = obstacle['world_position']
            
            for existing_obstacle in filtered_obstacles:
                existing_pos = existing_obstacle['world_position']
                distance = np.sqrt(
                    (obstacle_pos[0] - existing_pos[0])**2 + 
                    (obstacle_pos[1] - existing_pos[1])**2 + 
                    (obstacle_pos[2] - existing_pos[2])**2
                )
                
                if distance < min_distance_threshold:
                    # 如果距离很近，保留置信度更高的
                    if obstacle['confidence'] > existing_obstacle['confidence']:
                        filtered_obstacles.remove(existing_obstacle)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_obstacles.append(obstacle)
        
        return filtered_obstacles
    
    def draw_obstacles(self, image, obstacles):
        """
        在图像上绘制检测到的障碍物
        :param image: 输入图像
        :param obstacles: 障碍物列表
        :return: 绘制障碍物后的图像
        """
        result_image = image.copy()
        
        for i, obstacle in enumerate(obstacles):
            pixel_pos = obstacle['pixel_position']
            distance = obstacle['distance']
            obstacle_type = obstacle['type']
            confidence = obstacle['confidence']
            is_critical = obstacle.get('is_critical', False)
            
            # 根据障碍物类型和重要性选择颜色和大小
            if obstacle_type == 'depth_obstacle':
                base_color = (0, 0, 255)  # 红色
            elif obstacle_type == 'color_obstacle':
                base_color = (0, 255, 255)  # 黄色
            else:  # edge_obstacle
                base_color = (255, 0, 255)  # 紫色
            
            # 如果是关键障碍物，使用更亮的颜色和更大的标记
            if is_critical:
                color = base_color
                thickness = 3
                radius = 15
                # 绘制关键障碍物的外圈
                cv2.circle(result_image, pixel_pos, radius + 5, (0, 255, 0), 2)
            else:
                color = tuple(int(c * 0.6) for c in base_color)  # 降低亮度
                thickness = 2
                radius = 10
            
            # 绘制圆形标记
            cv2.circle(result_image, pixel_pos, radius, color, -1)
            cv2.circle(result_image, pixel_pos, radius + 3, color, thickness)
            
            # 绘制障碍物信息
            info_text = f"{obstacle_type[:-8]}: {distance:.2f}m, {confidence:.1f}"
            cv2.putText(result_image, info_text, 
                       (pixel_pos[0] + 20, pixel_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image

def test_with_realsense_camera():
    """使用RealSense相机测试障碍物检测"""
    print("正在初始化RealSense相机...")
    
    # 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # 启动相机
        pipeline.start(config)
        
        # 创建对齐对象
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # 初始化障碍物检测器
        detector = ObstacleDetector()
        
        print("相机初始化完成，开始检测障碍物...")
        print("按 'q' 键退出测试")
        
        while True:
            # 获取一帧数据
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 获取深度相机内参
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            
            # 记录检测开始时间
            start_time = time.time()
            
            # 执行障碍物检测
            obstacles = detector.detect_obstacles_opencv(
                color_image, depth_image, aligned_depth_frame, depth_intrin
            )
            
            # 计算检测耗时
            detection_time = time.time() - start_time
            print(f"检测耗时: {detection_time:.3f}秒")
            
            # 在图像上绘制障碍物
            result_image = detector.draw_obstacles(color_image, obstacles)
            
            # 在图像上显示FPS
            fps_text = f"FPS: {1.0/detection_time:.1f}"
            cv2.putText(result_image, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Obstacle Detection Result', result_image)
            
            # 如果深度图像不为空，也显示深度图像
            if depth_image.size > 0:
                # 为了更好的可视化，将深度图像缩放到可见范围
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('Depth Image', depth_colormap)
            
            # 按q键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"相机测试过程中发生错误: {e}")
    
    finally:
        # 停止相机
        pipeline.stop()
        cv2.destroyAllWindows()
        print("测试结束")

def test_with_sample_images():
    """使用示例图像测试障碍物检测（如果没有RealSense相机）"""
    print("使用示例图像模式测试障碍物检测")
    print("注意：此模式需要提前准备RGB图像和深度图像")
    
    # 创建一个简单的模拟环境
    detector = ObstacleDetector()
    
    # 创建测试图像（模拟RGB图像）
    height, width = 480, 640
    rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景
    
    # 绘制一些彩色物体（模拟障碍物）
    cv2.rectangle(rgb_image, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色物体
    cv2.rectangle(rgb_image, (300, 150), (400, 250), (255, 0, 0), -1)  # 蓝色物体
    cv2.circle(rgb_image, (500, 200), 50, (0, 255, 0), -1)  # 绿色物体
    
    # 创建模拟深度图像（简单的距离渐变）
    depth_image = np.zeros((height, width), dtype=np.uint16)
    for i in range(height):
        for j in range(width):
            # 模拟深度数据，中心区域更近
            center_dist = np.sqrt((i - height/2)**2 + (j - width/2)** 2)
            depth_value = int(500 + center_dist * 2)  # 500mm到约2000mm
            depth_image[i, j] = depth_value
    
    # 为彩色物体设置更近的深度
    depth_image[100:200, 100:200] = 300  # 红色物体在300mm
    depth_image[300:250, 300:400] = 400  # 蓝色物体在400mm
    depth_image[150:250, 450:550] = 350  # 绿色物体在350mm
    
    # 创建模拟的深度内参
    class MockIntrinsics:
        def __init__(self):
            self.width = width
            self.height = height
            self.ppx = width / 2
            self.ppy = height / 2
            self.fx = width  # 假设焦距约等于图像宽度
            self.fy = width
            self.model = rs.distortion.none
            self.coeffs = [0, 0, 0, 0, 0]
    
    depth_intrin = MockIntrinsics()
    
    # 执行障碍物检测
    print("执行障碍物检测...")
    obstacles = detector.detect_obstacles_opencv(
        rgb_image, depth_image, None, depth_intrin
    )
    
    # 在图像上绘制障碍物
    result_image = detector.draw_obstacles(rgb_image, obstacles)
    
    # 显示结果
    cv2.imshow('模拟RGB图像', rgb_image)
    cv2.imshow('障碍物检测结果', result_image)
    
    print("按任意键退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("===== 障碍物检测算法测试 ======")
    print("1. 使用RealSense相机测试")
    print("2. 使用模拟图像测试（无相机时）")
    
    choice = input("请选择测试模式 (1/2): ")
    
    if choice == '1':
        try:
            test_with_realsense_camera()
        except ImportError:
            print("无法导入pyrealsense2库，请确保已安装Intel RealSense SDK")
            print("切换到模拟图像测试模式")
            test_with_sample_images()
    elif choice == '2':
        test_with_sample_images()
    else:
        print("无效的选择，默认使用模拟图像测试")
        test_with_sample_images()