"""
乒乓球检测模型性能测试脚本
用于测试仅检测乒乓球时的帧率
"""

import cv2
import time
import numpy as np
from config.config import Config
from modules.vision import VisionSystem
import pyrealsense2 as rs

def calculate_box_size(box):
    """
    计算检测框的大小（面积）
    :param box: 检测框对象
    :return: 检测框的面积
    """
    # 获取边界框坐标
    box_position = box.xyxy.reshape(4)
    x1, y1, x2, y2 = map(int, box_position)
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1
    
    # 计算面积
    area = width * height
    
    return width, height, area

def test_pingpang_detection():
    print("乒乓球检测性能测试开始")
    
    # 初始化视觉系统
    vision_system = VisionSystem()
    
    # 帧率计算变量
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # 简化的视觉处理函数
    def simplified_vision_process():
        # 获取对齐的图像与相机内参
        frames = vision_system.pipeline.wait_for_frames()
        aligned_frames = vision_system.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # 获取相机参数
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 转换为numpy数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 仅检测乒乓球，使用GPU进行推理（device='0'表示使用第一块GPU）
        results = vision_system.light_model.predict(
            color_image,
            conf=Config.light_model_confidence_threshold,
            imgsz=640,
            device='0',  # 这里指定使用GPU 0进行推理
            verbose=False
        )
        
        # 获取检测框
        results_boxes = results[0].boxes.cpu()
        
        # 计算每个乒乓球的三维坐标
        camera_coordinate_list = []
        for box in results_boxes:
            # 获取边界框坐标
            box_position = box.xyxy.reshape(4)
            x1, y1, x2, y2 = map(int, box_position)
            w, h = x2 - x1, y2 - y1
            
            # 计算中心点
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            distance = vision_system.calculate_distance(center_x, center_y, aligned_depth_frame)
            if 0.3 <= distance <= 3.0:
                camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], distance)
                camera_coordinate[1] = -camera_coordinate[1]
                camera_coordinate[1], camera_coordinate[2] = camera_coordinate[2], camera_coordinate[1]
            else:
                camera_coordinate = [0, 0, 0]
            camera_coordinate_list.append(camera_coordinate)
        return color_image, results_boxes, camera_coordinate_list, depth_image, aligned_depth_frame, depth_intrin
    
    try:
        while True:
            # 计时开始
            frame_start_time = time.time()
            
            # 简化的视觉处理
            rgb, results_boxes, camera_coordinate_list, depth_image, aligned_depth_frame, depth_intrin = simplified_vision_process()
            pos = vision_system.choose_pingpang_new(results_boxes, camera_coordinate_list, rgb, depth_image, aligned_depth_frame, depth_intrin)
            
            # 计算并打印每个检测框的大小和深度信息
            for i, box in enumerate(results_boxes):
                width, height, area = calculate_box_size(box)
                confidence = box.conf.item()
                
                # 获取对应的深度信息
                depth = "未知"
                if i < len(camera_coordinate_list) and camera_coordinate_list[i][2] > 0:
                    depth = f"{camera_coordinate_list[i][2]:.4f}m"
                
                print(f"检测框 #{i+1}: 宽={width}, 高={height}, 面积={area}, 置信度={confidence:.2f}, 深度={depth}")
            
            # 打印choose_pingpang_new返回的深度信息
            if pos is not None:
                print(f"选定乒乓球位置信息 - x: {pos[0]:.4f}, y: {pos[1]:.4f}, z: {pos[2]:.4f} (深度)")
            else:
                print("未检测到有效的乒乓球位置")

            # 计算单帧处理时间
            frame_time = time.time() - frame_start_time
            
            # 在图像上可视化检测结果
            rgb_display = rgb.copy()
            
            # 显示检测框和置信度
            for i, box in enumerate(results_boxes):
                # 获取边界框坐标
                box_position = box.xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)
                # 获取置信度
                conf = box.conf.item()
                
                # 绘制检测框
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 显示置信度
                cv2.putText(rgb_display, f"Conf: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 计算并显示帧率
            frame_count += 1
            if frame_count >= 10:  # 每10帧更新一次FPS
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                frame_count = 0
                start_time = current_time
            
            # 显示帧率和处理时间
            cv2.putText(rgb_display, f"FPS: {fps:.1f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(rgb_display, f"Frame time: {frame_time*1000:.1f}ms", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow("Pingpang Detection Test", rgb_display)
            
            # 按ESC键退出
            key = cv2.waitKey(1)
            if key == 27:
                break
    
    except KeyboardInterrupt:
        print("测试被用户中断")
    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭相机
        vision_system.close()
        print(f"测试结束，最终帧率: {fps:.1f} FPS")

if __name__ == "__main__":
    test_pingpang_detection()