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
        results = vision_system.model.predict(
            color_image,
            conf=Config.confidence_threshold,
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
        return color_image, results_boxes, camera_coordinate_list
    
    try:
        while True:
            # 计时开始
            frame_start_time = time.time()
            
            # 简化的视觉处理
            rgb, results_boxes, camera_coordinate_list = simplified_vision_process()
            target = vision_system.choose_pingpang(results_boxes, camera_coordinate_list)
            
            # 计算单帧处理时间
            frame_time = time.time() - frame_start_time
            
            # 在图像上可视化检测结果
            rgb_display = rgb.copy()
            
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