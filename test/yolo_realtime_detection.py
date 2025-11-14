#!/usr/bin/env python3
"""
简易YOLO实时检测脚本
仅运行模型推理并绘制所有检测框
添加推理时延显示
"""

import cv2
import time
import os
from ultralytics import YOLO

class YOLORealtimeDetector:
    def __init__(self, model_path, camera_id=0, confidence_threshold=0.05):
        """
        初始化YOLO实时检测器
        :param model_path: 模型路径
        :param camera_id: 摄像头ID
        :param confidence_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        self.running = False
        
    def initialize(self):
        """
        初始化模型和摄像头
        """
        print(f"加载模型: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False
        
        print(f"打开摄像头 (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"无法打开摄像头 (ID: {self.camera_id})")
            return False
        
        return True
    
    def process_frame(self, frame):
        """
        处理单帧图像
        :param frame: 输入帧
        :return: 处理后的帧
        """
        # 记录推理开始时间
        inference_start_time = time.time()
        
        device_used = "CPU"
        try:
            # 尝试使用GPU
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                imgsz=(960, 608),
                device=0,
                verbose=False
            )
            device_used = "GPU"
        except:
            # 切换到CPU
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                imgsz=(960, 608),
                device='cpu',
                verbose=False
            )
        
        # 计算推理时间（毫秒）
        inference_time_ms = (time.time() - inference_start_time) * 1000
        
        # 绘制所有检测框
        result_frame = results[0].plot()
        
        # 在图像左上角显示推理时延信息
        cv2.rectangle(result_frame, (0, 0), (200, 60), (0, 0, 0), -1)
        cv2.putText(result_frame, f"推理时间: {inference_time_ms:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, f"设备: {device_used}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_frame
    
    def run(self):
        """
        运行实时检测
        """
        if not self.initialize():
            print("初始化失败，退出程序")
            return
        
        self.running = True
        print("实时检测开始。按 'q' 键退出。")
        
        try:
            while self.running:
                # 读取一帧
                ret, frame = self.cap.read()
                
                if not ret:
                    print("无法读取摄像头帧")
                    break
                
                # 处理帧
                result_frame = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow('YOLO 实时检测', result_frame)
                
                # 检查按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"错误: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        清理资源
        """
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()


def main():
    # 模型路径
    MODEL_PATH = '../copy.pt'
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件 {MODEL_PATH}")
        return
    
    # 摄像头ID
    CAMERA_ID = 0
    
    # 置信度阈值
    CONFIDENCE_THRESHOLD = 0.5
    
    # 创建并运行检测器
    detector = YOLORealtimeDetector(
        model_path=MODEL_PATH,
        camera_id=CAMERA_ID,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    detector.run()


if __name__ == "__main__":
    main()