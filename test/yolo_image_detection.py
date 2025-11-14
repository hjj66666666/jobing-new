#!/usr/bin/env python3
"""
精简YOLO图片检测脚本
仅执行模型推理和绘制检测框
"""

import cv2
import os
from ultralytics import YOLO

def main():
    # 配置参数
    MODEL_PATH = '../pingpang_five.pt'  # 模型路径
    IMAGE_PATH = '../photo/WechatIMG42.jpg'  # 图片路径
    CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
    
    # 加载模型
    model = YOLO(MODEL_PATH)
    
    # 读取图片
    image = cv2.imread(IMAGE_PATH)
    
    # 使用GPU或CPU进行推理 (YOLO自动尝试使用可用设备)
    results = model.predict(
        image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )
    
    # 绘制检测框
    result_image = results[0].plot()
    
    # 保存结果
    output_path = '../photo/WechatIMG42_yolo_detection.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存至: {output_path}")
    
    # 显示结果
    cv2.imshow("YOLO检测结果", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()