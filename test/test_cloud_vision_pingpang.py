#!/usr/bin/env python3
"""
测试云端视觉增强功能 - 使用Ark API
"""
import cv2
import time
import numpy as np
import base64
import requests
import os
import json

# 配置参数
DEFAULT_MODEL = "doubao-seed-1-6-250615"  # 替换为实际的模型ID
# IMAGE_PATH = "../WechatIMG42.jpg"  # 图片路径
IMAGE_PATH = "../photo/pp4_new.jpg"  # 图片路径
PROMPT = "框出乒乓球的位置，输出 bounding box 的坐标"
BBOX_TAG_START = "<bbox>"
BBOX_TAG_END = "</bbox>"

# 读取API密钥
api_key = os.environ.get("ARK_API_KEY", "2612d766-851a-495c-921a-a259eabd8e2a")

# 图片压缩质量
IMAGE_QUALITY = 80

# 是否使用图片模式
USE_IMAGE = True

def test_cloud_vision():
    """测试云端视觉增强功能"""
    print("开始测试云端视觉增强功能...")
    
    # 读取图片并转换为Base64
    def get_image_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    # 调用Ark API
    def call_ark_api(image_base64):
        """调用Ark API进行图像分析"""
        try:
            # API端点
            api_endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
            
            # 准备API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # 构建请求体
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "图片中是否有乒乓球？如果有，请给出它们的位置。格式为：" + 
                                   BBOX_TAG_START + "x_min y_min x_max y_max" + BBOX_TAG_END
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }],
                "thinking": {"type": "disabled"}
            }
            
            print("正在调用Ark API...")
            
            # 发送请求
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            # 解析结果
            if response.status_code == 200:
                print("API调用成功")
                return response.json()
            else:
                print(f"API请求失败: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return None
    
    # 解析边界框
    def parse_bbox(response):
        """从API响应中解析边界框坐标"""
        try:
            # 获取响应内容
            content = response['choices'][0]['message']['content']
            print(f"API返回内容: {content}")
            
            # 检查是否包含边界框标签
            if BBOX_TAG_START in content and BBOX_TAG_END in content:
                # 提取边界框坐标
                bbox_start = content.find(BBOX_TAG_START) + len(BBOX_TAG_START)
                bbox_end = content.find(BBOX_TAG_END)
                coords_str = content[bbox_start:bbox_end].strip()
                
                # 解析坐标
                coords = list(map(int, coords_str.split()))
                
                # 验证坐标数量
                if len(coords) != 4:
                    print(f"坐标数量不正确，需要4个数值: {coords_str}")
                    return None
                
                x_min, y_min, x_max, y_max = coords
                return x_min, y_min, x_max, y_max
            else:
                print("未找到边界框标签")
                return None
        except Exception as e:
            print(f"解析边界框失败: {str(e)}")
            return None
    
    try:
        if USE_IMAGE and os.path.exists(IMAGE_PATH):
            print(f"使用图片: {IMAGE_PATH}")
            
            # 读取图片
            image = cv2.imread(IMAGE_PATH)
            if image is None:
                print(f"无法读取图片: {IMAGE_PATH}")
                return
            
            # 获取图片尺寸
            height, width = image.shape[:2]
            
            # 显示原始图像
            cv2.imshow("Original Image", image)
            
            # 获取图片的Base64编码
            image_base64 = get_image_base64(IMAGE_PATH)
            
            # 调用API
            start_time = time.time()
            response = call_ark_api(image_base64)
            elapsed_time = time.time() - start_time
            
            if response:
                print(f"API调用耗时: {elapsed_time:.2f}秒")
                
                # 解析边界框
                bbox = parse_bbox(response)
                
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    
                    # 计算实际像素坐标（假设API返回的是0-1000范围的归一化坐标）
                    x_min_real = int(x_min * width / 1000)
                    y_min_real = int(y_min * height / 1000)
                    x_max_real = int(x_max * width / 1000)
                    y_max_real = int(y_max * height / 1000)
                    
                    # 在图像上绘制边界框
                    image_with_bbox = image.copy()
                    cv2.rectangle(image_with_bbox, 
                                 (x_min_real, y_min_real), 
                                 (x_max_real, y_max_real), 
                                 (0, 0, 255), 2)
                    
                    # 显示带边界框的图像
                    cv2.imshow("Image with Bounding Box", image_with_bbox)
                    
                    # 保存结果图像
                    output_path = os.path.splitext(IMAGE_PATH)[0] + "_with_bbox.jpg"
                    cv2.imwrite(output_path, image_with_bbox)
                    print(f"已保存结果图像: {output_path}")
                    
                    # 打印坐标信息
                    print(f"检测到乒乓球，边界框坐标: ({x_min}, {y_min}, {x_max}, {y_max})")
                    print(f"实际像素坐标: ({x_min_real}, {y_min_real}, {x_max_real}, {y_max_real})")
            
            # 等待按键退出
            print("按 'q' 键退出...")
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:  # q 或 ESC 键退出
                    break
        else:
            print("未找到指定图片或未启用图片模式")
    
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()
        print("测试结束")

if __name__ == "__main__":
    test_cloud_vision()