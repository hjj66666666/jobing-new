#!/usr/bin/env python3
"""
测试视频中的乒乓球检测效果 - 使用Ark API
优化版：减少API调用频率，专注于摄像头输入
"""
import cv2
import time
import numpy as np
import base64
import requests
import os
import json
import threading
from queue import Queue, Empty

# 配置参数
DEFAULT_MODEL = "doubao-seed-1-6-250615"  # 替换为实际的模型ID
VIDEO_PATH = 0  # 0表示使用摄像头
PROMPT = "框出乒乓球的位置，输出 bounding box 的坐标"
BBOX_TAG_START = "<bbox>"
BBOX_TAG_END = "</bbox>"

# 读取API密钥
api_key = os.environ.get("ARK_API_KEY", "2612d766-851a-495c-921a-a259eabd8e2a")

# 图片压缩质量和尺寸
IMAGE_QUALITY = 70  # 降低质量以减少传输数据量
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 抽帧参数 - 优化后
FRAME_SKIP = 20  # 每隔多少帧处理一次，考虑到API平均耗时1.5秒，增加抽帧间隔
DETECTION_INTERVAL = 3.0  # 最小检测间隔（秒），确保API有足够时间响应
MAX_FPS = 30  # 最大帧率限制，避免处理过多无用帧

# 结果队列
result_queue = Queue(maxsize=1)

# 性能监控变量
api_call_times = []
MAX_TIMES_TO_TRACK = 5  # 跟踪最近5次API调用时间

def encode_frame_to_base64(frame):
    """将视频帧编码为Base64字符串，并进行预处理以减小大小"""
    # 调整图像大小以减少数据量
    if frame.shape[0] > FRAME_HEIGHT or frame.shape[1] > FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # 压缩图像
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_QUALITY])
    return base64.b64encode(buffer).decode('utf-8')

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
            "thinking": {"type": "disabled"}  # 禁用思考过程，加快响应速度
        }
        
        # 发送请求
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=30.0
        )
        
        # 解析结果
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API请求失败: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return None

def parse_bbox(response, frame_width, frame_height):
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
            
            # 计算实际像素坐标（假设API返回的是0-1000范围的归一化坐标）
            x_min_real = int(x_min * frame_width / 1000)
            y_min_real = int(y_min * frame_height / 1000)
            x_max_real = int(x_max * frame_width / 1000)
            y_max_real = int(y_max * frame_height / 1000)
            
            return (x_min_real, y_min_real, x_max_real, y_max_real)
        else:
            print("未找到边界框标签")
            return None
    except Exception as e:
        print(f"解析边界框失败: {str(e)}")
        return None

def detection_thread(frame_queue):
    """后台检测线程"""
    global api_call_times
    
    while True:
        try:
            # 从队列获取帧
            frame = frame_queue.get(timeout=1.0)
            if frame is None:  # 结束信号
                break
                
            # 编码为Base64
            image_base64 = encode_frame_to_base64(frame)
            
            # 调用API
            start_time = time.time()
            response = call_ark_api(image_base64)
            elapsed_time = time.time() - start_time
            
            # 记录API调用时间
            api_call_times.append(elapsed_time)
            if len(api_call_times) > MAX_TIMES_TO_TRACK:
                api_call_times.pop(0)
            
            # 计算平均API调用时间
            avg_time = sum(api_call_times) / len(api_call_times)
            print(f"API调用耗时: {elapsed_time:.2f}秒, 平均: {avg_time:.2f}秒")
            
            # 根据平均API调用时间动态调整抽帧参数
            global FRAME_SKIP, DETECTION_INTERVAL
            if avg_time > 2.0:
                # 如果API调用时间较长，增加抽帧间隔
                FRAME_SKIP = min(30, FRAME_SKIP + 2)
                DETECTION_INTERVAL = min(5.0, avg_time * 1.5)
            elif avg_time < 1.0:
                # 如果API调用时间较短，减少抽帧间隔
                FRAME_SKIP = max(10, FRAME_SKIP - 2)
                DETECTION_INTERVAL = max(2.0, avg_time * 2)
            
            if response:
                # 解析边界框
                height, width = frame.shape[:2]
                bbox = parse_bbox(response, width, height)
                
                # 将结果放入结果队列
                if bbox:
                    # 清空队列中的旧结果
                    while not result_queue.empty():
                        result_queue.get_nowait()
                    # 添加新结果
                    result_queue.put((bbox, time.time(), frame.copy()))
            
            frame_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"检测线程错误: {str(e)}")
            continue

def test_video_vision():
    """测试视频中的乒乓球检测"""
    print("开始测试视频中的乒乓球检测...")
    print(f"抽帧参数: 每{FRAME_SKIP}帧处理一次，最小间隔{DETECTION_INTERVAL}秒")
    
    # 打开摄像头
    print("尝试打开摄像头...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # 检查是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头连接")
        return
    
    print("成功打开摄像头")
    
    # 设置视频分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {actual_width}x{actual_height}")
    
    # 创建帧队列
    frame_queue = Queue(maxsize=1)
    
    # 启动检测线程
    detect_thread = threading.Thread(target=detection_thread, args=(frame_queue,))
    detect_thread.daemon = True
    detect_thread.start()
    
    # 记录上次检测时间和帧计数
    last_detection_time = 0
    frame_count = 0
    current_bbox = None
    bbox_timestamp = 0
    detected_frame = None
    last_frame_time = time.time()
    
    # 显示参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (0, 255, 0)
    
    try:
        while True:
            # 帧率控制
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < 1.0/MAX_FPS:
                # 如果帧率过高，等待一段时间
                time.sleep(1.0/MAX_FPS - elapsed)
            
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧，尝试重新打开...")
                # 尝试重新打开摄像头
                cap.release()
                cap = cv2.VideoCapture(VIDEO_PATH)
                if not cap.isOpened():
                    print("无法重新打开摄像头，退出程序")
                    break
                continue
            
            # 更新帧时间
            last_frame_time = time.time()
            
            # 帧计数器增加
            frame_count += 1
            
            # 显示原始帧
            display_frame = frame.copy()
            
            # 检查是否有新的检测结果
            try:
                if not result_queue.empty():
                    current_bbox, bbox_timestamp, detected_frame = result_queue.get_nowait()
                    print(f"获取到新的边界框: {current_bbox}")
            except Empty:
                pass
            
            # 如果有边界框且未过期，则绘制
            if current_bbox and (time.time() - bbox_timestamp) < 5.0:  # 5秒内的结果视为有效
                x_min, y_min, x_max, y_max = current_bbox
                
                # 绘制边界框
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # 显示坐标信息
                coord_text = f"({x_min},{y_min})-({x_max},{y_max})"
                cv2.putText(display_frame, coord_text, 
                           (x_min, y_min-10), font, font_scale, (0, 0, 255), font_thickness)
                
                # 计算中心点
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # 绘制中心点
                cv2.circle(display_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # 显示帧信息
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                       font, font_scale, text_color, font_thickness)
            
            # 显示抽帧信息
            cv2.putText(display_frame, f"Skip: {FRAME_SKIP} frames, Interval: {DETECTION_INTERVAL:.1f}s", (10, 60), 
                       font, font_scale, text_color, font_thickness)
            
            # 显示操作提示
            cv2.putText(display_frame, "Press 'q' to quit, SPACE to save frame", (10, 90), 
                       font, font_scale, text_color, font_thickness)
            
            # 显示帧
            cv2.imshow("Video Detection", display_frame)
            
            # 如果有检测到的帧，显示它
            if detected_frame is not None:
                cv2.imshow("Detected Frame", detected_frame)
            
            # 抽帧处理 - 每隔FRAME_SKIP帧处理一次，且满足最小时间间隔
            current_time = time.time()
            if (frame_count % FRAME_SKIP == 0) and (current_time - last_detection_time > DETECTION_INTERVAL):
                # 如果队列未满，添加新帧
                if frame_queue.empty():
                    print(f"处理第 {frame_count} 帧，抽帧参数: 每{FRAME_SKIP}帧，间隔{DETECTION_INTERVAL:.1f}秒")
                    frame_queue.put(frame.copy())
                    last_detection_time = current_time
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC 键退出
                break
            
            # 按空格键保存当前帧
            if key == 32:  # 空格键
                output_path = f"frame_capture_{int(time.time())}.jpg"
                cv2.imwrite(output_path, frame)
                print(f"已保存当前帧: {output_path}")
    
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 发送结束信号给检测线程
        frame_queue.put(None)
        
        # 等待检测线程结束
        detect_thread.join(timeout=1.0)
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("测试结束")

if __name__ == "__main__":
    test_video_vision()
