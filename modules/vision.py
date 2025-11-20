"""
视觉处理模块，包含相机初始化、图像处理、目标检测等功能
"""
from config.config import Config
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import ImageFont, ImageDraw, Image # type: ignore
import pyrealsense2 as rs # type: ignore
import json
import math
from ultralytics import YOLO # type: ignore
import time
import logging
import base64
import requests

# 定义旋转角度（使用配置文件中的值）
theta = math.radians(Config.rotate_x)
cos_theta = math.cos(theta)
sin_theta = math.sin(theta)

class VisionSystem:

    def __init__(self):
        # 初始化RealSense相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # 加载乒乓球检测模型
        self.light_model = YOLO(Config.light_model)  # 轻量模型（用于近距离检测）
        self.heavy_model = YOLO(Config.heavy_model)  # 大体积模型（用于远距离检测）
        
        # 当前使用的模型
        self.model_type = Config.model_type if hasattr(Config, 'model_type') else "light"  # "light" or "heavy"
        self.current_model = self.light_model if self.model_type == "light" else self.heavy_model
        print(f"视觉系统初始化完成，当前模型: {self.model_type}, 轻量模型: {Config.light_model}, 大体积模型: {Config.heavy_model}")
        
        # 自动降级检测相关变量
        self.consecutive_success_detections = 0  # 连续成功检测次数
        self.last_detection_time = time.time()  # 上次检测时间


        # 初始化性能监控变量
        self.frame_times = []
        self.max_frame_times = 30  # 保存最近30帧的处理时间
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # 云端视觉增强相关变量
        self.use_cloud_vision = Config.use_cloud_vision
        self.cloud_vision_endpoint = Config.cloud_vision_endpoint
        self.cloud_vision_threshold = Config.cloud_vision_threshold
        self.cloud_vision_interval = Config.cloud_vision_interval
        self.cloud_vision_confidence = Config.cloud_vision_confidence
        self.cloud_vision_max_size = Config.cloud_vision_max_size
        self.last_cloud_vision_call = 0
        self.cloud_vision_results = None
        self.cloud_vision_results_time = 0
        self.cloud_vision_results_ttl = 3.0  # 云端结果有效期（秒）
        
        # 双模式切换相关变量
        self.detection_mode = "local"  # 初始模式为本地识别
        self.consecutive_no_detection = 0  # 连续未检测到乒乓球的次数
        self.max_consecutive_no_detection = 5  # 触发云端模式的连续未检测次数阈值
        self.move_time_multiplier = 2.0  # 云端模式下移动时间的倍数
        self.is_vehicle_moving = False  # 车辆是否在移动
        self.last_detection_time = time.time()  # 上次检测到乒乓球的时间

        print(f"视觉系统初始化完成，{'启用' if self.use_cloud_vision else '禁用'}云端视觉增强")
        if self.use_cloud_vision:
            print(f"云端视觉配置: 端点={self.cloud_vision_endpoint}, 间隔={self.cloud_vision_interval}秒")
            print(f"双模式切换: 连续未检测阈值={self.max_consecutive_no_detection}次")
            print(f"云端模式下移动时间倍数: {self.move_time_multiplier}倍")

    def update_detection_mode(self, results_boxes):
        """
        根据检测结果更新检测模式
        :param results_boxes: 检测框列表
        :return: 当前检测模式和移动时间倍数
        """
        current_time = time.time()

        # 如果检测到乒乓球
        if len(results_boxes) > 0:
            # 重置连续未检测计数
            self.consecutive_no_detection = 0
            self.last_detection_time = current_time

            # 如果当前是云端模式，切换回本地模式
            if self.detection_mode == "cloud":
                old_mode = self.detection_mode
                self.detection_mode = "local"
                print(f"检测到乒乓球，切换到本地识别模式")
        else:
            # 未检测到乒乓球，增加连续未检测计数
            self.consecutive_no_detection += 1

            # 如果连续未检测次数达到阈值，切换到云端模式
            if self.consecutive_no_detection >= self.max_consecutive_no_detection and self.detection_mode == "local":
                old_mode = self.detection_mode
                self.detection_mode = "cloud"
                print(f"连续 {self.consecutive_no_detection} 次未检测到乒乓球，切换到云端识别模式")

        # 返回当前模式和移动时间倍数
        time_multiplier = self.move_time_multiplier
        return self.detection_mode, time_multiplier
        
    def switch_model(self, model_type):
        """
        切换当前使用的模型
        :param model_type: 模型类型，"light" 或 "heavy"
        :return: 是否切换成功
        """
        if model_type not in ["light", "heavy"]:
            print(f"无效的模型类型: {model_type}，保持当前模型")
            return False
        
        if model_type == "light" and self.model_type != "light":
            self.current_model = self.light_model
            self.model_type = "light"
            print("切换到轻量模型检测")
            return True
        elif model_type == "heavy" and self.model_type != "heavy":
            self.current_model = self.heavy_model
            self.model_type = "heavy"
            print("切换到大体积模型检测")
            return True
        print(f"已经在使用{model_type}模型")
        return False
        
    def check_auto_downgrade(self, results_boxes):
        """
        检查是否需要自动降级到轻量模型
        :param results_boxes: 检测框列表
        :return: 是否需要降级
        """
        # 如果当前使用的是大体积模型
        if self.model_type == "heavy" and results_boxes:
            # 计算最大检测框面积
            max_area = 0
            for box in results_boxes:
                # 计算检测框的宽度和高度
                width = box.xyxy[0][2] - box.xyxy[0][0]
                height = box.xyxy[0][3] - box.xyxy[0][1]
                # 计算面积
                area = width * height
                # 更新最大面积
                if area > max_area:
                    max_area = area
            
            # 如果最大检测框面积大于400，自动降级
            if max_area > 550:
                print(f"检测到最大框面积{max_area:.2f}大于400，满足降级条件")
                self.switch_model("light")
                return True
        
        return False
        # """
        # 根据检测结果更新检测模式
        # :param results_boxes: 检测框列表
        # :return: 当前检测模式和移动时间倍数
        # """
        # current_time = time.time()

        # # 如果检测到乒乓球
        # if len(results_boxes) > 0:
        #     # 重置连续未检测计数
        #     self.consecutive_no_detection = 0
        #     self.last_detection_time = current_time

        #     # 如果当前是云端模式，切换回本地模式
        #     if self.detection_mode == "cloud":
        #         old_mode = self.detection_mode
        #         self.detection_mode = "local"
        #         print(f"检测到乒乓球，切换到本地识别模式")
        # else:
        #     # 未检测到乒乓球，增加连续未检测计数
        #     self.consecutive_no_detection += 1

        #     # 如果连续未检测次数达到阈值，切换到云端模式
        #     if self.consecutive_no_detection >= self.max_consecutive_no_detection and self.detection_mode == "local":
        #         old_mode = self.detection_mode
        #         self.detection_mode = "cloud"
        #         print(f"连续 {self.consecutive_no_detection} 次未检测到乒乓球，切换到云端识别模式")

        # # 返回当前模式和移动时间倍数
        # time_multiplier = self.move_time_multiplier if self.detection_mode == "cloud" else 1.0
        # return self.detection_mode, time_multiplier

    def set_vehicle_moving_state(self, is_moving):
        """
        设置车辆移动状态
        :param is_moving: 车辆是否在移动
        """
        self.is_vehicle_moving = is_moving

    def get_move_time_multiplier(self):
        """
        获取当前移动时间的倍数
        :return: 移动时间倍数
        """
        return self.move_time_multiplier if self.detection_mode == "cloud" else 2.0

    def check_need_cloud_vision(self, results_boxes):
        """
        检查是否需要使用云端视觉增强
        :param results_boxes: 本地检测到的乒乓球框
        :return: 是否需要使用云端视觉增强
        """
        # 如果未启用云端视觉增强，直接返回False
        if not self.use_cloud_vision:
            return False
            
        # 如果当前是云端模式且本地没有检测到乒乓球，需要云端视觉
        if self.detection_mode == "cloud" and len(results_boxes) == 0:
            print("云端模式下未检测到乒乓球，需要云端视觉增强")
            return True
            
        return False
    
    def call_cloud_vision(self, rgb, depth=None, aligned_depth_frame=None, depth_intrin=None):
        """
        调用云端视觉API进行远距离乒乓球检测 - 使用Ark API
        :param rgb: RGB图像
        :param depth: 深度图像（可选）
        :param aligned_depth_frame: 对齐的深度帧（可选）
        :param depth_intrin: 深度相机内参（可选）
        :return: (cloud_boxes, cloud_coordinates) 或 None（如果调用失败）
        """
        try:
            # 获取图像尺寸
            height, width = rgb.shape[:2]
            
            # 调整图像大小以减少传输数据量
            if width > self.cloud_vision_max_size or height > self.cloud_vision_max_size:
                scale = self.cloud_vision_max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_resized = cv2.resize(rgb, (new_width, new_height))
            else:
                rgb_resized = rgb
            
            # 将RGB图像编码为JPEG
            _, jpeg_data = cv2.imencode('.jpg', rgb_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rgb_base64 = base64.b64encode(jpeg_data).decode('utf-8')
            
            # 准备API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.cloud_api_key}"
            }
            
            # 构建请求体 - 使用Ark API格式
            payload = {
                "model": "doubao-seed-1-6-250615",  # 使用豆包视觉模型
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "图片中是否有乒乓球？如果有，请给出它们的位置。格式为：<bbox>x_min y_min x_max y_max</bbox>"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{rgb_base64}"
                            }
                        }
                    ]
                }],
                "thinking": {"type": "disabled"}  # 禁用思考过程，加快响应速度
            }
            
            # 发送请求到云端视觉API
            print("正在调用云端视觉增强API...")
            response = requests.post(
                self.cloud_vision_endpoint,
                headers=headers,
                json=payload,
                timeout=10.0  # 增加超时时间，确保有足够时间处理
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                
                # 解析边界框
                content = result['choices'][0]['message']['content']
                print(f"API返回内容: {content}")
                
                # 检查是否包含边界框标签
                if "<bbox>" in content and "</bbox>" in content:
                    # 提取边界框坐标
                    bbox_start = content.find("<bbox>") + len("<bbox>")
                    bbox_end = content.find("</bbox>")
                    coords_str = content[bbox_start:bbox_end].strip()
                    
                    # 解析坐标
                    coords = list(map(int, coords_str.split()))
                    
                    # 验证坐标数量
                    if len(coords) != 4:
                        print(f"坐标数量不正确，需要4个数值: {coords_str}")
                        return None
                    
                    x_min, y_min, x_max, y_max = coords
                    
                    # 计算实际像素坐标（假设API返回的是0-1000范围的归一化坐标）
                    x_min_real = int(x_min * width / 1000)
                    y_min_real = int(y_min * height / 1000)
                    x_max_real = int(x_max * width / 1000)
                    y_max_real = int(y_max * height / 1000)
                    
                    # 创建与本地检测结果兼容的格式
                    box_obj = type('CloudBox', (), {
                        'xyxy': np.array([x_min_real, y_min_real, x_max_real, y_max_real]).reshape(2, 2),
                        'conf': np.array([0.95]),  # 使用较高的置信度
                        '_is_cloud': True  # 标记为云端检测结果
                    })
                    cloud_boxes = [box_obj]
                    
                    # 计算中心点
                    center_x = (x_min_real + x_max_real) / 2
                    center_y = (y_min_real + y_max_real) / 2
                    
                    # 使用边界框大小估计距离
                    box_width = x_max_real - x_min_real
                    box_height = y_max_real - y_min_real
                    box_area = box_width * box_height
                    
                    # 边界框越大，估计距离越近
                    estimated_distance = 2.5  # 默认估计距离为2.5米（远距离）
                    
                    # 改进的距离估计逻辑
                    if box_area > 0:
                        # 使用非线性映射，更好地处理近距离情况
                        reference_area = 10000  # 参考面积
                        area_ratio = reference_area / box_area
                        
                        # 当边界框很大时（近距离），使用更保守的估计
                        if box_area > 15000:  # 非常近的距离
                            estimated_distance = 0.3 + (0.2 * area_ratio)  # 最小估计为0.3米
                        elif box_area > 8000:  # 较近的距离
                            estimated_distance = 0.5 + (0.3 * area_ratio)  # 最小估计为0.5米
                        else:
                            # 远距离使用原来的估计方法，但限制最大值
                            estimated_distance = min(3.0, max(1.0, area_ratio * 0.5))
                    
                    # 计算水平偏移（相对于图像中心）
                    image_center_x = width / 2
                    horizontal_offset = (center_x - image_center_x) / image_center_x  # 归一化到[-1, 1]
                    
                    # 创建估计的三维坐标
                    estimated_position = [
                        horizontal_offset * estimated_distance * 0.5,  # x坐标，根据水平偏移估计
                        estimated_distance,  # y坐标，估计距离
                        0.0  # z坐标，假设在同一高度
                    ]
                    
                    cloud_coordinates = [estimated_position]
                    
                    print(f"云端视觉增强成功，检测到乒乓球，估计距离: {estimated_distance:.2f}米")
                    
                    # 保存云端结果和时间戳
                    self.cloud_vision_results = (cloud_boxes, cloud_coordinates)
                    self.cloud_vision_results_time = time.time()
                    
                    return cloud_boxes, cloud_coordinates
                else:
                    print("未找到边界框标签")
                    return None
            else:
                print(f"云端视觉增强API调用失败: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"调用云端视觉增强API异常: {str(e)}")
            return None
        
    def draw_cloud_detections(self, image, cloud_boxes):
        """
        在图像上标记云端检测结果
        :param image: 图像
        :param cloud_boxes: 云端检测到的边界框
        :return: 标记后的图像
        """
        # 复制图像，避免修改原图
        marked_image = image.copy()
        
        # 使用不同颜色标记云端检测结果（蓝色）
        for box in cloud_boxes:
            x1, y1, x2, y2 = box.xyxy.reshape(4).astype(int)
            conf = float(box.conf)
            
            # 绘制边界框
            cv2.rectangle(marked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 绘制置信度
            label = f"Cloud: {conf:.2f}"
            cv2.putText(marked_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        return marked_image
    
    def merge_detection_results(self, local_boxes, cloud_boxes):
        """
        合并本地和云端的检测结果
        :param local_boxes: 本地检测到的边界框
        :param cloud_boxes: 云端检测到的边界框
        :return: 合并后的边界框列表
        """
        # 简单合并两个列表
        # 注意：这里可以实现更复杂的合并逻辑，如NMS（非极大值抑制）
        return local_boxes + cloud_boxes
        
    def merge_coordinate_lists(self, local_coords, cloud_coords):
        """
        合并本地和云端的坐标列表
        :param local_coords: 本地检测到的坐标列表
        :param cloud_coords: 云端检测到的坐标列表
        :return: 合并后的坐标列表
        """
        # 简单合并两个列表
        return local_coords + cloud_coords


    def get_aligned_images(self):
        """
        获取对齐的RGB图像和深度图像
        :return: 相机内参、深度参数、彩色图、深度图、对齐的深度帧
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 获取相机参数
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        camera_parameters = {
            'fx': intr.fx, 'fy': intr.fy,
            'ppx': intr.ppx, 'ppy': intr.ppy,
            'height': intr.height, 'width': intr.width,
            'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
        }

        # 保存内参到本地
        with open('./intrinsics.json', 'w') as fp:
            json.dump(camera_parameters, fp)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))
        color_image = np.asanyarray(color_frame.get_data())

        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

    def draw_boxes(self, image, boxes, real_position_list):

        """
        在图像上绘制检测框和信息
        """
        # 转换为PIL格式
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i in range(len(boxes)):
            box = boxes[i]
            box_position = box.xyxy.reshape(4)
            x = round(float(real_position_list[i][0]) * 100, 2)
            y = round(float(real_position_list[i][1]) * 100, 2)
            z = round(float(real_position_list[i][2]) * 100, 2)
            # 画框
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([(int(box_position[0]), int(box_position[1])), (int(box_position[2]), int(box_position[3]))],
                        outline=(255, 0, 0), width=2)
            # 加载默认字体
            font = ImageFont.load_default()
            # 写字,写在框上面
            draw.text((int(box_position[0]), int(box_position[1])),
                    f'置信度：{round(float(box.conf), 2)}，三维坐标：({x}cm,{y}cm,{z}cm)',
                    (255, 0, 0),
                    font=font)
        # 转换回cv2格式
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image

    def calculate_density(self, boxes, image, aligned_depth_frame, depth_intrin):
        """
        计算每个检测框中心点的三维坐标
        """
        density_list = []
        for i in range(len(boxes)):
            box = boxes[i]
            box_position = box.xyxy.reshape(4)
            x_center = int((box_position[0] + box_position[2]) / 2)
            y_center = int((box_position[1] + box_position[3]) / 2)
            # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
            dis = self.calculate_distance(x_center, y_center, aligned_depth_frame)  # 真实深度值
            if dis == 0:  # 如果深度值为0，标记为无效数据
                camera_coordinate = [0, 0, 0]
            else:
                camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_center, y_center], dis)
                camera_coordinate[1] = -camera_coordinate[1]  # y轴取反
                camera_coordinate[1], camera_coordinate[2] = camera_coordinate[2], camera_coordinate[1]
            density_list.append(camera_coordinate)
            cv2.circle(image, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=-1)
            print(camera_coordinate)
        return density_list

    def calculate_distance(self, x, y, depth_frame):
        # 取(x,y)点为中心的5*5的矩阵的深度值的平均值（排除深度值为0的像素点）同时避免所计算的像素点超出画面之外
        # dis = aligned_depth_frame.get_distance(x, y)
        dis = 0
        count = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x + i < 1280 and 0 <= y + j < 720:
                    temp_dis = depth_frame.get_distance(x + i, y + j)
                    if temp_dis != 0:
                        dis += temp_dis
                        count += 1
        if count != 0:
            dis = dis / count
        return dis

    def position_change(self, position):
        """
        将相机坐标系转换为机械臂坐标系
        :param position: 相机坐标系下的位置[x,y,z]
        :return: 机械臂坐标系下的位置[x,y,z]
        """
        # 拷贝输入，避免原地修改
        pos = list(position)
        # 应用绕X轴的旋转矩阵
        temp_y = pos[1]
        temp_z = pos[2]
        pos[0] = pos[0]  # X分量保持不变
        pos[1] = temp_y * cos_theta + temp_z * sin_theta  # Y分量
        pos[2] = temp_z * cos_theta - temp_y * sin_theta  # Z分量
        temp_y = position[1]
        temp_z = position[2]
        pos[0] = pos[0] + Config.x_offset
        pos[1] = pos[1] + Config.y_offset
        pos[2] = pos[2] + Config.z_offset
        position[0] = position[0]  # X分量保持不变
        position[1] = temp_y * cos_theta + temp_z * sin_theta  # Y分量
        pos[0] = int(pos[0] * 1000)
        pos[1] = int(pos[1] * 1000)
        pos[2] = int(pos[2] * 1000)
        position[2] = temp_z * cos_theta - temp_y * sin_theta  # Z分量
        return pos

    def check_safe_distance(self, pingpang_position, min_safe_distance=0.2, max_safe_distance=2.0):
        """
        检查乒乓球是否在安全距离范围内
        :param pingpang_position: 乒乓球的位置 [x, y, z]
        :param min_safe_distance: 最小安全距离(米)，小于此距离应停止逼近
        :param max_safe_distance: 最大安全距离(米)，大于此距离可能不准确
        :return: 是否在安全距离内，距离值，状态描述
        """
        if pingpang_position is None or (pingpang_position[0] == 0 and pingpang_position[1] == 0 and pingpang_position[2] == 0):
            return False, 0, "无效位置数据"

        # 计算距离（欧氏距离）
        distance = (pingpang_position[0]**2 + pingpang_position[1]**2 + pingpang_position[2]**2)**0.5

        # 检查距离是否在安全范围内
        if distance < min_safe_distance:
            return False, distance, "距离过近"
        elif distance > max_safe_distance:
            return False, distance, "距离过远"
        else:
            return True, distance, "距离正常"
        
    # 添加类变量用于跟踪上一次选择的目标和历史位置
    last_selected_target = None
    target_selection_count = 0
    target_history = []  # 存储最近几帧的目标位置
    max_history_length = 5  # 历史记录最大长度
    last_valid_depth = None  # 上一次有效的深度数据
    target_lost_count = 0  # 目标丢失计数
    max_target_lost = 3  # 允许目标丢失的最大次数
    last_timestamp = None  # 上一次处理的时间戳
    vehicle_motion = [0, 0, 0]  # 估计的车辆运动向量 [dx, dy, dz]
    depth_invalid_count = 0  # 深度数据无效计数
    max_depth_invalid = 5  # 允许深度数据无效的最大次数
    near_distance_threshold = 0.4  # 近距离阈值（米）
    near_distance_max_lost = 1  # 近距离时允许目标丢失的最大次数

    # 添加远程识别相关的类变量
    no_detection_count = 0  # 连续未检测到乒乓球的计数
    max_no_detection_before_cloud = 10  # 触发远程识别的连续未检测阈值
    cloud_vision_triggered = False  # 是否已触发远程识别
    cloud_vision_cooldown = 0  # 远程识别冷却时间计数


    
    def choose_pingpang_new(self, boxes, real_position_list, rgb=None, depth_image=None, aligned_depth_frame=None, depth_intrin=None):
        """
        选择最佳的乒乓球目标 - 增强版
        包含连续未检测到乒乓球的计数逻辑，并在需要时直接调用云端视觉增强
        :param boxes: 检测框列表
        :param real_position_list: 三维坐标列表
        :param rgb: RGB图像，用于云端视觉增强
        :param depth_image: 深度图像，用于云端视觉增强
        :param aligned_depth_frame: 对齐的深度帧，用于云端视觉增强
        :param depth_intrin: 深度相机内参，用于云端视觉增强
        :return: 选择的目标三维坐标
        """
        # 检查检测模式：如果是云端检测模式，使用云端检测结果
        if self.detection_mode == "cloud":
            print("云端检测模式：使用云端检测结果进行目标选择")
            # 检查是否有云端检测结果
            if hasattr(self, 'cloud_vision_results') and self.cloud_vision_results:
                cloud_boxes, cloud_coordinates = self.cloud_vision_results
                if cloud_coordinates:
                    # 选择第一个云端检测结果
                    return cloud_coordinates[0].copy()
            # 如果没有云端检测结果，返回None
            return None
        
        # 本地检测模式：使用本地检测结果
        # 如果没有检测到乒乓球，增加未检测计数
        if len(boxes) == 0:
            self.no_detection_count += 1
            # 未检测到乒乓球，返回None
            return None
        
        # 检测到乒乓球，重置计数器
        self.no_detection_count = 0
        
        # 检查是否有有效的深度数据
        valid_indices = []
        for i in range(len(real_position_list)):
            pos = real_position_list[i]
            if not (pos[0] == 0 and pos[1] == 0 and pos[2] == 0):
                valid_indices.append(i)
        
        # 如果没有有效深度数据，但有检测框，判断是否是近距离情况
        if len(valid_indices) == 0:
            # 获取图像中心点坐标
            image_center_x = 640 // 2
            image_center_y = 480 // 2

            # 找到最大的边界框和最接近中心的边界框
            largest_box_index = None
            max_box_area = 0
            closest_to_center = None
            min_center_distance = float('inf')
            
            for i, box in enumerate(boxes):
                # 获取边界框坐标
                box_position = box.xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)
            
                # 计算边界框面积
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # 更新最大边界框
                if box_area > max_box_area:
                    max_box_area = box_area
                    largest_box_index = i

                # 计算中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 计算到图像中心的距离
                center_distance = ((center_x - image_center_x)**2 + (center_y - image_center_y)**2)**0.5
            
                # 更新最接近中心的边界框
                if center_distance < min_center_distance:
                    min_center_distance = center_distance
                    closest_to_center = i
                
            # 判断是否是近距离情况：边界框面积大且位于画面中下部分
            if largest_box_index is not None:
                box_position = boxes[largest_box_index].xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                center_y = int((y1 + y2) / 2)

                # 如果边界框面积超过阈值且位于画面中下部分，认为是近距离情况
                if box_area > 15000 and center_y > image_center_y:
                    print(f"检测到近距离乒乓球（面积: {box_area}，位置: y={center_y}），但无深度数据，返回None")
                    return None

            # 如果不是近距离情况，使用最接近中心的边界框进行估计
            if closest_to_center is not None:
                # 使用边界框大小估计距离
                box_position = boxes[closest_to_center].xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)
                box_width = x2 - x1
                box_height = y2 - y1
                
                # 边界框越大，估计距离越近
                box_area = box_width * box_height
                estimated_distance = 1.0  # 默认估计距离为1米
                    
                # 根据实际测试数据改进的距离估计逻辑
                # 实际测试数据：3500对应0.3m，10000对应0.16m，15000对应0.14m
                if box_area > 0:
                    # 使用分段函数更准确地拟合实际测试数据
                    if box_area >= 15000:
                        # 非常近的距离：15000对应0.14m，面积更大时保持在0.14m附近
                        estimated_distance = 0.14 + (15000 - box_area) * 0.000002
                        estimated_distance = max(0.12, estimated_distance)  # 限制最小值
                    elif box_area >= 10000:
                        # 中等近距离：10000-15000之间，使用线性插值
                        # 10000对应0.16m，15000对应0.14m
                        t = (box_area - 10000) / 5000
                        estimated_distance = 0.16 + t * (0.14 - 0.16)
                    elif box_area >= 3500:
                        # 中距离：3500-10000之间，使用非线性映射
                        # 3500对应0.3m，10000对应0.16m
                        area_ratio = box_area / 3500
                        # 使用指数函数拟合曲线关系
                        estimated_distance = 0.3 * (0.16/0.3) ** (math.log(area_ratio, 10000/3500))
                    else:
                        # 远距离：使用反向比例关系，但更符合实际测试
                        # 3500对应0.3m，更小的面积对应更远的距离
                        base_distance = 0.3
                        base_area = 3500
                        # 使用平方根倒数关系，距离随面积减小而增加
                        estimated_distance = base_distance * math.sqrt(base_area / box_area)
                        # 限制最大值
                        estimated_distance = min(2.0, estimated_distance)
                    
                    print(f"距离估计 - 边界框面积: {box_area}, 估计距离: {estimated_distance:.3f}m")

                # 计算中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
            
                # 计算水平偏移（相对于图像中心）
                horizontal_offset = (center_x - image_center_x) / image_center_x  # 归一化到[-1, 1]
                
                # 创建估计的三维坐标
                estimated_position = [
                    horizontal_offset * estimated_distance * 0.5,  # x坐标，根据水平偏移估计
                    estimated_distance,  # y坐标，估计距离
                    0.0  # z坐标，假设在同一高度
                ]
                
                print(f"基于图像位置估计乒乓球位置: [{estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f}], 边界框面积: {box_area}")
                return estimated_position
            
            return None
                    
        # 如果有多个有效目标，选择最近的一个
        if len(valid_indices) > 1:
            # 找到最近的目标
            nearest_index = valid_indices[0]
            min_distance = float('inf')
                    
            for i in valid_indices:
                pos = real_position_list[i]
                distance = pos[0]**2 + pos[1]**2 + pos[2]**2  # 不需要开方，只比较大小
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_index = i
                    
            return real_position_list[nearest_index].copy()
                    
        # 只有一个有效目标，直接返回
        return real_position_list[valid_indices[0]].copy()
                    

    def is_cloud_vision_needed(self):
        """
        检查是否需要触发远程识别
        :return: 是否需要触发远程识别
        """
        # 如果已触发远程识别且未冷却，返回True
        if self.cloud_vision_triggered and self.cloud_vision_cooldown == 0:
            return True
        return False

    def vision_process(self):
        """
        优化的视觉处理函数，减少不必要的计算和内存使用
        """
        # 记录开始时间
        start_time = time.time()

        # 获取对齐的图像与相机内参
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 获取相机参数
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # 转换为numpy数组，避免重复转换
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        rgb = np.asanyarray(color_frame.get_data())

        # 检查检测模式：如果是云端检测模式，调用云端检测
        if self.detection_mode == "cloud":
            print("云端检测模式：调用云端检测")
            
            # 调用云端检测
            cloud_results = self.call_cloud_vision(rgb, depth_image, aligned_depth_frame, depth_intrin)
            
            if cloud_results:
                cloud_boxes, cloud_coordinates = cloud_results
                # 保存云端检测结果，供choose_pingpang_new方法使用
                self.cloud_vision_results = (cloud_boxes, cloud_coordinates)
                self.cloud_vision_results_time = time.time()
                
                # 在图像上标记云端检测结果
                rgb_display = self.draw_cloud_detections(rgb, cloud_boxes)
                
                # 显示云端检测模式状态和结果
                cv2.putText(rgb_display, "云端检测模式 - 检测到目标", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 云端检测成功后，调用大体积模型进行检测，检查是否可以降级
                print("云端检测成功，尝试大体积模型检测进行降级检查...")
                
                # 使用大体积模型进行检测
                heavy_results = self.heavy_model.predict(
                    rgb,
                    conf=Config.heavy_model_confidence_threshold,
                    imgsz=640,
                    device='0',
                    verbose=False
                )
                
                # 检查大体积模型是否检测到目标
                if heavy_results and len(heavy_results[0].boxes) > 0:
                    print("大体积模型检测成功，执行降级操作")
                    # 将检测模式切换为本地
                    self.detection_mode = "local"
                    # 切换到大体积模型
                    self.switch_model("heavy")
                    
                    # 更新显示信息
                    cv2.putText(rgb_display, "已降级到大体积模型", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                # 云端检测失败，返回空的检测结果
                results = None
                results_boxes = []
                camera_coordinate_list = []
                self.cloud_vision_results = None
                
                # 创建显示图像
                rgb_display = rgb.copy()
                
                # 显示云端检测模式状态
                cv2.putText(rgb_display, "云端检测模式 - 未检测到目标", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 云端检测模式下返回空的本地检测结果
            results = None
            results_boxes = []
            camera_coordinate_list = []
        else:
            # 本地检测模式：执行正常的模型推理
            # 根据模型类型选择置信度阈值
            confidence_threshold = Config.light_model_confidence_threshold if self.model_type == "light" else Config.heavy_model_confidence_threshold

            # 检测乒乓球（使用当前模型）
            results = self.current_model.predict(
                rgb,
                conf=confidence_threshold,
                imgsz=640,  # 保持与训练时相同的尺寸
                device='0',  # 使用GPU加速
                verbose=False
            )
            
            # 检查是否需要自动降级
            self.check_auto_downgrade(results[0].boxes)

            # 获取检测框
            results_boxes = results[0].boxes.cpu()

            # 计算每个乒乓球的三维坐标
            camera_coordinate_list = []
            for box in results_boxes:
                # 获取边界框坐标
                box_position = box.xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)

                # 计算中心点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 获取深度信息
                dis = self.calculate_distance(center_x, center_y, aligned_depth_frame)

                # 如果深度值有效，计算三维坐标
                if dis > 0:
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], dis)
                    camera_coordinate[1] = -camera_coordinate[1]  # y轴取反
                    camera_coordinate[1], camera_coordinate[2] = camera_coordinate[2], camera_coordinate[1]  # 交换y和z轴
                else:
                    camera_coordinate = [0, 0, 0]  # 无效深度值

                camera_coordinate_list.append(camera_coordinate)

                # 在原图上标记中心点，用于调试
                cv2.circle(rgb, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)

            # 简化绘制过程，只绘制基本信息
            rgb_display = rgb.copy()
            for i, box in enumerate(results_boxes):
                box_position = box.xyxy.reshape(4)
                x1, y1, x2, y2 = map(int, box_position)

                # 绘制矩形框
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 显示置信度
                confidence = float(box.conf)
                cv2.putText(rgb_display, f"{confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 如果有有效的三维坐标，显示距离
                if i < len(camera_coordinate_list) and camera_coordinate_list[i][2] > 0:
                    distance = camera_coordinate_list[i][2]
                    cv2.putText(rgb_display, f"{distance:.2f}m", (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 计算并显示FPS
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)

        # 保持固定长度的帧时间列表
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)

        # 每秒更新一次FPS
        if end_time - self.last_fps_update > 1.0:
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.last_fps_update = end_time

        # 显示FPS、帧时间和当前模型类型
        cv2.putText(rgb_display, f"FPS: {self.current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(rgb_display, f"Frame: {frame_time*1000:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(rgb_display, f"Model: {self.model_type}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        return intr, depth_intrin, rgb, depth_image, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display

    def wait_with_vision(self, wait_time, message=None):
        """
        等待指定时间，同时保持视觉处理和画面更新
        :param wait_time: 等待时间(秒)
        :param message: 显示在画面上的消息
        """
        if message:
            print(message)

        start_time = time.time()
        while time.time() - start_time < wait_time:
            # 进行视觉处理
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb = self.vision_process()

            # 如果有消息，显示在画面上
            if message:
                # 在画面底部添加文字
                cv2.putText(rgb, message, (50, rgb.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('RGB image', rgb)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                    break


    def close(self):
        """
        关闭视觉系统，释放资源
        """
        try:
            # 停止相机流
            self.pipeline.stop()
            print("相机流已停止")
        except Exception as e:
            print(f"关闭相机流时发生错误: {str(e)}")
