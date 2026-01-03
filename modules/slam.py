"""
SLAM 模块，基于 BreezySLAM 实现轻量级 2D 建图
"""
import time
import threading
import numpy as np
import cv2
from PIL import Image
from config.config import Config

# 尝试导入 BreezySLAM
try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
    BREEZYSLAM_AVAILABLE = True
except ImportError:
    BREEZYSLAM_AVAILABLE = False
    print("[Warning] BreezySLAM not found. SLAM functionality will be disabled.")

class SlamSystem:
    def __init__(self):
        self.map_lock = threading.Lock()
        self.current_map_img = None
        self.robot_pose = (0, 0, 0)  # x, y, theta (meters, meters, degrees)
        self.running = False
        
        # SLAM 参数
        # RPLidar A1 规格：最大检测距离约 12m（室内），理想条件下可达 15m
        # 地图大小应覆盖雷达检测范围：从中心到边缘至少 12m，所以总大小至少 24m x 24m
        # 使用 800 像素，每像素 0.03m，得到 24m x 24m 地图（更符合实际）
        self.MAP_SIZE_PIXELS = 800
        self.MAP_METERS_PER_PIXEL = 0.03  # 3cm per pixel -> 24m x 24m area (适合RPLidar A1的12m范围)
        self.SCAN_SIZE = 360  # 我们将 lidar 数据重新采样为 360 个点 (1度1个)
        self.DETECTION_ANGLE = 360
        self.DISTANCE_NO_DETECTION_MM = 12000 # 12m max range for RPLidar A1 (室内典型值)
        
        self.slam = None
        self.laser = None
        self.map_bytes = bytearray(self.MAP_SIZE_PIXELS * self.MAP_SIZE_PIXELS)
        
        # 存储 ROI 信息
        self.available_rois = []
        self.current_mask = None 
        
        # --- 新增：桌腿识别增强 ---
        self.table_rect_smooth = None # [x, y, w, h] 平滑后的球台边界
        self.valid_frames_count = 0   # 连续检测到有效球台的帧数
        
        if BREEZYSLAM_AVAILABLE:
            self._init_slam()
            
        self.last_update_time = time.time()

    def _init_slam(self):
        try:
            self.laser = Laser(
                scan_size=self.SCAN_SIZE,
                scan_rate_hz=10, 
                detection_angle_degrees=self.DETECTION_ANGLE,
                distance_no_detection_mm=self.DISTANCE_NO_DETECTION_MM
            )
            
            # RMHC_SLAM 参数: (laser, map_size_pixels, map_size_meters, ...)
            # map_size_meters 是地图的总大小（米），不是每像素的米数
            map_size_meters = self.MAP_SIZE_PIXELS * self.MAP_METERS_PER_PIXEL
            self.slam = RMHC_SLAM(
                self.laser, 
                self.MAP_SIZE_PIXELS, 
                map_size_meters
            )
            self.running = True
            print("[SLAM] Initialized successfully")
        except Exception as e:
            print(f"[SLAM] Initialization failed: {e}")
            self.running = False

    def update(self, raw_scan):
        """
        处理新的雷达扫描数据
        :param raw_scan: List of (angle, distance, quality) tuples from RPLidar.iter_scans()
                         格式：[(angle1, distance1, quality1), (angle2, distance2, quality2), ...]
                         - angle: 角度 (0-360度)
                         - distance: 距离 (mm)
                         - quality: 数据质量 (0-255)
        """
        if not self.running or not BREEZYSLAM_AVAILABLE:
            return

        # 1. 将原始扫描数据转换为固定长度的数组 (360 points)
        # 初始化为0 (无检测)
        scan_data = [0] * self.SCAN_SIZE
        
        # 填充数据
        # RPLidar.iter_scans() 返回格式: (angle, distance, quality)
        for angle, distance, quality in raw_scan:
            # 角度通常是 0-360，我们需要将其映射到 0-359 的索引
            # 注意 RPLidar 的角度方向和 BreezySLAM 的期望方向可能需要校准
            # 这里假设直接映射
            idx = int(angle % 360)
            if 0 <= idx < self.SCAN_SIZE:
                # 距离单位 mm，只有当距离有效且质量足够时才使用
                if distance > 0 and quality > 0:
                    scan_data[idx] = int(distance)
                else:
                    scan_data[idx] = 0 # No detection

        # 2. 更新 SLAM
        try:
            self.slam.update(scan_data)
            
            # 3. 获取位姿
            x, y, theta = self.slam.getpos()
            self.robot_pose = (x / 1000.0, y / 1000.0, theta) # Convert mm to meters
            # print(f"SLAM Pose: x={x:.0f}mm, y={y:.0f}mm, theta={theta:.1f}")

            # 4. 定期获取地图 (例如每 1-2 秒一次，避免消耗过多 CPU)
            if time.time() - self.last_update_time > 1.0:
                self._update_map()
                self.last_update_time = time.time()
                
        except Exception as e:
            print(f"[SLAM] Update failed: {e}")

    def _update_map(self):
        """
        从 BreezySLAM 获取地图并保存/缓存
        """
        self.slam.getmap(self.map_bytes)
        
        # 将 bytearray 转换为 numpy 图像
        map_arr = np.frombuffer(self.map_bytes, dtype=np.uint8).reshape(self.MAP_SIZE_PIXELS, self.MAP_SIZE_PIXELS)
        
        # --- 自动 ROI 与不可达区域处理 START ---
        
        # 1. 二值化：提取所有障碍物
        # 假设 map_arr 中 >127 为障碍物/未知
        _, obstacles = cv2.threshold(map_arr, 127, 255, cv2.THRESH_BINARY)
        
        # 2. 膨胀：给围挡和桌腿加粗 10cm (Buffer)
        # 3cm/pixel -> 10cm ≈ 3.3 pixels -> 使用 7x7 核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated_mask = cv2.dilate(obstacles, kernel)
        
        # 3. 识别桌腿并计算球台边界框
        contours, _ = cv2.findContours(obstacles, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        table_legs = []
        map_center = np.array([self.MAP_SIZE_PIXELS // 2, self.MAP_SIZE_PIXELS // 2])
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 筛选逻辑：排除围挡（太大）和噪点（太小），且在地图中心附近
            if 5 < area < 2000: 
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    dist = np.linalg.norm(np.array([cx, cy]) - map_center)
                    # 假设球台在地图中心 250 像素 (约7.5米) 范围内
                    if dist < 250: 
                        table_legs.append(cnt)
        
        table_mask = np.zeros_like(map_arr)
        
        # --- 增强算法 START ---
        current_valid_rect = None
        
        if len(table_legs) >= 3:
            all_points = np.vstack(table_legs)
            hull = cv2.convexHull(all_points)
            tx, ty, tw, th = cv2.boundingRect(hull)
            
            # 1. 尺寸约束检查 (Size Constraint)
            # 标准球台: 1.525m x 2.74m
            # 像素尺寸: w = 1.525 / 0.03 ≈ 51px, h = 2.74 / 0.03 ≈ 91px (假设竖放)
            # 或者横放: w ≈ 91px, h ≈ 51px
            # 允许误差 ±30%
            
            # 转换回米
            w_m = tw * self.MAP_METERS_PER_PIXEL
            h_m = th * self.MAP_METERS_PER_PIXEL
            
            # 检查是否大致符合球台比例 (不论横竖)
            # 最小边约 1.5m, 最大边约 2.7m
            min_side = min(w_m, h_m)
            max_side = max(w_m, h_m)
            
            # 宽松范围：最小边 1.0-2.0m, 最大边 2.0-3.5m
            if (1.0 < min_side < 2.0) and (2.0 < max_side < 3.5):
                current_valid_rect = [tx, ty, tw, th]
            else:
                pass
                # print(f"[SLAM] 忽略异常尺寸球台: {w_m:.1f}m x {h_m:.1f}m")
        
        # 2. 时间滤波 (Time Filtering / Smoothing)
        if current_valid_rect:
            if self.table_rect_smooth is None:
                self.table_rect_smooth = [float(x) for x in current_valid_rect]
            else:
                # 低通滤波: smooth = 0.9*old + 0.1*new
                alpha = 0.1
                for i in range(4):
                    self.table_rect_smooth[i] = self.table_rect_smooth[i] * (1-alpha) + current_valid_rect[i] * alpha
            
            self.valid_frames_count += 1
        else:
            # 如果当前帧没检测到有效球台，慢慢衰减置信度? 
            # 或者保持旧值不变（假设球台没跑）
            # 这里选择保持旧值，直到检测到新的有效值
            pass
            
        # 使用平滑后的结果
        if self.table_rect_smooth and self.valid_frames_count > 5: # 至少稳定检测5帧才启用
            tx, ty, tw, th = [int(x) for x in self.table_rect_smooth]
            
            # 更新 table_mask (用于生成不可达区域)
            # 注意：这里我们用矩形代替了凸包，因为平滑后的凸包不好算
            # 矩形足够覆盖球台下方
            cv2.rectangle(table_mask, (tx, ty), (tx+tw, ty+th), 255, -1)
        else:
            # 默认值
            tx, ty, tw, th = self.MAP_SIZE_PIXELS//2 - 25, self.MAP_SIZE_PIXELS//2 - 45, 50, 90
        
        # --- 增强算法 END ---
        
        # 4. 生成 4 个 ROI 区域 (基于平滑后的 tx, ty, tw, th)
        # 注意：Left和Right区域只包含球台高度范围，不包含与上下方的公共区域
        W, H = self.MAP_SIZE_PIXELS, self.MAP_SIZE_PIXELS
        
        self.available_rois = [
            {"id": 0, "name": "Top",    "rect": (0, 0, W, ty)},
            {"id": 1, "name": "Bottom", "rect": (0, ty+th, W, H-(ty+th))},
            {"id": 2, "name": "Left",   "rect": (0, ty, tx, th)},  # 只包含球台高度范围
            {"id": 3, "name": "Right",  "rect": (tx+tw, ty, W-(tx+tw), th)}  # 只包含球台高度范围
        ]
        
        # 5. 应用当前选择的 ROI
        base_unreachable = cv2.bitwise_or(dilated_mask, table_mask)
        self.current_mask = base_unreachable.copy()
        
        if hasattr(Config, 'current_roi_id') and Config.current_roi_id is not None:
            roi_idx = Config.current_roi_id
            if 0 <= roi_idx < len(self.available_rois):
                roi = self.available_rois[roi_idx]["rect"]
                rx, ry, rw, rh = roi
                if rw > 0 and rh > 0:
                    roi_block_mask = np.ones_like(map_arr) * 255
                    roi_block_mask[ry:ry+rh, rx:rx+rw] = 0
                    self.current_mask = cv2.bitwise_or(base_unreachable, roi_block_mask)
        
        # --- 自动 ROI 与不可达区域处理 END ---

        # 转换为可视化图像
        display_map = cv2.cvtColor(map_arr, cv2.COLOR_GRAY2BGR)
        
        # 可视化不可达区域 (红色)
        if self.current_mask is not None:
            display_map[self.current_mask == 255] = [0, 0, 255]
            
        # 绘制 ROI 分割线 (调试用，青色)
        cv2.rectangle(display_map, (tx, ty), (tx+tw, ty+th), (255, 0, 255), 2) # 球台框
        
        # 绘制机器人轨迹/位置
        # 转换机器人坐标 (mm) 到像素坐标
        # Center of map is MAP_SIZE_PIXELS / 2
        
        # BreezySLAM map origin is top-left? Or center? 
        # Usually SLAM starts at center.
        # Let's assume standard behavior or check documentation. 
        # Actually in RMHC_SLAM implementation, it handles map growth or fixed size.
        # If fixed size, initial position usually determines relative coordinates.
        
        # 简单保存
        with self.map_lock:
            self.current_map_img = display_map
            
        # 保存到文件供 Web 读取
        try:
            # 标记机器人位置
            rob_x_mm = self.robot_pose[0] * 1000
            rob_y_mm = self.robot_pose[1] * 1000
            
            # Map pixels conversion
            # SLAM 地图通常中心在地图中心
            center_offset = self.MAP_SIZE_PIXELS // 2
            
            # 如果是 BreezySLAM，初始位置通常就是地图中心
            px = int(rob_x_mm / (self.MAP_METERS_PER_PIXEL * 1000)) + center_offset
            py = int(rob_y_mm / (self.MAP_METERS_PER_PIXEL * 1000)) + center_offset
            
            # 画机器人
            cv2.circle(display_map, (px, py), 5, (0, 255, 0), -1)
            
            cv2.imwrite(Config.map_path, display_map)
            # print(f"Map updated and saved to {Config.map_path}")
            
        except Exception as e:
            print(f"Error saving map: {e}")

    def get_map(self):
        with self.map_lock:
            return self.current_map_img
    
    def get_pose(self):
        return self.robot_pose
    
    def get_mask(self):
        """获取当前的导航掩码"""
        return self.current_mask
