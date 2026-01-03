"""
硬件控制模块，包含Arduino控制、机械臂控制等功能
"""

import time
import random
import serial # type: ignore
from pyfirmata import Arduino, util # type: ignore
from config.config import Config


def init_arduino():
    """
    初始化Arduino控制
    :return: 车辆移动函数, Arduino板对象, 传感器读取函数
    """
    try:
        board = Arduino(Config.serial_port2)  # 使用配置文件中的串口
        it = util.Iterator(board)
        it.start()
        pin1 = board.get_pin("d:5:o")
        pin2 = board.get_pin("d:6:o")
        pin3 = board.get_pin("d:10:o")
        pin4 = board.get_pin("d:11:o")
        pena1 = board.get_pin("d:3:p")
        pena2 = board.get_pin("d:9:p")
        

        # 实际的车辆控制函数
        def car_move(direction, speed):
            print(f"车辆移动: {direction}, 速度: {speed}")
            if direction == "left":
                pin1.write(1)
                pin2.write(0)
                pin3.write(1)
                pin4.write(0)
                pena1.write(speed)
                pena2.write(speed)
            elif direction == "right":
                pin1.write(0)
                pin2.write(1)
                pin3.write(0)
                pin4.write(1)
                pena1.write(speed)
                pena2.write(speed)
            elif direction == "back":
                pin1.write(1)
                pin2.write(0)
                pin3.write(0)
                pin4.write(1)
                pena1.write(speed)
                pena2.write(speed)
            elif direction == "front":
                pin1.write(0)
                pin2.write(1)
                pin3.write(1)
                pin4.write(0)
                pena1.write(speed)
                pena2.write(speed)
            elif direction == "stop":
                pin1.write(0)
                pin2.write(0)
                pin3.write(0)
                pin4.write(0)
                pena1.write(speed)
                pena2.write(speed)

        return car_move, board

    except Exception as e:
        print(f"Arduino初始化失败: {e}")

        # 创建模拟的Arduino控制函数
        def car_move(direction, speed):
            print(f"模拟车辆移动: {direction}, 速度: {speed}")
        

        return car_move, None


def init_arm():
    """
    初始化机械臂
    :return: 机械臂串口对象
    """
    if Config.arm_switch:
        try:
            ser = serial.Serial(Config.serial_port1, 115200, timeout=1)
            time.sleep(2)
            ser.write("release;".encode())
            return ser
        except Exception as e:
            print(f"机械臂串口初始化失败: {e}")
            return None
    else:
        return None


def move(direction, speed, period, car_move_func=None):
    """
    控制车辆移动一段时间后停止
    :param direction: 移动方向
    :param speed: 移动速度
    :param period: 移动时间(秒)
    :param car_move_func: 车辆移动函数，如果为None则初始化一个新的
    """
    if car_move_func is None:
        car_move_func, _, _ = init_arduino()

    car_move_func("stop", 0)
    car_move_func(direction, speed)
    time.sleep(period)
    car_move_func("stop", 0)


def arm_control(ser, position, wait_with_vision_func):
    """
    控制机械臂移动和抓取
    :param ser: 机械臂串口
    :param position: 目标位置[x,y,z]
    :param wait_with_vision_func: 等待函数，同时保持视觉处理
    """
    if not ser:
        print("机械臂未初始化")
        return

    print(position)
    # 安全检查(防止撞击摄像头，同时判断位置是否在可抓取范围内)
    if not secure_check(position):
        return

    # 发送数据到串口并等待执行完成，同时保持视觉处理
    print(f"move;{position[0]};{position[1]};{position[2]};")
    ser.write(f"move;{position[0]};{position[1]};{position[2]};".encode())
    wait_with_vision_func(3, "移动机械臂到目标位置")

    ser.write("release;".encode())
    wait_with_vision_func(3, "松开机械爪")

    ser.write(f"move;{position[0]};{position[1]};{position[2]};".encode())
    wait_with_vision_func(3, "再次确认位置")

    ser.write("catch;".encode())
    wait_with_vision_func(3, "抓取乒乓球")

    ser.write(f"move;200;0;200;".encode())
    wait_with_vision_func(3, "移动到放置位置")

    ser.write("release;".encode())
    wait_with_vision_func(3, "释放乒乓球")

    ser.write(f"move;0;200;200;".encode())
    wait_with_vision_func(3, "回到初始位置")


def secure_check(position):
    """
    安全检查，确保机械臂不会碰撞
    :param position: 目标位置[x,y,z]
    :return: 是否安全
    """
    info_list = []
    if position[0] == 0 and position[1] == 0 and position[2] == 0:
        info_list.append("未识别到")
    if position[0] < -70:
        info_list.append("太左")
    if position[0] > 350:
        info_list.append("太右")
    if position[1] < 30:
        info_list.append("太近")
    if position[1] > 265:
        info_list.append("太远")
    if position[2] < -25:
        info_list.append("太低")
    if position[2] > 280:
        info_list.append("太高")

    if info_list:
        # 输出info_list中的信息，逗号分隔
        print(f"安全检查失败: {','.join(info_list)}")
        return False

    return True

class LidarMonitor:
    """后台运行的激光雷达监视器"""
    def __init__(self, port='COM3', cluster_distance=150, min_cluster_points=3, min_radius=30, max_radius=500, slam_system=None):
        """
        :param port: 激光雷达端口
        :param cluster_distance: 聚类距离阈值E（mm），距离小于此值的点归为一类
                                 - 关键参数，用于判断两点是否属于同一物体
                                 - 过小会导致完整障碍物被切割；过大会导致不同障碍物被误认为同一物体
                                 - 默认150mm适合室内环境常见障碍物（桌腿、椅子等）
        :param min_cluster_points: 最小聚类点数，少于此数的聚类视为噪声
        :param min_radius: 最小障碍物半径（mm）
        :param max_radius: 最大障碍物半径（mm）
        :param slam_system: SLAM 系统实例 (optional)
        """
        self.port = port
        self.lidar = None
        self.obstacles = []
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.slam_system = slam_system
        
        # 聚类参数
        self.cluster_distance = cluster_distance
        self.min_cluster_points = min_cluster_points
        self.min_radius = min_radius
        self.max_radius = max_radius

    def start(self):
        if self.running:
            return
        try:
            self.lidar = RPLidar(self.port)
            self.running = True
            self.thread = threading.Thread(target=self._scan_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"激光雷达监控已启动 (端口: {self.port})")
        except Exception as e:
            print(f"激光雷达启动失败: {e}")
            self.running = False

    def stop(self):
        self.running = False
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
            except:
                pass
            self.lidar = None
        print("激光雷达监控已停止")

    def _cluster_point_cloud(self, points):
        """
        对点云进行聚类，基于最近邻的聚类算法
        
        基于"同一物体在场景中是连续的"这一事实，使用最近邻聚类方法：
        - 一帧数据中同一物体的相邻点是连续的
        - 如果存在位置突变的情况，则这两点有很大概率属于两个不同的物体
        
        实现的关键是设定阈值E（self.cluster_distance），当两个障碍点之间的间隔小于E时，
        将这两个障碍点归为同一个物体；当两点间隔大于E时，将这两个障碍点归为两个不同的物体。
        
        阈值选择原则：
        - 阈值过小：会导致原本完整的障碍物被切割成很多小块
        - 阈值过大：会导致原本不同的障碍物被机器人视为同一个障碍物，造成障碍物漏检
        - 默认阈值：150mm（适合室内环境中的常见障碍物，如桌腿、椅子等）
        
        :param points: 点云列表 [(x, y), ...] (mm)
        :return: 聚类列表，每个聚类包含点集
        """
        if not points:
            return []
        
        clusters = []
        used = [False] * len(points)
        
        for i, point in enumerate(points):
            if used[i]:
                continue
            
            # 创建新聚类
            cluster = [point]
            used[i] = True
            
            # 查找附近的点
            changed = True
            while changed:
                changed = False
                for j, other_point in enumerate(points):
                    if used[j]:
                        continue
                    
                    # 计算距离
                    dx = point[0] - other_point[0]
                    dy = point[1] - other_point[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # 如果距离小于阈值，加入聚类
                    if distance < self.cluster_distance:
                        # 检查是否与聚类中任意点距离小于阈值
                        in_cluster = False
                        for cluster_point in cluster:
                            dx2 = cluster_point[0] - other_point[0]
                            dy2 = cluster_point[1] - other_point[1]
                            dist2 = math.sqrt(dx2*dx2 + dy2*dy2)
                            if dist2 < self.cluster_distance:
                                cluster.append(other_point)
                                used[j] = True
                                changed = True
                                in_cluster = True
                                break
                        if in_cluster:
                            break
            
            # 只保留足够大的聚类
            if len(cluster) >= self.min_cluster_points:
                clusters.append(cluster)
        
        return clusters
    
    def _estimate_obstacle_radius(self, cluster_points):
        """
        估算障碍物半径
        
        :param cluster_points: 聚类点集 [(x, y), ...]
        :return: 估算的半径（mm）
        """
        if len(cluster_points) < 2:
            return self.min_radius
        
        # 计算聚类中心
        center_x = sum(p[0] for p in cluster_points) / len(cluster_points)
        center_y = sum(p[1] for p in cluster_points) / len(cluster_points)
        
        # 计算所有点到中心的距离
        distances = []
        for x, y in cluster_points:
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx*dx + dy*dy)
            distances.append(dist)
        
        # 使用最大距离作为半径（保守估计）
        max_dist = max(distances)
        
        # 也可以使用平均距离 + 标准差（更准确但计算稍慢）
        # mean_dist = sum(distances) / len(distances)
        # std_dist = math.sqrt(sum((d - mean_dist)**2 for d in distances) / len(distances))
        # estimated_radius = mean_dist + std_dist
        
        # 限制在合理范围内
        estimated_radius = max(self.min_radius, min(max_dist, self.max_radius))
        
        return estimated_radius

    def _scan_loop(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break
                
                # 0. 如果有 SLAM 系统，更新 SLAM
                if self.slam_system:
                    self.slam_system.update(scan)

                # 1. 收集原始点云数据
                point_cloud = []
                for angle, distance, quality in scan:
                    if distance > 0 and quality > 0:
                        # 转换为笛卡尔坐标 (mm)
                        rad = np.radians(angle)
                        x = distance * np.cos(rad)
                        y = distance * np.sin(rad)
                        point_cloud.append((x, y))
                
                # 2. 对点云进行聚类
                clusters = self._cluster_point_cloud(point_cloud)
                
                # 3. 为每个聚类计算中心点和半径
                current_obstacles = []
                for cluster in clusters:
                    # 计算聚类中心
                    center_x = sum(p[0] for p in cluster) / len(cluster)
                    center_y = sum(p[1] for p in cluster) / len(cluster)
                    
                    # 估算半径
                    radius = self._estimate_obstacle_radius(cluster)
                    
                    current_obstacles.append({
                        'x': center_x,
                        'y': center_y,
                        'radius': radius
                    })
                
                # 4. 更新障碍物列表
                with self.lock:
                    self.obstacles = current_obstacles
                    
        except Exception as e:
            print(f"激光雷达扫描循环异常: {e}")
            self.running = False

    def get_obstacles(self):
        with self.lock:
            return list(self.obstacles)