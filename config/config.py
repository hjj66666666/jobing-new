class Config:
    # 运行模式 0：自动捡球 1：手动回车捡球 2：云端决策捡球
    mode = 0

    # 模型配置
    # 轻量模型（用于近距离检测）
    light_model = "./models/table_tennis_near.pt"  # 轻量乒乓球检测模型
    # 大体积模型（用于远距离检测）
    heavy_model = "./models/table_tennis_far.pt"  # 大体积乒乓球检测模型
    
    obstacle_model = "./yolo11n.pt"  # 障碍物检测模型（使用YOLO官方模型）
    light_model_confidence_threshold = 0.8  # 轻量模型置信度阈值
    heavy_model_confidence_threshold = 0.2  # 大体积模型置信度阈值
    obstacle_confidence_threshold = 0.4  # 障碍物检测置信度阈值
    
    # 检测架构参数
    detection_switch_distance = 1.5  # 检测切换距离（米）
    detection_switch_time = 2.0  # 检测切换时间（秒）

    # 硬件配置
    arm_switch = True  # 机械臂通信开关
    serial_port1 = "/dev/ttyCH341USB0"  # 机械臂串口端口号
    serial_port2 = "/dev/ttyACM0"  # 车辆串口端口号


    # 云端视觉增强配置
    use_cloud_vision = True  # 是否使用云端视觉增强
    cloud_vision_endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"  # Ark API端点
    cloud_vision_threshold = 2.0  # 使用云端视觉的距离阈值（米）
    cloud_vision_interval = 3.0  # 云端视觉请求间隔（秒），增加间隔减少API调用频率
    cloud_vision_confidence = 0.7  # 云端视觉检测置信度阈值
    cloud_vision_max_size = 640  # 发送到云端的图像最大尺寸（像素）
    cloud_api_key = "2612d766-851a-495c-921a-a259eabd8e2a"  # Ark API密钥

    # 避障参数
    min_safe_distance = 0.6  # 最小安全距离（米）
    safe_width = 200  # 安全通道宽度（像素）
    obstacle_size_threshold = 1000  # 障碍物最小像素数量
    max_avoidance_attempts = 3  # 最大避障尝试次数
    avoidance_turn_time = 1.0  # 避障转向时间（秒）
    avoidance_move_time = 0.8  # 避障前进时间（秒）
    
    # 360度搜索参数
    search_rotation_angle = 90  # 360度搜索每次旋转角度（改为90度）
    search_rotation_interval = 2.0  # 360度搜索旋转间隔（秒）
    search_direction_count = 6  # 360度搜索方向数量（四个方向）
    search_pause_time = 1.0  # 旋转后暂停时间（秒）
    
    # 深度检测参数
    min_detection_distance = 0.1  # 最小检测距离（米）
    max_detection_distance = 3.0  # 最大检测距离（米）
    threat_threshold = 0.5  # 威胁等级阈值
    depth_confidence_threshold = 0.3  # 深度检测置信度阈值
    
    # 2D避障参数（轻量级模式）
    use_2d_obstacle_detection = False  # 是否使用2D避障检测（算力有限时启用）
    path_width_threshold = 0.1        # 路径宽度阈值（图像宽度比例）
    motion_threshold = 30             # 运动检测阈值
    max_2d_obstacles = 10             # 最大2D障碍物数量
    obstacle_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "chair",
        "sofa",
        "bottle",
        "cup",
        "bowl",
        "laptop",
        "cell phone",
        "book",
    ]  # 要检测的障碍物类别

    # 机器人参数
    vehicle_width = 300  # 车辆宽度（毫米）
    vehicle_length = 400  # 车辆长度（毫米）
    vehicle_safety_margin = 100  # 车辆安全边距（毫米）
    speed = 0.6  # 车辆速度（0-1），从0.5增加到0.6
    turn_speed = 0.55  # 转向速度（0-1），从0.45增加到0.55
    search_speed = 0.5  # 360度搜索时的速度（0-1）
    search_pause_time = 1.2

    # 坐标系转换参数
    x_offset = 0  # X轴偏移量（米）
    y_offset = 0.08  # Y轴偏移量（米）
    z_offset = 0.04  # Z轴偏移量（米）
    rotate_x = 15  # 绕X轴旋转角度（度）

    # 乒乓球捡取参数
    target_distance = 222  # 目标距离（毫米）
    target_x = 0  # 目标X轴位置（毫米）
    max_attempts = 15  # 最大尝试次数
    
    # 自动降级检测参数
    auto_downgrade_threshold = 2  # 连续成功检测次数达到此值自动降级
    downgrade_confidence_threshold = 0.6  # 降级检测的置信度阈值

    # 系统参数
    frame_interval = 0.03  # 帧间隔（秒），约30FPS
    delay_time = 1.5  # 机械臂延迟时间（秒）
    stop_delay_time = 0.7  # 车辆停稳延迟时间（秒），从0.5秒增加到0.7秒

    # 调试参数
    debug_mode = False  # 调试模式
    save_images = False  # 是否保存图像
    log_level = "INFO"  # 日志级别

    # 安全参数
    ensure_stop_on_exit = True  # 确保程序退出时车辆停止
