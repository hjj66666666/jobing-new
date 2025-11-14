from ultralytics import YOLO
from config.config import Config
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyrealsense2 as rs
import json
import serial
import time
import socket
import math


pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 配置color流
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
# 旋转参数
m_rad = math.radians(Config.rotate_x)
n_rad = math.radians(Config.rotate_z)
cos_m = math.cos(m_rad)
sin_m = math.sin(m_rad)
cos_n = math.cos(n_rad)
sin_n = math.sin(n_rad)
# 车辆采摘行走计时器
car_timer = 0
# 时间戳
time_stamp = 0

# 创建串口对象
if Config.arm_switch:
    ser = serial.Serial(
        Config.serial_port, 115200, timeout=1
    )  # 根据实际情况修改串口号和波特率

if Config.mode in [0, 2]:
    car_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("10.42.0.132", 8000)
    car_socket.connect(server_address)
    print("connected to rasperypi!")


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = (
        aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    )  # 获取深度参数（像素坐标系转相机坐标系会用到）
    camera_parameters = {
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "height": intr.height,
        "width": intr.width,
        "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
    }
    # 保存内参到本地
    with open("./intrinsics.json", "w") as fp:
        json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit)
    )  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


def calculate_maturity_degree(image, masks):
    maturity_degree_results = []
    # 将img转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将img用mask进行掩膜
    for mask in masks:
        mask_img = gray[:, :] * mask.data[0, :, :].numpy()
        # 找到非零值的索引
        nonzero_indices = np.nonzero(mask_img)
        # 提取非零值
        nonzero_values = mask_img[nonzero_indices]
        # 计算非零值的平均值
        average_nonzero_value = np.mean(nonzero_values)
        # 将大于阈值的值置为maxValue,小于阈值的值置为minValue
        average_nonzero_value = (
            Config.maxValue
            if average_nonzero_value > Config.maxValue
            else average_nonzero_value
        )
        average_nonzero_value = (
            Config.minValue
            if average_nonzero_value < Config.minValue
            else average_nonzero_value
        )
        # 求百分比
        maturity_degree_results.append(
            (Config.maxValue - average_nonzero_value)
            / (Config.maxValue - Config.minValue)
        )
    return maturity_degree_results


def draw_boxes(image, boxes, maturity_degree_list, real_position_list):
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
        draw.rectangle(
            [
                (int(box_position[0]), int(box_position[1])),
                (int(box_position[2]), int(box_position[3])),
            ],
            outline=(255, 0, 0),
            width=2,
        )
        font = ImageFont.truetype("simhei.ttf", 10, encoding="utf-8")
        # 写字,写在框上面
        draw.text(
            (int(box_position[0]), int(box_position[1])),
            f"置信度：{round(float(box.conf), 2)}，成熟度：{round(maturity_degree_list[i], 2)}\n三维坐标：({x}cm,{y}cm,{z}cm)",
            (255, 0, 0),
            font=font,
        )
    # 转换回cv2格式
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image


def choose_grape(boxes, maturity_degree_list, real_position_list):
    # 获取boxes列表中box.conf最大的且对应到maturity_degree_list中满足成熟度大于Config.maturity_threshold的索引
    max_conf = 0
    max_index = 0
    for i in range(len(boxes)):
        if (
            boxes[i].conf > max_conf
            and maturity_degree_list[i] > Config.maturity_threshold
        ):
            max_conf = boxes[i].conf
            max_index = i
    if max_conf == 0:
        return None
    else:
        return real_position_list[max_index]


def choose_grape_in_running(boxes, maturity_degree_list, real_position_list):
    # 获取boxes列表中box.conf最大的且对应到maturity_degree_list中满足成熟度大于Config.maturity_threshold的索引
    max_conf = 0
    max_index = 0
    for i in range(len(boxes)):
        x = int(boxes[i].xyxy.reshape(4)[0])
        if (
            boxes[i].conf > max_conf
            and maturity_degree_list[i] > Config.maturity_threshold
            and secure_check(position_change(real_position_list[i].copy()))
            and x > 0
        ):
            max_conf = boxes[i].conf
            max_index = i
    if max_conf == 0:
        return None
    else:
        return real_position_list[max_index]


def calculate_density(boxes, image, aligned_depth_frame, depth_intrin, masks):
    # 将图像灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    density_list = []
    for i in range(len(boxes)):
        box = boxes[i]
        box_position = box.xyxy.reshape(4)
        x_center = int((box_position[0] + box_position[2]) / 2)
        y_center = int((box_position[1] + box_position[3]) / 2)
        # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        dis = calculate_distance(
            x_center, y_center, aligned_depth_frame
        )  # 根茎点的真实深度值(直接计算根茎点处距离容易计算到后面的背景，故用葡萄中心点距离代替)
        x, y = top_point(gray, masks[i])
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dis)
        camera_coordinate[1] = -camera_coordinate[1]  # y轴取反
        camera_coordinate[1], camera_coordinate[2] = (
            camera_coordinate[2],
            camera_coordinate[1],
        )
        density_list.append(camera_coordinate)
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        print(camera_coordinate)
    return density_list


def top_point(gray, mask):
    mask_img = gray[:, :] * mask.data[0, :, :].numpy()
    # 二值化,将非0位置的值置为255
    mask_img = np.where(mask_img > 0, 255, 0).astype(np.uint8)
    # 找到轮廓
    contours, hierarchy = cv2.findContours(
        mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    # 剔除点数小于30的轮廓
    contours = [
        contour for contour in contours if len(contour) > Config.min_point_outline
    ]
    # 合并点数大于30的轮廓
    contours_result = np.vstack(contours)
    # 拟合可旋转的面积最小的矩形
    rect = cv2.minAreaRect(contours_result)
    # 获取矩形的四个顶点
    points = cv2.boxPoints(rect)
    points = np.intp(points)
    # 将每个点按y轴从小到大排序
    points = points[np.lexsort(points.T)]
    # 取points前两个点的中心点和后两个点的中心点
    center1 = np.mean(points[:2], axis=0)
    center2 = np.mean(points[2:], axis=0)
    center1 = [int(center1[0]), int(center1[1])]
    center2 = [int(center2[0]), int(center2[1])]
    # 计算两个中心点的距离
    distance = math.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )
    # 向上延申的距离（像素值）
    up_distance = Config.stem_offset_pixel
    # 计算x,y方向延申的权重
    weight_x = (center1[0] - center2[0]) / distance
    weight_y = (center1[1] - center2[1]) / distance
    # 计算目标点
    target = [
        int(center1[0] + weight_x * up_distance),
        int(center1[1] + weight_y * up_distance),
    ]
    return target


def calculate_distance(x, y, depth_frame):
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


def position_change(position):
    # 从摄像头位置变更到机械臂位置
    # 旋转变化
    """
    x_new = cos_n * x + sin_n * cos_m * y + sin_n * sin_m * z
    y_new = -sin_n * x + cos_n * cos_m * y + cos_n * sin_m * z
    z_new = -sin_m * y + cos_m * z
    """
    temp_position = position.copy()
    position[0] = (
        cos_n * temp_position[0]
        + sin_n * cos_m * temp_position[1]
        + sin_n * sin_m * temp_position[2]
    )
    position[1] = (
        -sin_n * temp_position[0]
        + cos_n * cos_m * temp_position[1]
        + cos_n * sin_m * temp_position[2]
    )
    position[2] = -sin_m * temp_position[1] + cos_m * temp_position[2]
    # 平移变化
    position[0] = position[0] + Config.x_offset
    position[1] = position[1] + Config.y_offset
    position[2] = position[2] + Config.z_offset
    # 变更单位到mm
    position[0] = int(position[0] * 1000)
    position[1] = int(position[1] * 1000)
    position[2] = int(position[2] * 1000)
    return position


def arm_control(position):
    position = position_change(position)
    print(position)
    # 安全检查(防止撞击摄像头，同时判断位置是否在可抓取范围内)
    if not secure_check(position):
        return
    # 发送数据到串口
    # 将字符串编码为字节并发送
    print(f"move;{position[0]};{position[1]};{position[2]};")
    ser.write("release;".encode())
    vison_only(Config.delay_time)
    ser.write(f"move;0;200;280;".encode())
    vison_only(Config.delay_time)
    ser.write(f"move;{position[0]};{position[1]};{position[2]};".encode())
    vison_only(Config.delay_time)
    ser.write("catch;".encode())
    vison_only(Config.delay_time)
    ser.write(f"move;170;0;200;".encode())
    vison_only(Config.delay_time)
    ser.write("release;".encode())
    vison_only(Config.delay_time)


def secure_check(position):
    info_list = []
    if position[0] < -50:
        info_list.append("太左")
    if position[0] > 350:
        info_list.append("太右")
    if position[1] < 30:
        info_list.append("太近")
    if position[1] > 265:
        info_list.append("太远")
    if position[2] < 50:
        info_list.append("太低")
    if position[2] > 280:
        info_list.append("太高")
    if info_list:
        # 输出info_list中的信息，逗号分隔
        print(",".join(info_list))
        return False
    return True


def vision_process():
    # 获取对齐的图像与相机内参
    (
        intr_local,
        depth_intrin_local,
        rgb_local,
        depth_local,
        aligned_depth_frame_local,
    ) = get_aligned_images()
    # 检测葡萄串
    results_local = model.predict(
        rgb_local,
        conf=Config.confidence_threshold,
        retina_masks=True,
        verbose=False,
        classes=Config.pick_type,
    )
    # 计算成熟度
    maturity_degree_local = calculate_maturity_degree(
        rgb_local, results_local[0].masks.cpu() if results_local[0].masks else []
    )
    # 获取检测框
    results_boxes_local = results_local[0].boxes.cpu()
    # 获取segment
    masks_local = results_local[0].masks.cpu() if results_local[0].masks else []
    # 计算每个葡萄茎的三维坐标
    camera_coordinate_list_local = calculate_density(
        results_boxes_local,
        rgb_local,
        aligned_depth_frame_local,
        depth_intrin_local,
        masks_local,
    )
    # 画框
    rgb_local = draw_boxes(
        rgb_local,
        results_boxes_local,
        maturity_degree_local,
        camera_coordinate_list_local,
    )
    # 显示图像
    cv2.imshow("RGB image", rgb_local)  # 显示彩色图像
    return (
        intr_local,
        depth_intrin_local,
        rgb_local,
        depth_local,
        aligned_depth_frame_local,
        results_local,
        maturity_degree_local,
        results_boxes_local,
        camera_coordinate_list_local,
        rgb_local,
    )


def vison_only(delay):
    # 机械臂抓取和车辆停下的过程中保持视觉处理，保证画面流畅性
    start = time.time()
    while time.time() - start < delay:
        vision_process()
        cv2.waitKey(1)


def pick(results_boxes_pick, maturity_degree_pick, camera_coordinate_list_pick):
    global car_timer, time_stamp
    # choose_grape_in_running
    pos_pick = choose_grape_in_running(
        results_boxes_pick, maturity_degree_pick, camera_coordinate_list_pick
    )
    if pos_pick is not None and Config.arm_switch:
        # car stop
        car_socket.send("stop".encode("utf-8"))
        car_timer += time.time() - time_stamp
        print("重新计算葡萄位置")
        # 车辆稳定后再次获取坐标
        vison_only(Config.stop_delay_time)
        # 重新计算
        (
            intr_pick,
            depth_intrin_pick,
            rgb_pick,
            depth_pick,
            aligned_depth_frame_pick,
            results_pick,
            maturity_degree_pick,
            results_boxes_pick,
            camera_coordinate_list_pick,
            rgb_pick,
        ) = vision_process()
        # 重新选择葡萄
        pos_pick = choose_grape_in_running(
            results_boxes_pick, maturity_degree_pick, camera_coordinate_list_pick
        )
        if pos_pick is not None:
            arm_control(pos_pick)
        # car run
        car_socket.send(f"run:{Config.speed}".encode("utf-8"))
        time_stamp = time.time()


def find_road(model_road_detect):
    (
        intr_local,
        depth_intrin_local,
        rgb_local,
        depth_local,
        aligned_depth_frame_local,
    ) = get_aligned_images()
    has_road = False
    results_road = model_road_detect.predict(
        rgb_local, conf=Config.confidence_threshold_road, verbose=False
    )
    for road_pos in results_road[0].boxes.cpu():
        box_position = road_pos.xyxy.reshape(4)
        x_center = int((box_position[0] + box_position[2]) / 2)
        y_center = int((box_position[1] + box_position[3]) / 2)
        if 320 <= x_center <= 960 and 180 <= y_center <= 540:
            has_road = True
            break
    return has_road


def turn_left_90():
    # 左转
    car_socket.send(f"turn:{-Config.turn_speed}".encode("utf-8"))
    # 左转90°
    time.sleep(2)
    # 左转后变为直走
    car_socket.send(f"run:{Config.speed}".encode("utf-8"))


def turn_around():
    # 左转
    car_socket.send(f"turn:{-Config.turn_speed}".encode("utf-8"))
    # 左转180°
    time.sleep(4)
    # 左转后变为直走
    car_socket.send(f"run:{Config.speed}".encode("utf-8"))


if __name__ == "__main__":
    # 加载模型
    model = YOLO(Config.model)
    # 加载道路检测模型
    model_road = None
    if Config.mode == 2:
        model_road = YOLO(Config.model_road)
    # 车辆状态
    car_status = "finding"
    if Config.mode in [0, 2]:
        # car run
        car_socket.send(f"run:{Config.speed}".encode("utf-8"))
        time_stamp = time.time()
    while True:
        # 进行视觉处理
        (
            intr,
            depth_intrin,
            rgb,
            depth,
            aligned_depth_frame,
            results,
            maturity_degree,
            results_boxes,
            camera_coordinate_list,
            rgb,
        ) = vision_process()

        key = cv2.waitKey(1)
        # 按esc或q键关闭窗口
        if key & 0xFF == ord("q") or key == 27:
            pipeline.stop()
            break
        if Config.mode == 1:
            # 按回车键开始抓取葡萄
            if key == 13 and Config.arm_switch:
                if len(results_boxes) == 0:
                    print("未检测到葡萄")
                else:
                    # 默认取一个葡萄的坐标
                    pos = choose_grape(
                        results_boxes, maturity_degree, camera_coordinate_list
                    )
                    if pos is None:
                        print("未检测到成熟度大于阈值的葡萄")
                    else:
                        arm_control(pos)
        if Config.mode == 0:
            if len(results_boxes) == 0:
                print("未检测到葡萄")
            else:
                pick(results_boxes, maturity_degree, camera_coordinate_list)
        if Config.mode == 2:
            if car_status == "finding":
                road = find_road(model_road)
                if road:
                    # 左转开始采摘
                    turn_left_90()
                    car_status = "picking"
                    time_stamp = time.time()
            if car_status == "picking":
                car_timer += time.time() - time_stamp
                time_stamp = time.time()
                # 走完进去的路
                if car_timer > Config.road_time / 2:
                    turn_around()
                # 走完出来的路
                if car_timer > Config.road_time:
                    turn_left_90()
                    car_timer = 0
                    car_status = "finding"
                pick(results_boxes, maturity_degree, camera_coordinate_list)
    # 关闭串口连接
    if Config.arm_switch:
        ser.close()
    cv2.destroyAllWindows()
    if Config.mode in [0, 2]:
        # car stop
        car_socket.send("stop".encode("utf-8"))
        car_socket.close()
