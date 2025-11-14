"""
导航模块，包含路径规划、运动控制、避障等功能
"""

import time
import math
import cv2
import numpy as np
from config.config import Config


def smooth_move(car_move_func, direction, target_speed, last_direction=None, last_speed=0,
                transition_time=0.5, stop_before_direction_change=True):
    """
    平滑移动控制，处理方向变化和速度变化
    :param car_move_func: 车辆移动函数
    :param direction: 目标移动方向
    :param target_speed: 目标移动速度
    :param last_direction: 上一次的移动方向，如果为None表示之前是静止的
    :param last_speed: 上一次的移动速度
    :param transition_time: 过渡时间（秒）
    :param stop_before_direction_change: 方向改变前是否需要停止
    :return: 新的方向和速度
    """
    # 方向改变
    if last_direction != direction:
        # 方向改变，先停止
        if last_direction is not None and stop_before_direction_change:
            # 逐渐减速停止
            if last_speed > 0.3:
                car_move_func(last_direction, last_speed * 0.7)
                time.sleep(transition_time)
                car_move_func(last_direction, last_speed * 0.4)
                time.sleep(transition_time)
            car_move_func("stop", 0)
            time.sleep(transition_time)  # 短暂停顿

        # 使用新方向但逐渐加速
        start_speed = min(target_speed * 0.5, 0.25)  # 起始速度为目标速度的一半，最大0.25
        car_move_func(direction, start_speed)
        time.sleep(transition_time)

        mid_speed = min(target_speed * 0.7, 0.35)  # 中间速度为目标速度的70%，最大0.35
        car_move_func(direction, mid_speed)
        time.sleep(transition_time)

        # 然后使用目标速度
        car_move_func(direction, target_speed)
    # 速度变化较大
    elif abs(last_speed - target_speed) > 0.1:
        # 平滑过渡
        transition_speed = (last_speed + target_speed) / 2
        car_move_func(direction, transition_speed)
        time.sleep(transition_time)  # 短暂移动
        car_move_func(direction, target_speed)
    # 方向和速度变化不大
    else:
        # 直接使用新速度
        car_move_func(direction, target_speed)

    return direction, target_speed

def smooth_stop(car_move_func, direction, speed, transition_time=0.5):
    """
    平滑停止控制
    :param car_move_func: 车辆移动函数
    :param direction: 当前移动方向
    :param speed: 当前移动速度
    :param transition_time: 过渡时间（秒）
    """
    if speed > 0.3:
        # 第一级减速
        reduced_speed_1 = speed * 0.7
        car_move_func(direction, reduced_speed_1)
        print(f"车辆减速: {direction}, 速度: {reduced_speed_1:.2f}, 时间: {time.time():.3f}")
        time.sleep(transition_time)

        # 第二级减速
        reduced_speed_2 = speed * 0.4
        car_move_func(direction, reduced_speed_2)
        print(f"车辆减速: {direction}, 速度: {reduced_speed_2:.2f}, 时间: {time.time():.3f}")
        time.sleep(transition_time)

    # 停止
    car_move_func("stop", 0)
    print(f"车辆停止，时间: {time.time():.3f}")
    time.sleep(transition_time)

def calculate_speed(diff_value, base_speed=0.35, max_speed=0.5, reference_value=300,
                   is_near_distance=False, near_distance_max_speed=0.4):
    """
    计算移动速度，使用非线性映射
    :param diff_value: 差值（绝对值）
    :param base_speed: 基础速度
    :param max_speed: 最大速度
    :param reference_value: 参考值，用于归一化差值
    :param is_near_distance: 是否处于近距离模式
    :param near_distance_max_speed: 近距离模式下的最大速度
    :return: 计算后的速度
    """
    # 使用三次方根函数使速度变化更平滑
    speed_factor = min(1.0, math.pow(abs(diff_value) / reference_value, 1/3))
    adjusted_speed = base_speed + (max_speed - base_speed) * speed_factor

    # 在近距离模式下，降低速度以提高精度
    if is_near_distance:
        adjusted_speed = min(adjusted_speed, near_distance_max_speed)

    return adjusted_speed

def calculate_move_time(diff_value, reference_value=300, min_time=0.5, extra_time_threshold=100, extra_time=0.3):
    """
    计算移动时间，使用非线性映射
    :param diff_value: 差值（绝对值）
    :param reference_value: 参考值，用于归一化差值
    :param min_time: 最小移动时间（秒）
    :param extra_time_threshold: 额外时间阈值，超过此值会增加额外时间
    :param extra_time: 额外时间（秒）
    :return: 计算后的移动时间（秒）
    """
    # 计算基础移动时间，使用平方根函数使时间变化更平滑
    base_time = 0.2 + 0.2 * min(1.0, math.pow(abs(diff_value) / reference_value, 1/2))

    # 确保移动时间不小于最小值
    move_time = max(base_time, min_time)

    # 对于较大的偏差，增加额外时间
    if abs(diff_value) > extra_time_threshold:
        move_time += extra_time

    return move_time

def adjust_position_to_target(car_move_func, current_pos, target_pos, is_near_distance=False,
                             x_priority_factor=1.5, min_x_diff=30, min_distance_diff=20):
    """
    调整位置到目标位置
    :param car_move_func: 车辆移动函数
    :param current_pos: 当前位置 [x, y, z]
    :param target_pos: 目标位置 [x, y, z]
    :param is_near_distance: 是否处于近距离模式
    :param x_priority_factor: X轴优先级因子，X轴偏差大于距离偏差的多少倍时优先调整X轴
    :param min_x_diff: 最小X轴偏差，小于此值不调整
    :param min_distance_diff: 最小距离偏差，小于此值不调整
    :return: 是否已调整到位，当前方向，当前速度
    """
    # 计算位置差距
    x_diff = current_pos[0] - target_pos[0]
    distance_diff = current_pos[1] - target_pos[1]

    # 设置容差范围
    if is_near_distance:
        x_tolerance = 30  # 近距离时的X容差（毫米）
        distance_tolerance = 20  # 近距离时的距离容差（毫米）
    else:
        x_tolerance = 40  # 远距离时的X容差（毫米）
        distance_tolerance = 40  # 远距离时的距离容差（毫米）

    # 判断是否已经达到目标位置
    if abs(x_diff) < x_tolerance and abs(distance_diff) < distance_tolerance:
        return True, None, 0

    # 记录上一次的移动方向和速度
    last_direction = None
    last_speed = 0

    # 确定主要调整方向（优先调整左右位置）
    if abs(x_diff) > abs(distance_diff) * x_priority_factor and abs(x_diff) > min_x_diff:
        # 调整左右位置
        # 计算速度
        adjusted_speed = calculate_speed(x_diff, is_near_distance=is_near_distance)

        # 确定移动方向
        if x_diff > 0:  # 向右移动
            direction = "right"
        else:  # 向左移动
            direction = "left"

        # 平滑移动
        last_direction, last_speed = smooth_move(car_move_func, direction, adjusted_speed)

        # 计算移动时间
        move_time = calculate_move_time(x_diff)

        # 移动
        print(f"调整左右位置: {direction}, 速度: {adjusted_speed:.2f}, X偏差: {x_diff}mm")
        print(f"移动时间: {move_time:.2f}秒")
        time.sleep(move_time)

        # 平滑停止
        smooth_stop(car_move_func, direction, adjusted_speed)

        return False, None, 0
    elif abs(distance_diff) > min_distance_diff:
        # 调整前后距离
        # 计算速度
        adjusted_speed = calculate_speed(distance_diff, reference_value=400, is_near_distance=is_near_distance)

        # 确定移动方向
        if distance_diff > 0:  # 向前移动
            direction = "front"
        else:  # 向后移动
            direction = "back"

        # 平滑移动
        last_direction, last_speed = smooth_move(car_move_func, direction, adjusted_speed)

        # 计算移动时间
        move_time = calculate_move_time(distance_diff, reference_value=400)

        # 移动
        print(f"调整前后距离: {direction}, 速度: {adjusted_speed:.2f}, 距离偏差: {distance_diff}mm")
        print(f"移动时间: {move_time:.2f}秒")
        time.sleep(move_time)

        # 平滑停止
        smooth_stop(car_move_func, direction, adjusted_speed)

        return False, None, 0
    else:
        # 两个方向都已调整好或偏差很小，停止移动
        car_move_func("stop", 0)
        print("位置已调整，停止移动")
        return True, None, 0
