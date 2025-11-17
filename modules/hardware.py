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
            ser.write(f"move;0;200;200;".encode())
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
