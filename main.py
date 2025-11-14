"""
乒乓球捡取系统主程序
"""

import cv2 # type: ignore
import time
from config.config import Config
from modules.vision import VisionSystem
from modules.hardware import init_arduino, init_arm
from modules.controller import Controller


def main():
    print("乒乓球捡取系统启动")
    print(f"运行模式: {['自动捡球', '手动回车捡球', '云端决策捡球'][Config.mode]}")

    # 初始化硬件
    car_move_func, _, read_sensors_func = init_arduino()
    arm_serial = init_arm()

    # 初始化视觉系统
    vision_system = VisionSystem()

    # 初始化控制器
    controller = Controller(vision_system, car_move_func, arm_serial, read_sensors_func)

    try:
        while True:
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break

            # 按回车键开始自动逼近并抓取乒乓球
            if key == 13 and Config.mode == 1 and Config.arm_switch:
                # 使用高级乒乓球检测与逼近流程
                success = controller.advanced_pingpong_detection_and_approach(
                    target_distance=Config.target_distance,
                    target_x=Config.target_x,
                    max_attempts=Config.max_attempts,
                )
                # 如果所有检测方法（包括云端检测四个方向）都失败，终止程序
                if not success and Config.use_cloud_vision:
                    print("云端检测四个方向都没有找到乒乓球，系统停止运行")
                    break

            # 自动模式下，使用新的检测架构进行乒乓球捡取
            if Config.mode == 0 and Config.arm_switch:
                # 使用高级乒乓球检测与逼近流程
                success = controller.advanced_pingpong_detection_and_approach(
                    target_distance=Config.target_distance,
                    target_x=Config.target_x,
                    max_attempts=Config.max_attempts,
                )
                # 如果所有检测方法（包括云端检测四个方向）都失败，终止程序
                if not success and Config.use_cloud_vision:
                    print("云端检测四个方向都没有找到乒乓球，系统停止运行")
                    # 发送停止指令到小车
                    car_move_func("stop", 0)
                    break

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保车辆停止
        print("确保车辆停止...")
        car_move_func("stop", 0)
        time.sleep(0.5)  # 等待车辆完全停止

        # 确保相机流被停止
        vision_system.close()
        print("乒乓球捡取系统已关闭")


if __name__ == "__main__":
    main()
