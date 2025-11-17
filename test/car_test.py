"""
车辆测试程序
"""

import sys
import os
import time

# 添加项目根目录到Python路径，以便正确导入modules模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from modules.hardware import init_arduino, move


def main():
    """主函数"""
    print("车辆测试程序启动")

    # 初始化Arduino
    car_move_func, _ = init_arduino()

    try:
        # # 测试移动
        # print("测试前进")
        # move("front", 0.1, 4, car_move_func)

        # print("测试左转")
        # move("left", 0.3, 2.2, car_move_func)

        # print("测试前进")
        # move("front", 0.1, 4, car_move_func)

        # print("测试左转")
        # move("left", 0.3, 1.1, car_move_func)

        # print("测试前进")
        # move("front", 0.1, 2, car_move_func)

        # print("测试左转")
        # move("left", 0.3, 1.1, car_move_func)

        # print("测试前进")
        # move("front", 0.1, 4, car_move_func)

        # print("测试左转")
        # move("left", 0.3, 2.2, car_move_func)

        # print("测试前进")
        # move("front", 0.1, 4, car_move_func)

        print("测试左转")
        move("left", 0.5, 1.8, car_move_func)

        print("测试停止")
        car_move_func("stop", 0)

    except KeyboardInterrupt:
        print("程序被用户中断")
        car_move_func("stop", 0)
    except Exception as e:
        print(f"程序异常: {e}")
        car_move_func("stop", 0)
    finally:
        print("测试完成")


if __name__ == "__main__":
    main()
