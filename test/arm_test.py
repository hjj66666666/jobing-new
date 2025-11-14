"""
机械臂测试程序
"""

import sys
import os
import time

# 添加项目根目录到Python路径，以便正确导入modules模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hardware import init_arm


def main():
    """主函数"""
    print("机械臂测试程序启动")

    # 初始化机械臂
    ser = init_arm()
    time.sleep(2)

    if not ser:
        print("机械臂初始化失败")
        return

    try:
        print("测试抓取")
        ser.write(f"move;0;200;100;".encode())
        time.sleep(3)
        ser.write("catch;".encode())
        time.sleep(3)
        ser.write(f"move;200;0;200;".encode())
        time.sleep(3)
        ser.write("release;".encode())
        time.sleep(3)
        ser.write(f"move;0;200;200;".encode())
        time.sleep(3)
        ser.write("catch;".encode())
        time.sleep(3)
        ser.write("release;".encode())
        time.sleep(4)

    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        ser.close()
        print("测试完成")


if __name__ == "__main__":
    main()
