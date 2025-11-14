import cv2
import numpy as np
import os

class LaneAssistTest:
    def __init__(self, image_path=None, imgsz=640):
        # 初始化参数
        self.image_path = image_path if image_path else self._get_default_image_path()
        self.imgsz = imgsz
        
        # 车道线基本参数
        self.lane_color = (0, 165, 255)  # 橙色车道线 (B, G, R)
        self.lane_thickness = 3          # 车道线粗细
        self.lane_alpha = 0.7            # 车道线透明度
        
        # 三角形车道线参数
        self.base_width = 300            # 底部宽度(px)
        self.apex_y = 100                # 顶点Y坐标
        self.robot_width = 28            # 机器人宽度(cm)，用于比例参考
        
        # 初始化图像
        self.image = self._load_and_preprocess_image()
        
        # 显示帮助信息
        self._print_help()
    
    def _get_default_image_path(self):
        """获取默认图像路径"""
        default_paths = [
            "c:/Users/hjj/Desktop/jobing/image.png",
            "./image.png",
            "../image.png"
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                return path
        return "./image.png"  # 默认返回路径
    
    def _load_and_preprocess_image(self):
        """加载并预处理图像"""
        # 尝试加载图像
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"警告: 无法加载图像 '{self.image_path}'，创建空白图像")
            # 创建一个带有网格背景的空白图像，便于测试
            image = np.ones((self.imgsz, self.imgsz, 3), dtype=np.uint8) * 240
            
            # 绘制网格线
            grid_size = 50
            for i in range(0, self.imgsz, grid_size):
                cv2.line(image, (i, 0), (i, self.imgsz), (200, 200, 200), 1)
                cv2.line(image, (0, i), (self.imgsz, i), (200, 200, 200), 1)
        
        # 调整图像尺寸
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        print(f"图像已加载，分辨率: {self.imgsz}x{self.imgsz}")
        return image
    
    def _print_help(self):
        """打印操作帮助信息"""
        print("\n=== 车道辅助线测试工具 ===")
        print("操作键盘按键控制车道线参数:")
        print("  w/s: 增加/减少底部宽度 (+/-20px)")
        print("  e/d: 上移/下移顶点 (+/-20px)")
        print("  r/f: 增加/减少车道线粗细 (+/-1px)")
        print("  t/g: 增加/减少透明度 (+/-0.1)")
        print("  c:   居中显示车道线")
        print("  q:   退出程序")
        print("=========================")
    
    def draw_lane_lines(self, frame):
        """绘制车道辅助线"""
        img_height, img_width = frame.shape[:2]
        center_x = img_width // 2  # 车道中心线
        bottom_y = img_height      # 底部坐标
        
        # 计算车道线端点
        left_start_x = center_x - self.base_width // 2
        right_start_x = center_x + self.base_width // 2
        
        # 创建透明覆盖层用于绘制车道线
        overlay = frame.copy()
        
        # 绘制三角形车道线
        # 左侧车道线
        cv2.line(overlay, (left_start_x, bottom_y), (center_x, self.apex_y), 
                 self.lane_color, self.lane_thickness)
        # 右侧车道线
        cv2.line(overlay, (right_start_x, bottom_y), (center_x, self.apex_y), 
                 self.lane_color, self.lane_thickness)
        # 可选：绘制中心线
        cv2.line(overlay, (center_x, bottom_y), (center_x, self.apex_y), 
                 (0, 255, 0), 1, cv2.LINE_AA)
        
        # 应用透明度
        cv2.addWeighted(overlay, self.lane_alpha, frame, 1 - self.lane_alpha, 0, frame)
        
        # 绘制机器人宽度指示
        self._draw_robot_width_indicator(frame, center_x, bottom_y)
        
        # 显示参数信息
        self._display_info(frame)
    
    def _draw_robot_width_indicator(self, frame, center_x, bottom_y):
        """绘制机器人宽度指示"""
        # 简化计算：假设底部宽度对应真实世界的某个距离
        # 这里仅作示意，实际应用需要根据相机标定计算
        robot_px_width = int(self.robot_width * 3)  # 简化比例转换
        robot_left = center_x - robot_px_width // 2
        robot_right = center_x + robot_px_width // 2
        
        # 绘制机器人宽度指示线
        indicator_y = bottom_y - 30
        cv2.line(frame, (robot_left, indicator_y), (robot_right, indicator_y), 
                 (255, 0, 0), 2)
        
        # 绘制机器人宽度文本
        robot_text = f"机器人宽度: {self.robot_width}cm"
        text_size = cv2.getTextSize(robot_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = center_x - text_size[0] // 2
        cv2.putText(frame, robot_text, (text_x, indicator_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    def _display_info(self, frame):
        """显示当前参数信息"""
        info = [
            f"底部宽度: {self.base_width}px",
            f"顶点Y坐标: {self.apex_y}px",
            f"车道线粗细: {self.lane_thickness}px",
            f"透明度: {self.lane_alpha:.1f}"
        ]
        
        # 绘制信息背景
        bg_height = len(info) * 25 + 10
        cv2.rectangle(frame, (5, 5), (200, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (200, bg_height), (255, 255, 255), 1)
        
        # 绘制信息文本
        for i, text in enumerate(info):
            cv2.putText(frame, text, (10, 25 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _update_parameters(self, key):
        """根据按键更新参数"""
        updated = False
        
        if key == ord('w'):  # 增加底部宽度
            self.base_width += 20
            updated = True
        elif key == ord('s'):  # 减少底部宽度
            self.base_width = max(100, self.base_width - 20)
            updated = True
        elif key == ord('e'):  # 上移顶点
            self.apex_y = max(50, self.apex_y - 20)
            updated = True
        elif key == ord('d'):  # 下移顶点
            self.apex_y = min(self.image.shape[0] - 100, self.apex_y + 20)
            updated = True
        elif key == ord('r'):  # 增加车道线粗细
            self.lane_thickness += 1
            updated = True
        elif key == ord('f'):  # 减少车道线粗细
            self.lane_thickness = max(1, self.lane_thickness - 1)
            updated = True
        elif key == ord('t'):  # 增加透明度
            self.lane_alpha = min(1.0, self.lane_alpha + 0.1)
            updated = True
        elif key == ord('g'):  # 减少透明度
            self.lane_alpha = max(0.1, self.lane_alpha - 0.1)
            updated = True
        elif key == ord('c'):  # 居中重置
            self.apex_y = 100
            self.base_width = 300
            updated = True
        
        return updated
    
    def run(self):
        """运行车道辅助线测试"""
        print("启动车道辅助线测试...")
        
        try:
            while True:
                # 创建图像副本以避免重叠绘制
                frame = self.image.copy()
                
                # 绘制车道辅助线
                self.draw_lane_lines(frame)
                
                # 显示图像
                cv2.imshow('车道辅助线测试', frame)
                
                # 键盘控制参数
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 退出
                    print("用户退出")
                    break
                
                # 更新参数
                if self._update_parameters(key):
                    print(f"更新参数: 底部宽度={self.base_width}, 顶点Y={self.apex_y}, ")
        
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.__del__()
    
    def __del__(self):
        """清理资源"""
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test = LaneAssistTest()
    test.run()
    print("测试结束")