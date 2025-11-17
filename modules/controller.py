"""
控制器模块，包含自动逼近、抓取等高级功能
"""
import time
import cv2 # type: ignore
from config.config import Config
from modules.navigation import (
    smooth_move, smooth_stop, calculate_speed, calculate_move_time, adjust_position_to_target
)
import math

class Controller:
    def __init__(self, vision_system, car_move_func, arm_serial):
        """
        初始化控制器
        :param vision_system: 视觉系统对象
        :param car_move_func: 车辆移动函数
        :param arm_serial: 机械臂串口对象
        :param read_sensors_func: 传感器读取函数
        """
        self.vision_system = vision_system
        self.car_move_func = car_move_func
        self.arm_serial = arm_serial
        self._stop_requested = False

    def arm_control(self, position):
        """
        控制机械臂移动和抓取
        :param position: 目标位置[x,y,z]
        """
        if not self.arm_serial:
            print("机械臂未初始化")
            return

        position = self.vision_system.position_change(position.copy())
        print(position)

        # 安全检查(防止撞击摄像头，同时判断位置是否在可抓取范围内)
        if not self.secure_check(position):
            return

        # 发送数据到串口并等待执行完成，同时保持视觉处理
        print(f'move;{position[0]};{position[1]};{position[2]};')
        self.arm_serial.write(f'move;{position[0]};{position[1]};{position[2]};'.encode())
        self.vision_system.wait_with_vision(2, "移动机械臂到目标位置")

        # self.arm_serial.write('release;'.encode())
        # self.vision_system.wait_with_vision(3, "松开机械爪")

        self.arm_serial.write('catch;'.encode())
        self.vision_system.wait_with_vision(2, "抓取乒乓球")

        self.arm_serial.write(f'move;200;0;200;'.encode())
        self.vision_system.wait_with_vision(2, "移动到放置位置")

        self.arm_serial.write('release;'.encode())
        self.vision_system.wait_with_vision(2.5, "释放乒乓球")

        self.arm_serial.write(f'move;0;200;200;'.encode())
        self.vision_system.wait_with_vision(2, "回到初始位置")

    def secure_check(self, position):
        """
        安全检查，确保机械臂不会碰撞
        :param position: 目标位置[x,y,z]
        :return: 是否安全
        """
        info_list = []
        if position[0] == 0 and position[1] == 0 and position[2] == 0:
            info_list.append('未识别到')
        if position[0] < -70:
            info_list.append('太左')
        if position[0] > 350:
            info_list.append('太右')
        if position[1] < 30:
            info_list.append('太近')
        if position[1] > 265:
            info_list.append('太远')
        if position[2] < -30:
            info_list.append('太低')
        if position[2] > 280:
            info_list.append('太高')

        if info_list:
            # 输出info_list中的信息，逗号分隔
            print(f"安全检查失败: {','.join(info_list)}")
            return False
        return True
    
    def request_stop(self):
        """外部调用以中断高级检测流程"""
        self._stop_requested = True

    def _check_interrupt(self):
        """检查用户或外部是否请求中断"""
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            print("检测到用户按下 ESC，准备中断流程")
            return True
        return self._stop_requested

    def _update_status_display(self, image, status_text, vehicle_moving=False):
        """在高级流程中实时更新状态信息"""
        if image is None:
            return
        display = image.copy()
        cv2.putText(display, f"状态: {status_text}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"车辆: {'移动中' if vehicle_moving else '静止'}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Advanced Status", display)
        cv2.waitKey(1)

    def auto_approach_without_obstacle_avoidance(self, target_distance=None, target_x=None, max_attempts=None):
        """
        自动逼近功能（无避障）- 改进版
        在运动过程中持续进行目标处理，提高响应速度
        :param target_distance: 目标Y轴距离(mm)，默认使用配置文件中的值
        :param target_x: 目标X轴位置(mm)，默认使用配置文件中的值
        :param max_attempts: 最大尝试次数，默认使用配置文件中的值
        :return: 是否成功抓取
        """
        try:
            # 使用配置文件中的参数或默认值
            target_distance = target_distance or Config.target_distance
            target_x = target_x or Config.target_x
            max_attempts = max_attempts or Config.max_attempts

            print(f"开始无避障自动逼近，目标距离: {target_distance}mm, 目标X: {target_x}mm")

            # 确保车辆停止
            self.car_move_func("stop", 0)
            time.sleep(0.5)

            attempts = 0

            # 设置容差范围
            x_tolerance = 40  # X容差（毫米）
            distance_tolerance = 40  # 距离容差（毫米）

            # 近距离时的容差范围
            near_x_tolerance = 60  # 近距离X容差（毫米）
            near_distance_tolerance = 30  # 近距离距离容差（毫米）

            # 近距离阈值（米）
            near_distance_threshold = 0.4
            
            # 记录移动状态
            is_moving = False
            move_start_time = 0
            move_direction = None
            move_duration = 0
            
            # 最大连续未检测次数
            max_no_detection = 5
            no_detection_count = 0

            while attempts < max_attempts:
                current_time = time.time()
                
                # 进行视觉处理
                intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = self.vision_system.vision_process()
                
                # 获取当前检测模式和速度因子
                detection_mode = self.vision_system.detection_mode

                # 在图像上显示当前状态
                cv2.putText(rgb_display, f"尝试: {attempts+1}/{max_attempts} | 无避障模式",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 如果正在移动，显示移动状态
                if is_moving:
                    elapsed_time = current_time - move_start_time
                    remaining_time = max(0, move_duration - elapsed_time)
                    cv2.putText(rgb_display, f"移动中: {move_direction}, 剩余 {remaining_time:.1f}秒",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    # 检查是否需要停止移动
                    if elapsed_time >= move_duration:
                        print(f"移动完成: {move_direction}")
                        self.car_move_func("stop", 0)
                        self.vision_system.set_vehicle_moving_state(False)
                        time.sleep(0.3)  # 等待车辆完全停止
                        is_moving = False
                        move_direction = None

                # 获取目标乒乓球的三维坐标，传递完整参数以支持云端视觉增强
                pos = self.vision_system.choose_pingpang_new(results_boxes, camera_coordinate_list, rgb, depth, aligned_depth_frame, depth_intrin)

                # 如果没有检测到乒乓球
                if pos is None:
                    print("未检测到乒乓球或深度数据无效")
                    no_detection_count += 1
                    
                    # 如果连续多次未检测到，停止车辆
                    if no_detection_count >= max_no_detection:
                        print(f"连续 {no_detection_count} 次未检测到乒乓球，停止车辆")
                        self.car_move_func("stop", 0)
                        self.vision_system.set_vehicle_moving_state(False)
                        time.sleep(0.5)
                        is_moving = False
                        no_detection_count = 0
                    
                    cv2.imshow('RGB image', rgb_display)
                    cv2.waitKey(1)
                    
                    # 如果正在移动，继续移动
                    if is_moving:
                        time.sleep(0.1)  # 短暂等待，减少CPU使用
                        continue
                    else:
                        attempts += 1
                        continue

                # 重置未检测计数
                no_detection_count = 0
                
                # 将摄像头坐标系转换为机械臂坐标系
                transformed_pos = self.vision_system.position_change(pos.copy())
                
                # 检查转换后的坐标是否有效
                if transformed_pos[0] == 0 and transformed_pos[1] == 0 and transformed_pos[2] == 0:
                    print("转换后的坐标无效")
                    cv2.imshow('RGB image', rgb_display)
                    cv2.waitKey(1)
                    
                    # 如果正在移动，继续移动
                    if is_moving:
                        time.sleep(0.1)
                        continue
                    else:
                        attempts += 1
                        continue
                
                # 计算当前位置与目标位置的差距
                current_x = transformed_pos[0]
                current_distance = transformed_pos[1]
                x_diff = current_x - target_x
                distance_diff = current_distance - target_distance
                
                print(f"当前位置: X={current_x}mm, 距离={current_distance}mm")
                print(f"位置差距: X差距={x_diff}mm, 距离差距={distance_diff}mm")
                
                # 在图像上显示位置信息
                cv2.putText(rgb_display, f"X差距: {x_diff}mm, 距离差距: {distance_diff}mm",
                            (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                
                # 计算目标的实际距离（米）
                actual_distance = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
                is_near_distance = actual_distance < near_distance_threshold
                
                # 根据距离选择容差
                if is_near_distance:
                    print(f"近距离模式（{actual_distance:.2f}m）：使用更严格的控制参数")
                    current_x_tolerance = near_x_tolerance
                    current_distance_tolerance = near_distance_tolerance
                else:
                    current_x_tolerance = x_tolerance
                    current_distance_tolerance = distance_tolerance
                
                # 判断是否已经达到目标位置
                if abs(x_diff) < current_x_tolerance and abs(distance_diff) < current_distance_tolerance:
                    print("已到达目标位置，准备抓取")
                    self.car_move_func("stop", 0)
                    self.vision_system.set_vehicle_moving_state(False)
                    time.sleep(0.5)
                    
                    # 最终确认位置
                    intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = self.vision_system.vision_process()
                    pos = self.vision_system.choose_pingpang_new(results_boxes, camera_coordinate_list, rgb, depth, aligned_depth_frame, depth_intrin)
                    
                    if pos is not None:
                        transformed_pos = self.vision_system.position_change(pos.copy())
                        
                        if transformed_pos[0] == 0 and transformed_pos[1] == 0 and transformed_pos[2] == 0:
                            print("最终位置确认时，转换后的坐标无效")
                            attempts += 1
                            continue
                        
                        current_x = transformed_pos[0]
                        current_distance = transformed_pos[1]
                        x_diff = current_x - target_x
                        distance_diff = current_distance - target_distance
                        
                        # 计算目标的实际距离（米）
                        actual_distance = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
                        final_is_near_distance = actual_distance < near_distance_threshold
                        
                        # 根据距离选择最终确认容差
                        if final_is_near_distance:
                            final_x_tolerance = 60  # 近距离最终确认的X容差（毫米）
                            final_distance_tolerance = 45  # 近距离最终确认的距离容差（毫米）
                        else:
                            final_x_tolerance = 50  # 远距离最终确认的X容差（毫米）
                            final_distance_tolerance = 50  # 远距离最终确认的距离容差（毫米）
                        
                        if abs(x_diff) < final_x_tolerance and abs(distance_diff) < final_distance_tolerance:
                            print("位置确认无误，开始抓取乒乓球")
                            self.arm_control(pos)
                            return True
                        else:
                            print(f"最终位置有偏差，X偏差: {abs(x_diff)}mm, 距离偏差: {abs(distance_diff)}mm，重新调整")
                    else:
                        print("最终确认时未检测到合适的乒乓球")
                
                # 如果正在移动，继续移动并监控
                elif is_moving:
                    # 检查是否需要调整移动方向（如果目标位置发生显著变化）
                    significant_change = False
                    
                    if move_direction in ["left", "right"] and abs(x_diff) < 20:
                        # 如果左右移动时，X差距已经很小，提前停止
                        significant_change = True
                    elif move_direction in ["front", "back"] and abs(distance_diff) < 20:
                        # 如果前后移动时，距离差距已经很小，提前停止
                        significant_change = True
                    
                    if significant_change:
                        print(f"目标位置发生显著变化，调整移动策略")
                        self.car_move_func("stop", 0)
                        self.vision_system.set_vehicle_moving_state(False)
                        time.sleep(0.3)
                        is_moving = False
                    else:
                        # 继续当前移动
                        cv2.imshow('RGB image', rgb_display)
                        cv2.waitKey(1)
                        time.sleep(0.1)  # 短暂等待，减少CPU使用
                        continue
                
                # 如果没有在移动，决定下一步移动
                else:
                    # 获取当前速度、检测模式和模型类型
                    current_speed = Config.speed  # 使用固定速度
                    detection_mode = self.vision_system.detection_mode
                    model_type = self.vision_system.model_type  # 获取当前模型类型
                    move_time_multiplier = self.vision_system.get_move_time_multiplier()

                    print(f"当前模式: {detection_mode}, 模型类型: {model_type}, 移动时间倍数: {move_time_multiplier}")

                    # 优先调整左右位置
                    if abs(x_diff) > abs(distance_diff) * 1.5 and abs(x_diff) > 30:
                        # 调整左右位置
                        direction = "right" if x_diff > 0 else "left"
                        
                        # 根据差距大小设置速度
                        speed = current_speed
                        if abs(x_diff) < 100:
                            speed = current_speed * 0.8
                        if is_near_distance:
                            speed = current_speed * 0.6
                        
                        print(f"调整左右位置: {direction}, 速度: {speed:.2f}, X偏差: {x_diff}mm")
                        self.car_move_func(direction, speed)
                        self.vision_system.set_vehicle_moving_state(True)
                        
                        # 移动时间与差距成正比
                        move_duration = min(0.3, abs(x_diff) / 200)
                        move_duration = max(0.2, move_duration)
                        
                        # 左右移动时间不需要翻倍
                        print(f"左右移动时间: {move_duration:.2f}秒")

                        # 记录移动状态
                        is_moving = True
                        move_start_time = current_time
                        move_direction = direction
                    
                    # 调整前后距离
                    elif abs(distance_diff) > 20:
                        # 调整前后距离
                        direction = "front" if distance_diff > 0 else "back"
                        
                        # 根据差距大小设置速度
                        speed = current_speed
                        if abs(distance_diff) < 100:
                            speed = current_speed * 0.8
                        if is_near_distance:
                            speed = current_speed * 0.6
                        
                        print(f"调整前后距离: {direction}, 速度: {speed:.2f}, 距离偏差: {distance_diff}mm")
                        self.car_move_func(direction, speed)
                        self.vision_system.set_vehicle_moving_state(True)
                        
                        # 移动时间与差距成正比
                        move_duration = min(0.5, abs(distance_diff) / 200)
                        move_duration = max(0.2, move_duration)

                        # 应用移动时间倍数：云端检测模式或大体积模型检测
                        if direction == "front" and (detection_mode == "cloud" or model_type == "heavy"):
                            original_duration = move_duration
                            move_duration *= move_time_multiplier
                            mode_info = "云端检测" if detection_mode == "cloud" else "大体积模型检测"
                            print(f"前进移动时间: {move_duration:.2f}秒 (基础时间 {original_duration:.2f}秒 * {move_time_multiplier}, {mode_info})")
                        else:
                            print(f"后退移动时间: {move_duration:.2f}秒")

                        # 记录移动状态
                        is_moving = True
                        move_start_time = current_time
                        move_direction = direction
                    
                    else:
                        # 位置已经很接近，微调
                        print("位置接近目标，进行微调")
                        if abs(x_diff) > abs(distance_diff):
                            direction = "right" if x_diff < 0 else "left"
                            speed = current_speed * 0.4
                            move_duration = 0.2
                            # 左右微调不需要翻倍
                            print(f"左右微调移动时间: {move_duration:.2f}秒")
                        else:
                            direction = "front" if distance_diff < 0 else "back"
                            speed = current_speed * 0.4
                            move_duration = 0.2
                        
                            # 应用移动时间倍数：云端检测模式或大体积模型检测
                            if direction == "front" and (detection_mode == "cloud" or model_type == "heavy"):
                                original_duration = move_duration
                                move_duration *= move_time_multiplier
                                mode_info = "云端检测" if detection_mode == "cloud" else "大体积模型检测"
                                print(f"前进微调移动时间: {move_duration:.2f}秒 (基础时间 {original_duration:.2f}秒 * {move_time_multiplier}, {mode_info})")
                            else:
                                print(f"后退微调移动时间: {move_duration:.2f}秒")

                        self.car_move_func(direction, speed)
                        self.vision_system.set_vehicle_moving_state(True)
                        
                        # 记录移动状态
                        is_moving = True
                        move_start_time = current_time
                        move_direction = direction
                
                # 显示图像
                cv2.imshow('RGB image', rgb_display)
                cv2.waitKey(1)
                
                # 如果没有在移动，增加尝试次数
                if not is_moving:
                    attempts += 1
                
                # 短暂等待，减少CPU使用
                time.sleep(0.1)
            
            print(f"已尝试 {max_attempts} 次，未能成功逼近并抓取")
            return False
            
        except Exception as e:
            print(f"发生异常: {e}")
            self.car_move_func("stop", 0)
            self.vision_system.set_vehicle_moving_state(False)
            time.sleep(0.5)
            raise  # 重新抛出异常以便调试
        finally:
            # 无论如何都确保车辆停止
            self.car_move_func("stop", 0)
            self.vision_system.set_vehicle_moving_state(False)

    
    def search_by_directions(self, model_type="light"):
        """
        按四个方向搜索乒乓球（每次90度）
        :param model_type: 使用的模型类型，"light" 或 "heavy"
        :return: 是否找到乒乓球，找到的位置
        """
        print(f"开始按四个方向搜索乒乓球，每次旋转90度，使用{model_type}模型...")
        
        # 记录原始模型类型，用于恢复
        original_model_type = self.vision_system.model_type
        
        # 切换到指定模型
        if original_model_type != model_type:
            self.vision_system.switch_model(model_type)
        
        directions_searched = 0
        found_pingpang = None
        
        try:
            # 搜索四个方向
            for i in range(Config.search_direction_count):
                print(f"搜索方向 {i+1}/{Config.search_direction_count}...")
                
                # 停止并稳定
                self.car_move_func("stop", 0)
                self.vision_system.set_vehicle_moving_state(False)
                time.sleep(Config.search_pause_time)
                
                # 在当前方向进行多次检测，提高可靠性
                detection_attempts = 3
                for attempt in range(detection_attempts):
                    # 进行视觉处理
                    intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = self.vision_system.vision_process()
                    
                    # 在图像上显示搜索状态
                    cv2.putText(rgb_display, f"方向搜索 {i+1}/{Config.search_direction_count} (尝试 {attempt+1}/{detection_attempts})",
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(rgb_display, f"使用模型: {model_type}",
                               (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
                    
                    # 检查是否检测到乒乓球
                    pos = self.vision_system.choose_pingpang_new(results_boxes, camera_coordinate_list, rgb, depth, aligned_depth_frame, depth_intrin)
                    
                    if pos is not None:
                        # 检测到乒乓球
                        print(f"在方向 {i+1} 检测到乒乓球，位置: {pos}")
                        found_pingpang = pos
                        
                        # 显示检测结果
                        cv2.putText(rgb_display, "发现乒乓球！", (50, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        # 保存当前方向的检测结果
                        directions_searched = i + 1
                        
                        return True, found_pingpang
                    
                    # 显示图像
                    cv2.imshow('RGB image', rgb_display)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                        break
                    
                    # 短暂等待
                    time.sleep(0.2)
                
                # 如果已经搜索了四个方向，退出循环
                if i == Config.search_direction_count - 1:
                    break
                
                # 旋转90度到下一个方向
                print(f"旋转90度到下一个方向...")
                
                # 计算旋转时间
                rotation_time = Config.search_pause_time
                
                # 执行旋转
                self.car_move_func("left", Config.search_speed)
                self.vision_system.set_vehicle_moving_state(True)
                time.sleep(rotation_time)
                self.car_move_func("stop", 0)
                self.vision_system.set_vehicle_moving_state(False)
                time.sleep(0.5)  # 等待稳定
            
            print(f"四个方向搜索完成，未找到乒乓球")
            return False, None
            
        except Exception as e:
            print(f"方向搜索过程中发生异常: {e}")
            return False, None
        finally:
            # 确保车辆停止
            self.car_move_func("stop", 0)
            self.vision_system.set_vehicle_moving_state(False)
            
            # 恢复原始模型
            if original_model_type != model_type:
                self.vision_system.switch_model(original_model_type)

    
    def _cloud_detection_fallback(self, target_distance, target_x, rgb_display):
        """
        云端兜底检测公共方法
        
        :param target_distance: 目标距离
        :param target_x: 目标X位置
        :param rgb_display: 显示图像
        :return: 是否成功抓取乒乓球
        """
        print("开始云端兜底检测（四个方向）...")
        self._update_status_display(rgb_display, "云端兜底检测", vehicle_moving=False)
        
        # 四个方向的云端检测
        directions_cloud_found = False
        original_rotation = self.car_move_func("get_rotation", 0)
        
        # 存储原始车辆移动状态
        original_moving_state = self.vision_system.vehicle_moving
        self.vision_system.set_vehicle_moving_state(False)
        
        found_pingpang = False
        
        for i in range(4):
            if self._check_interrupt():
                print("云端检测过程中被中断")
                self.vision_system.set_vehicle_moving_state(original_moving_state)
                return False
            
            # 进行一次视觉处理，获取最新图像
            intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = self.vision_system.vision_process()
            
            # 显示当前云端检测方向
            cv2.putText(rgb_display, f"云端检测方向 {i+1}/4", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2, cv2.LINE_AA)
            cv2.imshow('RGB image', rgb_display)
            cv2.waitKey(1)
            
            # 调用云端视觉API
            cloud_results = self.vision_system.call_cloud_vision(rgb, depth, aligned_depth_frame, depth_intrin)
            
            if cloud_results:
                cloud_boxes, cloud_coordinates = cloud_results
                
                if cloud_coordinates:
                    # 选择第一个检测结果
                    cloud_pos = cloud_coordinates[0]
                    print(f"云端检测方向 {i+1}/4 发现乒乓球，位置: {cloud_pos}，开始逼近...")
                    directions_cloud_found = True
                    
                    # 尝试逼近并抓取
                    success = self.auto_approach_without_obstacle_avoidance(
                        target_distance=target_distance,
                        target_x=target_x,
                        max_attempts=5
                    )
                    
                    if success:
                        found_pingpang = True
                        break
                    else:
                        print(f"云端检测方向 {i+1}/4 逼近失败，继续其他方向检测...")
            else:
                print(f"云端检测方向 {i+1}/4 未发现乒乓球")
            
            if found_pingpang:
                break
            
            # 旋转到下一个方向（90度）
            if i < 3:  # 最后一个方向不需要旋转
                rotation_time = self.calculate_rotation_time(Config.search_rotation_angle)
                self._update_status_display(rgb_display, "旋转到下一个方向", vehicle_moving=True)
                self.car_move_func("rotate", rotation_time)
                time.sleep(rotation_time + 0.5)  # 额外等待以确保稳定
        
        
        # 恢复车辆移动状态
        self.vision_system.set_vehicle_moving_state(original_moving_state)
        
        return found_pingpang

    def advanced_pingpong_detection_and_approach(self, target_distance=None, target_x=None, max_attempts=None):
        """
        高级乒乓球检测与逼近流程
        1. 先用轻量模型执行近距离检测
        2. 如果检测失败，进行360度方向转换搜索（每次90度）
        3. 如果四个方向都没有乒乓球，切换到大体积模型检测
        4. 大体积模型检测到目标后执行逼近
        5. 自动降级检测，一旦轻量模型能检测到，转为轻量模型
        6. 大体积模型四个方向都没检测到，进行云端兜底检测
        
        :param target_distance: 目标距离
        :param target_x: 目标X位置
        :param max_attempts: 最大尝试次数
        :return: 是否成功抓取
        """
        print("开始高级乒乓球检测与逼近流程...")
        self._stop_requested = False
        
        # 使用配置文件中的参数或默认值
        target_distance = target_distance or Config.target_distance
        target_x = target_x or Config.target_x
        max_attempts = max_attempts or Config.max_attempts
        
        # 保存原始模型类型
        original_model_type = self.vision_system.model_type
        
        try:
            # 第一步：确保使用轻量模型
            if not self.vision_system.switch_model("light"):
                print("已在使用轻量模型")
            
            # 检测尝试次数
            detection_attempts = 0
            found_pingpang = False
            
            while detection_attempts < max_attempts:
                if self._check_interrupt():
                    print("高级流程被用户中断")
                    return False

                detection_attempts += 1
                print(f"\n--- 检测尝试 {detection_attempts}/{max_attempts} ---")
                
                # 1. 使用当前模型（初始为轻量模型）进行检测
                intr, depth_intrin, rgb, depth, aligned_depth_frame, results, results_boxes, camera_coordinate_list, rgb_display = self.vision_system.vision_process()
                
                # 显示当前状态
                cv2.putText(rgb_display, f"检测尝试: {detection_attempts}/{max_attempts}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(rgb_display, f"当前模型: {self.vision_system.model_type}",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
                self._update_status_display(rgb_display, f"检测尝试 {detection_attempts}/{max_attempts}", vehicle_moving=False)
                
                # 检查是否检测到乒乓球
                pos = self.vision_system.choose_pingpang_new(results_boxes, camera_coordinate_list, rgb, depth, aligned_depth_frame, depth_intrin)
                
                # 自动降级检测
                if self.vision_system.model_type == "heavy" and pos is not None:
                    # 检查是否可以降级到轻量模型
                    if self.vision_system.check_model_downgrade(results_boxes):
                        print("执行模型自动降级")
                        self.vision_system.switch_model("light")
                
                if pos is not None:
                    # 检测到乒乓球，执行逼近
                    print(f"检测到乒乓球，位置: {pos}，开始逼近...")
                    cv2.putText(rgb_display, "检测到乒乓球，开始逼近", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    self._update_status_display(rgb_display, "检测到乒乓球，执行逼近", vehicle_moving=True)
                    self.vision_system.set_vehicle_moving_state(True)
                    
                    # 尝试逼近并抓取
                    success = self.auto_approach_without_obstacle_avoidance(
                        target_distance=target_distance,
                        target_x=target_x,
                        max_attempts=None  # 逼近时的尝试次数
                    )
                    self.vision_system.set_vehicle_moving_state(False)
                    
                    if success:
                        found_pingpang = True
                        break
                else:
                    print("未检测到乒乓球")
                    cv2.putText(rgb_display, "未检测到乒乓球", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # 2. 如果未检测到，进行四个方向搜索
                    print("开始四个方向搜索...")
                    self._update_status_display(rgb_display, "执行四方向搜索", vehicle_moving=True)
                    
                    # 保存搜索前的模型类型
                    search_model_type = self.vision_system.model_type
                    
                    # 使用当前模型进行四个方向搜索
                    found, pos = self.search_by_directions(model_type=search_model_type)
                    self.vision_system.set_vehicle_moving_state(False)
                    
                    if found and pos is not None:
                        # 检测到乒乓球，执行逼近
                        print(f"方向搜索中发现乒乓球，位置: {pos}，开始逼近...")
                        self._update_status_display(rgb_display, "方向搜索命中，执行逼近", vehicle_moving=True)
                        
                        # 尝试逼近并抓取
                        success = self.auto_approach_without_obstacle_avoidance(
                            target_distance=target_distance,
                            target_x=target_x,
                            max_attempts=5
                        )
                        self.vision_system.set_vehicle_moving_state(False)
                        
                        if success:
                            found_pingpang = True
                            break
                    else:
                        # 3. 如果四个方向都没找到，判断是否需要切换到大体积模型
                        if self.vision_system.model_type == "light":
                            print("轻量模型四个方向搜索失败，切换到大体积模型...")
                            self.vision_system.switch_model("heavy")
                            
                            # 使用大体积模型进行四个方向搜索
                            print("使用大体积模型进行四个方向搜索...")
                            self._update_status_display(rgb_display, "大体积模型搜索", vehicle_moving=True)
                            found, pos = self.search_by_directions(model_type="heavy")
                            self.vision_system.set_vehicle_moving_state(False)
                            
                            if found and pos is not None:
                                print(f"大体积模型发现乒乓球，位置: {pos}，开始逼近...")
                                self._update_status_display(rgb_display, "大体积模型命中，执行逼近", vehicle_moving=True)
                                
                                # 尝试逼近并抓取
                                success = self.auto_approach_without_obstacle_avoidance(
                                    target_distance=target_distance,
                                    target_x=target_x,
                                    max_attempts=5
                                )
                                self.vision_system.set_vehicle_moving_state(False)
                                
                                if success:
                                    found_pingpang = True
                                    break
                            else:
                                # 4. 大体积模型也没找到，尝试云端兜底检测 - 四个方向
                                if Config.use_cloud_vision:
                                    # 设置云端检测模式
                                    self.vision_system.detection_mode = "cloud"
                                    found_pingpang = self._cloud_detection_fallback(target_distance, target_x, rgb_display)
                                    # 恢复检测模式
                                    self.vision_system.detection_mode = "local"
                                    if found_pingpang:
                                        break
                        else:
                            # 已经是大体积模型且四个方向都没找到，尝试云端兜底检测 - 四个方向
                            if Config.use_cloud_vision:
                                # 设置云端检测模式
                                self.vision_system.detection_mode = "cloud"
                                found_pingpang = self._cloud_detection_fallback(target_distance, target_x, rgb_display)
                                # 恢复检测模式
                                self.vision_system.detection_mode = "local"
                                if found_pingpang:
                                    break
                
                # 显示图像
                cv2.imshow('RGB image', rgb_display)
                cv2.waitKey(1)
                
                # 短暂等待
                time.sleep(0.5)
            
            if found_pingpang:
                print("高级检测与逼近流程成功完成")
                return True
            else:
                print(f"已尝试 {max_attempts} 次，未能找到并抓取乒乓球")
                return False
                
        except Exception as e:
            print(f"高级检测与逼近流程发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 恢复原始模型
            if self.vision_system.model_type != original_model_type:
                print(f"恢复原始模型: {original_model_type}")
                self.vision_system.switch_model(original_model_type)
            
            # 确保车辆停止
            self.car_move_func("stop", 0)
            self.vision_system.set_vehicle_moving_state(False)