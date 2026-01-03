"""
Web control server for the ping-pong ball picking system.

Run on Jetson Orin Nano:
  python web_server.py --host 0.0.0.0 --port 8000

If hardware/libraries are missing, it falls back to Mock Mode for UI testing.
"""

import argparse
import threading
import time
import sys
import os
import random
import math
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # type: ignore
import numpy as np
import shutil
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

from config.config import Config

# Global placeholders
VisionSystem = None
Controller = None
init_arduino = None
init_arm = None

# Attempt to import hardware modules
try:
    from modules.hardware import init_arduino, init_arm
except ImportError as e:
    print(f"[Warning] Hardware module import failed: {e}")

try:
    from modules.vision import VisionSystem
except ImportError as e:
    print(f"[Warning] Vision module import failed: {e}")

try:
    from modules.controller import Controller, LidarMonitor
except ImportError as e:
    print(f"[Warning] Controller module import failed: {e}")
    LidarMonitor = None


app = FastAPI(title="PingPong Collector Web API")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOCK CLASSES (For UI testing without hardware) ---

class MockVisionSystem:
    def __init__(self):
        self.width = 848
        self.height = 480
        self.ball_x = 100
        self.ball_y = 100
        self.dx = 5
        self.dy = 5
        self.detection_mode = "mock"
        print("[Mock] Vision System initialized")

    def vision_process(self):
        # Create a blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw background grid
        for i in range(0, self.width, 50):
            cv2.line(img, (i, 0), (i, self.height), (30, 30, 30), 1)
        for i in range(0, self.height, 50):
            cv2.line(img, (0, i), (self.width, i), (30, 30, 30), 1)

        # Update ball position
        self.ball_x += self.dx
        self.ball_y += self.dy
        if self.ball_x <= 20 or self.ball_x >= self.width - 20: self.dx *= -1
        if self.ball_y <= 20 or self.ball_y >= self.height - 20: self.dy *= -1
        
        # Draw "Ping Pong Ball"
        cv2.circle(img, (int(self.ball_x), int(self.ball_y)), 20, (0, 165, 255), -1)
        cv2.putText(img, "MOCK MODE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, time.strftime("%H:%M:%S"), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Return mock values matching the real signature
        # intr, depth_intrin, rgb, depth, aligned_depth, results, boxes, coords, display
        return (None, None, img, None, None, [], [], [], img)

    def choose_pingpang(self, results_boxes, camera_coordinate_list):
        # Return a fake coordinate for picking
        return (0.5, 0.0, 1.0) # x, y, z

class MockController:
    def __init__(self, vision, move_func, arm):
        self.vision = vision
        self.car_move_func = move_func
        self.arm_serial = arm
    
    def auto_approach_without_obstacle_avoidance(self, **kwargs):
        print("[Mock] Auto approach running...")
        time.sleep(1)
        return False  # 返回False表示未成功抓取
    
    def advanced_pingpong_detection_and_approach(self, target_distance=None, target_x=None, max_attempts=None):
        """
        Mock版本的advanced_pingpong_detection_and_approach方法
        用于在真实Controller初始化失败时提供兼容接口
        """
        print("[Mock] Advanced pingpong detection and approach running...")
        time.sleep(2)  # 模拟检测和逼近过程
        print("[Mock] Advanced detection completed (mock mode)")
        return False  # 返回False表示未成功抓取
    
    def arm_control(self, pos):
        print(f"[Mock] Arm picking at {pos}")
        time.sleep(1)

# --- STATE MANAGEMENT ---

vision_instance = None
controller_instance = None
car_move_func = None
arm_serial = None
lidar_monitor = None
slam_system = None

vision_lock = threading.Lock()
task_lock = threading.Lock()
current_task: Optional[threading.Thread] = None
current_task_name: Optional[str] = None
current_mode: str = "idle"
auto_stop_flag = threading.Event()
system_fully_initialized = False

def init_vision_only():
    """仅初始化视觉系统（用于连接和视频流显示）"""
    global vision_instance
    
    if vision_instance is None:
        if VisionSystem is None:
            print("[Warning] VisionSystem class not available. Using Mock Vision System.")
            vision_instance = MockVisionSystem()
            return vision_instance
        
        try:
            vision_instance = VisionSystem()
            print("[Init] Vision system initialized for video streaming")
        except Exception as e:
            error_msg = f"Vision system initialization failed: {e}"
            print(f"[Warning] {error_msg}. Using Mock Vision System.")
            # 降级到Mock模式，避免重复尝试初始化
            vision_instance = MockVisionSystem()
    
    return vision_instance

def init_full_system():
    """初始化完整系统（用于自动捡球）"""
    global vision_instance, controller_instance, car_move_func, arm_serial
    global lidar_monitor, slam_system, system_fully_initialized
    
    if system_fully_initialized:
        print("[Init] System already fully initialized")
        return
    
    try:
        # 1. 初始化硬件 (Car & Arm)
        if car_move_func is None:
            try:
                if init_arduino:
                    result = init_arduino()
                    if isinstance(result, tuple) and len(result) >= 2:
                        car_move_func, _ = result[0], result[1]
                    else:
                        car_move_func = result
                    print("[Init] Arduino/Car initialized")
                else:
                    raise ImportError("init_arduino missing")
            except Exception as e:
                error_msg = f"Arduino/Car initialization failed: {e}"
                print(f"[ERROR] {error_msg}")
                raise RuntimeError(error_msg) from e

        if arm_serial is None:
            try:
                if init_arm:
                    arm_serial = init_arm()
                    print("[Init] Arm initialized")
                else:
                    raise ImportError("init_arm missing")
            except Exception as e:
                error_msg = f"Arm initialization failed: {e}"
                print(f"[ERROR] {error_msg}")
                raise RuntimeError(error_msg) from e

        # 2. 初始化SLAM系统
        try:
            from modules.slam import SlamSystem
            slam_system = SlamSystem()
            print("[Init] SLAM system initialized")
        except Exception as e:
            print(f"SLAM init failed: {e}. Continuing without SLAM.")
            slam_system = None

        # 3. 初始化激光雷达监控
        try:
            if LidarMonitor:
                lidar_monitor = LidarMonitor(slam_system=slam_system)
                lidar_monitor.start()
                print("[Init] Lidar monitor started")
            else:
                raise ImportError("LidarMonitor not available")
        except Exception as e:
            print(f"Lidar monitor init failed: {e}. Continuing without lidar.")
            lidar_monitor = None

        # 4. 初始化视觉系统（如果还没初始化）
        if vision_instance is None:
            try:
                vision_instance = init_vision_only()
            except Exception as e:
                error_msg = f"Vision system initialization failed: {e}"
                print(f"[ERROR] {error_msg}")
                raise RuntimeError(error_msg) from e
        
        # 将SLAM系统传递给VisionSystem
        if slam_system and hasattr(vision_instance, 'slam_system'):
            vision_instance.slam_system = slam_system
            print("[Init] SLAM system linked to Vision system")
        
        # 启动位姿同步线程 (SLAM -> Vision)
        if slam_system:
            def sync_pose():
                while not auto_stop_flag.is_set() and system_fully_initialized:
                    try:
                        pose = slam_system.get_pose()
                        if hasattr(vision_instance, 'update_robot_pose'):
                            vision_instance.update_robot_pose(*pose)
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Pose sync error: {e}")
                        break
            
            pose_thread = threading.Thread(target=sync_pose, daemon=True)
            pose_thread.start()
            print("[Init] Pose sync thread started")

        # 5. 初始化控制器
        if controller_instance is None:
            try:
                if Controller:
                    # Controller只接受3个位置参数：vision_system, car_move_func, arm_serial
                    controller_instance = Controller(
                        vision_instance, 
                        car_move_func, 
                        arm_serial
                    )
                    print("[Init] Controller initialized")
                else:
                    raise ImportError("Controller class missing")
            except Exception as e:
                error_msg = f"Controller initialization failed: {e}"
                print(f"[Warning] {error_msg}. Using Mock Controller.")
                controller_instance = MockController(vision_instance, car_move_func, arm_serial)
        
        system_fully_initialized = True
        print("[Init] Full system initialization completed")
        
    except Exception as e:
        print(f"[Init] Full system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def cleanup_full_system():
    """清理完整系统资源"""
    global lidar_monitor, system_fully_initialized
    
    if lidar_monitor:
        try:
            lidar_monitor.stop()
            print("[Cleanup] Lidar monitor stopped")
        except Exception as e:
            print(f"[Cleanup] Error stopping lidar: {e}")
    
    if car_move_func:
        try:
            car_move_func("stop", 0)
            print("[Cleanup] Car stopped")
        except Exception as e:
            print(f"[Cleanup] Error stopping car: {e}")
    
    system_fully_initialized = False

# --- ROUTES ---

class ROIModel(BaseModel):
    x: float
    y: float
    w: float
    h: float

class ROISelection(BaseModel):
    id: int

@app.get("/")
def root() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Error</h1><p>index.html not found</p>")

@app.get("/api/map/image")
def get_map_image():
    """
    Get the current map image.
    """
    map_path = Path(Config.map_path)
    if not map_path.exists():
        # Return a placeholder or 404
        return Response(content="Map not found", status_code=404)
    return FileResponse(map_path)

@app.post("/api/map/upload")
async def upload_map(file: UploadFile = File(...)):
    """
    Upload a SLAM map image.
    """
    try:
        file_location = Config.map_path
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        return Response(content=f"Upload failed: {str(e)}", status_code=500)

@app.post("/api/map/roi")
def save_roi(roi: ROIModel):
    """
    Save the Region of Interest (ROI) for operation.
    """
    Config.map_roi = [roi.x, roi.y, roi.w, roi.h]
    print(f"ROI updated: {Config.map_roi}")
    return {"status": "roi saved", "roi": Config.map_roi}

@app.post("/api/map/roi/select")
def select_roi(selection: ROISelection):
    """
    Select a predefined ROI region (0=Top, 1=Bottom, 2=Left, 3=Right).
    This updates the global Config to restrict robot movement.
    """
    Config.current_roi_id = selection.id
    print(f"ROI Selection changed to ID: {selection.id}")
    return {"status": "roi selected", "id": selection.id}

class ModeModel(BaseModel):
    mode: int  # 0: 自动模式, 1: 手动模式

@app.post("/api/mode/set")
def set_mode(mode_data: ModeModel):
    """
    Set system mode: 0 = Auto mode, 1 = Manual mode
    设置模式时会初始化完整系统，确保后续操作可以直接执行
    """
    mode_value = mode_data.mode
    if mode_value not in [0, 1]:
        return Response(content="Invalid mode. Must be 0 (auto) or 1 (manual)", status_code=400)
    
    Config.mode = mode_value
    mode_name = ['自动模式', '手动模式'][mode_value]
    print(f"System mode changed to: {mode_name} ({mode_value})")
    
    # 设置模式时初始化完整系统，确保后续操作可以直接执行
    try:
        if not system_fully_initialized:
            print(f"[Mode Set] 初始化系统以支持 {mode_name}...")
            init_full_system()
            print(f"[Mode Set] 系统初始化完成，{mode_name} 已就绪")
        else:
            print(f"[Mode Set] 系统已初始化，{mode_name} 已就绪")
    except Exception as e:
        error_msg = f"系统初始化失败: {str(e)}"
        print(f"[Mode Set] {error_msg}")
        return Response(
            content=error_msg,
            status_code=500
        )
    
    return {"status": "ok", "mode": mode_value, "mode_name": mode_name, "system_initialized": system_fully_initialized}

@app.get("/api/mode")
def get_mode():
    """
    Get current system mode
    """
    mode_value = getattr(Config, 'mode', 0)
    mode_name = ['自动模式', '手动模式'][mode_value]
    return {"mode": mode_value, "mode_name": mode_name}

@app.delete("/api/map/roi")
def clear_roi():
    """
    Clear the Region of Interest.
    """
    Config.map_roi = None
    print("ROI cleared")
    return {"status": "roi cleared"}

@app.get("/video")
def video_feed():
    # 仅初始化视觉系统用于视频流
    try:
        init_vision_only()
    except Exception as e:
        return Response(
            content=f"Vision system initialization failed: {str(e)}",
            status_code=503,
            media_type="text/plain"
        )
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

def mjpeg_generator():
    # vision_instance should already be initialized by /video endpoint
    if vision_instance is None:
        error_img = np.zeros((480, 848, 3), dtype=np.uint8)
        cv2.putText(error_img, "Vision system not initialized", (20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        while True:
            ok, jpg = cv2.imencode(".jpg", error_img)
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(1)
        return
    
    while True:
        with vision_lock:
            try:
                # Unpack tuple from vision_process
                ret = vision_instance.vision_process()
                rgb_display = ret[-1] # The last item is usually rgb_display in the original code
                
                # Double check format
                if rgb_display is None:
                    raise ValueError("No frame returned")
                
            except Exception as e:
                # Error frame
                rgb_display = np.zeros((480, 848, 3), dtype=np.uint8)
                cv2.putText(rgb_display, f"Error: {str(e)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                time.sleep(0.5)

        # Encode
        ok, jpg = cv2.imencode(".jpg", rgb_display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.1)
            continue
            
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        
        # Limit FPS to ~30
        time.sleep(0.03)

@app.get("/api/status")
def api_status():
    # 仅初始化视觉系统用于状态查询
    vision_ok = False
    detection_mode = "unknown"
    
    try:
        init_vision_only()
        vision_ok = True
        if vision_instance:
            detection_mode = getattr(vision_instance, 'detection_mode', 'unknown')
            # 如果是MockVisionSystem，标记为错误
            if isinstance(vision_instance, MockVisionSystem):
                vision_ok = False
                detection_mode = "error: mock mode active"
    except Exception as e:
        vision_ok = False
        detection_mode = f"error: {str(e)}"
    
    mode_text = "Idle"
    system_mode = getattr(Config, 'mode', 0)
    if hasattr(Config, 'mode'):
        try:
            mode_text = ['自动捡球', '手动回车捡球'][Config.mode]
        except:
            mode_text = "Unknown"
            
    return {
        "web_mode": current_mode,
        "task": current_task_name or "idle",
        "arm": bool(Config.arm_switch) if hasattr(Config, 'arm_switch') else False,
        "detection_mode": detection_mode,
        "vision_ok": vision_ok,
        "system_ready": system_fully_initialized,
        "system_mode": system_mode,  # 添加系统模式
    }

@app.post("/api/move")
def api_move(payload: dict):
    # 如果系统未初始化，返回错误（应该在设置模式时初始化）
    if not system_fully_initialized:
        return Response(
            content="系统未初始化，请先设置运行模式（自动或手动）",
            status_code=503
        )
    
    if car_move_func is None:
        return Response(content="Car not initialized", status_code=503)
    
    direction = payload.get("direction")
    speed = float(payload.get("speed", 0.4))
    duration = float(payload.get("duration", 0.5))
    
    if direction not in ["left", "right", "front", "back"]:
        return Response(content="invalid direction", status_code=400)
        
    car_move_func(direction, speed)
    
    # Auto stop thread
    def _stop_after():
        time.sleep(duration)
        # Only stop if we haven't started another move? 
        # For simplicity, we just stop. Real driver might need better logic.
        if car_move_func:
            car_move_func("stop", 0)
        
    threading.Thread(target=_stop_after, daemon=True).start()
    return {"status": "moved", "direction": direction}

@app.post("/api/stop")
def api_stop():
    global auto_stop_flag
    auto_stop_flag.set()
    
    if car_move_func:
        car_move_func("stop", 0)
    
    return {"status": "stopped"}

@app.post("/api/auto_start")
def api_auto_start():
    global current_task, current_task_name
    
    with task_lock:
        if current_task is not None and current_task.is_alive():
            return Response(content="Task already running", status_code=409)
        
        # 初始化完整系统
        try:
            init_full_system()
        except Exception as e:
            return Response(content=f"System initialization failed: {str(e)}", status_code=500)
        
        if controller_instance is None:
            return Response(content="Controller not initialized", status_code=503)
        
        current_task = threading.Thread(target=_run_auto_loop, daemon=True)
        current_task_name = "auto_continuous"
        current_task.start()
    
    return {"status": "started", "message": "自动捡球任务已启动"}

@app.post("/api/auto_stop")
def api_auto_stop():
    global current_mode
    auto_stop_flag.set()
    
    if car_move_func:
        car_move_func("stop", 0)
    
    current_mode = "idle"
    return {"status": "stopping", "message": "正在停止自动捡球任务"}

def _run_auto_loop():
    """运行自动捡球主循环（类似main.py的逻辑）"""
    global current_mode
    
    current_mode = "auto"
    auto_stop_flag.clear()
    
    try:
        print("[Auto Loop] 自动捡球任务开始")
        
        # 使用高级乒乓球检测与逼近流程（类似main.py）
        while not auto_stop_flag.is_set():
            if not controller_instance:
                print("[Auto Loop] Controller not available")
                break
            
            # 调用主程序的核心方法
            success = controller_instance.advanced_pingpong_detection_and_approach(
                target_distance=getattr(Config, 'target_distance', 500),
                target_x=getattr(Config, 'target_x', 0),
                max_attempts=getattr(Config, 'max_attempts', 3),
            )
            
            # 如果失败，根据配置决定是否继续
            if not success:
                print("[Auto Loop] 检测失败，等待后继续...")
                time.sleep(0.5)
            
            # 短暂暂停避免过度占用CPU
            time.sleep(0.1)
            
    except Exception as e:
        print(f"[Auto Loop] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        current_mode = "idle"
        if car_move_func:
            car_move_func("stop", 0)
        print("[Auto Loop] 自动捡球任务结束")

@app.post("/api/pick")
def api_pick():
    # 如果系统未初始化，返回错误（应该在设置模式时初始化）
    if not system_fully_initialized:
        return Response(
            content="系统未初始化，请先设置运行模式（自动或手动）",
            status_code=503
        )
    
    if controller_instance is None:
        return Response(content="Controller not initialized", status_code=503)
    
    try:
        if car_move_func:
            car_move_func("stop", 0)
        
        # Vision Process
        with vision_lock:
            if vision_instance is None:
                init_vision_only()
            
            ret = vision_instance.vision_process()
            # Unpack depending on what vision_process returns
            # Assuming standard signature: ..., results_boxes, camera_coordinate_list, ...
            if len(ret) >= 8:
                results_boxes = ret[6]
                camera_coordinate_list = ret[7]
            else:
                # Mock return
                results_boxes = []
                camera_coordinate_list = []
        
        # 使用新的选择方法（如果有）
        if hasattr(vision_instance, 'choose_pingpang_new'):
            rgb = ret[2] if len(ret) > 2 else None
            depth = ret[3] if len(ret) > 3 else None
            aligned_depth_frame = ret[4] if len(ret) > 4 else None
            depth_intrin = ret[1] if len(ret) > 1 else None
            pos = vision_instance.choose_pingpang_new(
                results_boxes, camera_coordinate_list, rgb, depth, aligned_depth_frame, depth_intrin
            )
        else:
            pos = vision_instance.choose_pingpang(results_boxes, camera_coordinate_list)
        
        if pos is None:
            return Response(content="No target found", status_code=404)
        
        if controller_instance:
            controller_instance.arm_control(pos)
        else:
            return Response(content="Controller not initialized", status_code=503)
        
        return {"status": "picked", "pos": pos}
    except Exception as e:
        return Response(content=f"Pick failed: {str(e)}", status_code=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run("web_server:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
