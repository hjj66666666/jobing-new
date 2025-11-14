"""
Web control server for the ping-pong ball picking system.

Run on Jetson Orin Nano:
  python web_server.py --host 0.0.0.0 --port 8000

Then open http://<JETSON_IP>:8000 in a browser.
"""

import argparse
import io
import threading
import time
from typing import Optional

import cv2  # type: ignore
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from config.config import Config
from modules.hardware import init_arduino, init_arm
from modules.vision import VisionSystem
from modules.controller import Controller


app = FastAPI(title="PingPong Collector Web API")

# Allow all origins for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global system state
vision_system: Optional[VisionSystem] = None
car_move_func = None
arm_serial = None
controller: Optional[Controller] = None

vision_lock = threading.Lock()
task_lock = threading.Lock()
current_task: Optional[threading.Thread] = None
current_task_name: Optional[str] = None
current_mode: str = "idle"  # idle | auto | manual
auto_stop_flag = threading.Event()


def init_system_once():
    global vision_system, car_move_func, arm_serial, controller
    if vision_system is None or controller is None or car_move_func is None:
        car_move_func, _ = init_arduino()
        arm_serial = init_arm()
        vision_system = VisionSystem()
        controller = Controller(vision_system, car_move_func, arm_serial)


def mjpeg_generator():
    assert vision_system is not None
    while True:
        with vision_lock:
            try:
                (
                    intr,
                    depth_intrin,
                    rgb,
                    depth,
                    aligned_depth_frame,
                    results,
                    results_boxes,
                    camera_coordinate_list,
                    rgb_display,
                ) = vision_system.vision_process()
            except Exception as e:
                # In case camera hiccups, show a black frame with error text
                rgb_display = (255 * 0).astype("uint8") if False else None
                # create a placeholder image
                img = 255 * (cv2.ones((480, 848, 3), dtype="uint8")) if hasattr(cv2, "ones") else None
                if img is None:
                    img = (255 * cv2.UMat(480, 848, cv2.CV_8UC3)).get()
                rgb_display = img
                cv2.putText(rgb_display, f"Camera error: {str(e)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ok, jpg = cv2.imencode(".jpg", rgb_display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            time.sleep(0.05)
            continue
        frame = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.get("/")
def root() -> HTMLResponse:
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PingPong Collector</title>
    <style>
      body { font-family: sans-serif; margin: 16px; }
      .row { margin: 8px 0; }
      button { margin-right: 8px; padding: 8px 12px; }
      img { max-width: 100%; border: 1px solid #ccc; }
      .small { font-size: 12px; color: #666; }
    </style>
  </head>
  <body>
    <h2>PingPong Collector</h2>
    <div class="row">
      <img id="video" src="/video" alt="video" />
    </div>
    <div class="row">
      <button onclick="call('/api/auto_start')">自动拾取开始</button>
      <button onclick="call('/api/auto_stop')">自动拾取停止</button>
      <button onclick="call('/api/stop')">急停</button>
    </div>
    <div class="row">
      <button onclick="move('left')">Left</button>
      <button onclick="move('right')">Right</button>
      <button onclick="move('front')">Front</button>
      <button onclick="move('back')">Back</button>
      <button onclick="call('/api/pick')">确定拾取</button>
      <button onclick="stopMove()">Stop</button>
    </div>
    <div class="row small" id="status"></div>
    <script>
      async function call(path, method='POST', body=null){
        const res = await fetch(path, {method, headers: {'Content-Type':'application/json'}, body: body?JSON.stringify(body):null});
        const txt = await res.text();
        document.getElementById('status').innerText = txt;
      }
      async function move(direction){
        await call('/api/move', 'POST', {direction, speed: 0.4, duration: 0.3});
      }
      async function stopMove(){
        await call('/api/stop');
      }
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/video")
def video_feed():
    init_system_once()
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    init_system_once()
    mode_text = ['自动捡球', '手动回车捡球', '云端决策捡球'][Config.mode]
    return {
        "mode": mode_text,
        "arm": bool(Config.arm_switch),
        "cloud_vision": bool(getattr(vision_system, 'use_cloud_vision', False)),
        "detection_mode": getattr(vision_system, 'detection_mode', 'local') if vision_system else 'unknown',
        "task": current_task_name or "idle",
        "web_mode": current_mode,
    }


@app.post("/api/move")
def api_move(payload: dict):
    init_system_once()
    direction = payload.get("direction")
    speed = float(payload.get("speed", 0.4))
    duration = float(payload.get("duration", 0.5))
    if direction not in ["left", "right", "front", "back"]:
        return Response(content="invalid direction", status_code=400)
    car_move_func(direction, speed)
    def _stop_after():
        time.sleep(duration)
        car_move_func("stop", 0)
    threading.Thread(target=_stop_after, daemon=True).start()
    return f"move {direction} {speed} for {duration}s"


@app.post("/api/stop")
def api_stop():
    init_system_once()
    car_move_func("stop", 0)
    return "stopped"


def _run_auto_once():
    assert controller is not None
    try:
        controller.auto_approach_without_obstacle_avoidance(
            target_distance=Config.target_distance,
            target_x=Config.target_x,
            max_attempts=Config.max_attempts,
        )
    finally:
        global current_task, current_task_name
        with task_lock:
            current_task = None
            current_task_name = None


@app.post("/api/auto_once")
def api_auto_once():
    init_system_once()
    global current_task, current_task_name
    with task_lock:
        if current_task is not None and current_task.is_alive():
            return Response(content="task already running", status_code=409)
        current_task = threading.Thread(target=_run_auto_once, daemon=True)
        current_task_name = "auto_once"
        current_task.start()
    return "started auto_once"


def _run_auto_continuous():
    global current_mode
    assert controller is not None
    current_mode = "auto"
    auto_stop_flag.clear()
    try:
        while not auto_stop_flag.is_set():
            controller.auto_approach_without_obstacle_avoidance(
                target_distance=Config.target_distance,
                target_x=Config.target_x,
                max_attempts=Config.max_attempts,
            )
            # 小间隔防止紧凑循环
            time.sleep(0.2)
    finally:
        current_mode = "idle"
        car_move_func("stop", 0)


@app.post("/api/auto_start")
def api_auto_start():
    init_system_once()
    global current_task, current_task_name
    with task_lock:
        if current_task is not None and current_task.is_alive():
            return Response(content="task already running", status_code=409)
        current_task = threading.Thread(target=_run_auto_continuous, daemon=True)
        current_task_name = "auto_continuous"
        current_task.start()
    return "started auto"


@app.post("/api/auto_stop")
def api_auto_stop():
    auto_stop_flag.set()
    car_move_func("stop", 0)
    return "stopping auto"


@app.post("/api/pick")
def api_pick():
    init_system_once()
    # 手动拾取：当前画面选择最近的乒乓球并抓取
    try:
        # 先停稳
        car_move_func("stop", 0)
        # 取一帧并寻找目标
        with vision_lock:
            (
                intr,
                depth_intrin,
                rgb,
                depth,
                aligned_depth_frame,
                results,
                results_boxes,
                camera_coordinate_list,
                rgb_display,
            ) = vision_system.vision_process()
        pos = vision_system.choose_pingpang(results_boxes, camera_coordinate_list)
        if pos is None:
            return Response(content="no target", status_code=404)
        controller.arm_control(pos)
        return "picked"
    except Exception as e:
        return Response(content=f"pick failed: {str(e)}", status_code=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn  # type: ignore
    uvicorn.run("web_server:app", host=args.host, port=args.port, reload=False, access_log=False)


if __name__ == "__main__":
    main()


