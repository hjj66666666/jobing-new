# import pyrealsense2 as rs

# # 配置并启动数据流
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # 开始流式传输
# pipeline.start(config)

# try:
#     # 等待10帧让摄像头预热
#     for _ in range(10):
#         frames = pipeline.wait_for_frames()
    
#     # 获取深度和彩色帧
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
    
#     if not depth_frame or not color_frame:
#         print("❌ 未检测到深度或彩色帧！")
#     else:
#         print("✅ 摄像头工作正常！")
#         print(f"深度帧分辨率: {depth_frame.width}x{depth_frame.height}")
#         print(f"彩色帧分辨率: {color_frame.width}x{color_frame.height}")

# finally:
#     pipeline.stop()



from ultralytics import YOLO

# Load a model
model = YOLO("./table_tennis.pt")

# result = yolo(source=("4.jpg"), save = True)


# from ultralytics import YOLO

# # Load a model
# model = YOLO("./runs/detect/train/weights/best.pt")  # load a custom trained

# Export the model
model.export(format='engine', imgsz=640, half=True,simplify=True)

# Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# # Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model

# import numpy as np
# import torch
# import torchvision

# print("NumPy版本:", np.__version__)  # 应显示1.23.5
# print("PyTorch版本:", torch.__version__)
# print("TorchVision版本:", torchvision.__version__)
