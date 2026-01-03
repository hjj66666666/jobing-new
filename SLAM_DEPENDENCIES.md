# SLAM 模块依赖说明 (Jetson Orin Nano)

## 概述
`modules/slam.py` 模块基于 **BreezySLAM** 实现轻量级 2D 建图，用于 RPLidar A1 激光雷达的 SLAM 功能。

## 核心依赖

### 1. Python 库（必需）

#### BreezySLAM
**注意**：BreezySLAM **不在 PyPI 上**，需要从 GitHub 源码安装。

安装步骤（必须手动安装，setup.py 在 python 子目录中）：
```bash
# 1. 克隆仓库
git clone https://github.com/simondlevy/BreezySLAM.git
cd BreezySLAM/python

# 2. 安装（需要编译）
pip install .

# 或者使用 setup.py（需要 sudo，不推荐在虚拟环境中使用）
# sudo python3 setup.py install
```

**注意**：不能使用 `pip install git+https://github.com/simondlevy/BreezySLAM.git`，因为 setup.py 在 python 子目录中。

**系统依赖**：
- 需要 C 编译器（GCC）
- 可能需要安装开发工具：`sudo apt-get install build-essential python3-dev`

#### 基础库（已在 requirements.txt 中）
- `numpy>=1.20.0` - 数值计算
- `opencv-python>=4.5.0` - 图像处理和可视化
- `Pillow>=8.0.0` - 图像处理（PIL）

### 2. 系统依赖

#### OpenCV（如果未安装）
Jetson 上通常已预装 OpenCV，如果没有：
```bash
sudo apt-get update
sudo apt-get install python3-opencv libopencv-dev
```

### 3. 硬件依赖（可选但推荐）

#### RPLidar A1 激光雷达驱动
代码中使用 RPLidar A1，如果通过串口通信，可能需要：
```bash
# 如果使用 pyslam 或其他 RPLidar 驱动库
pip install rplidar-roboticia  # 或者使用其他 RPLidar 驱动
```

**注意**：当前代码中 SLAM 模块的 `update()` 方法接收原始扫描数据 `raw_scan`，假设调用者已经处理了 RPLidar 的串口通信。如果你需要 RPLidar 驱动，可以考虑：
- 使用 `rplidar-roboticia` 库
- 或直接通过 `pyserial` 读取 RPLidar 数据

## 完整安装步骤（Jetson Orin Nano）

### 步骤 1: 更新系统
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 步骤 2: 安装系统依赖（编译工具）
```bash
sudo apt-get install build-essential python3-dev
```

### 步骤 3: 安装 Python 依赖
```bash
# 激活你的 Python 环境（如果有虚拟环境）
# conda activate your_env  # 或 source venv/bin/activate

# 安装 BreezySLAM（必须手动安装，setup.py 在 python 子目录中）
git clone https://github.com/simondlevy/BreezySLAM.git
cd BreezySLAM/python
pip install .
cd ../..  # 返回项目根目录

# 安装其他基础依赖（如果还没安装）
pip install numpy opencv-python Pillow
```

### 步骤 4: 验证安装
```python
# 测试脚本 test_slam_import.py
try:
    from breezyslam.algorithms import RMHC_SLAM
    from breezyslam.sensors import Laser
    print("✓ BreezySLAM 安装成功")
except ImportError as e:
    print(f"✗ BreezySLAM 安装失败: {e}")

try:
    import cv2
    print(f"✓ OpenCV 版本: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV 未安装: {e}")

try:
    import numpy as np
    print(f"✓ NumPy 版本: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy 未安装: {e}")

try:
    from PIL import Image
    print("✓ Pillow 已安装")
except ImportError as e:
    print(f"✗ Pillow 未安装: {e}")
```

运行测试：
```bash
python test_slam_import.py
```

## 更新 requirements.txt

**注意**：BreezySLAM 不在 PyPI 上，且 setup.py 在 python 子目录中，**不能直接添加到 requirements.txt**。

只能在 requirements.txt 中添加注释说明：
```txt
# SLAM功能
# breezyslam 需要手动从 GitHub 安装：
# git clone https://github.com/simondlevy/BreezySLAM.git
# cd BreezySLAM/python
# pip install .

# RPLidar驱动（如果需要）
# rplidar-roboticia>=1.0.0  # RPLidar A1驱动（可选）
```

## 使用说明

### 1. 基本使用
如果 BreezySLAM 未安装，模块会自动降级，SLAM 功能将被禁用，但不会影响其他功能。

### 2. 检查 SLAM 状态
代码中有 `BREEZYSLAM_AVAILABLE` 标志，如果为 `False`，SLAM 功能将被禁用。

### 3. RPLidar 数据格式
`SlamSystem.update()` 方法期望的输入格式：
```python
raw_scan = [
    (quality, angle, distance),  # quality: 0-255, angle: 0-360, distance: mm
    (quality, angle, distance),
    ...
]
```

## 常见问题

### Q1: BreezySLAM 安装失败？
**A**: BreezySLAM **不在 PyPI 上**，需要从 GitHub 手动安装。如果失败，检查：
- Python 版本（建议 Python 3.8+）
- 是否安装了编译工具：`sudo apt-get install build-essential python3-dev`
- Git 是否安装：`sudo apt-get install git`
- 网络连接（需要访问 GitHub）
- **注意**：不能使用 `pip install breezyslam` 或 `pip install git+https://...`，因为 setup.py 在 python 子目录中，必须手动克隆并安装

### Q1.1: `pip install git+https://github.com/simondlevy/BreezySLAM.git` 失败？
**A**: 这是正常的！因为 BreezySLAM 的 setup.py 在 `python` 子目录中，不能直接从仓库根目录安装。必须：
1. 先克隆：`git clone https://github.com/simondlevy/BreezySLAM.git`
2. 进入 python 目录：`cd BreezySLAM/python`
3. 然后安装：`pip install .`

### Q2: 在 Jetson 上性能如何？
**A**: BreezySLAM 是一个轻量级库，在 Jetson Orin Nano 上运行良好。代码中地图更新频率已限制（每1-2秒一次），避免消耗过多 CPU。

### Q3: 需要 RPLidar 硬件吗？
**A**: 不需要。如果未连接 RPLidar，SLAM 功能会被禁用（`BREEZYSLAM_AVAILABLE = False`），但不会影响其他功能。

### Q4: 如何连接 RPLidar A1？
**A**: RPLidar A1 通过 USB 串口连接。确保：
- USB 串口设备存在（如 `/dev/ttyUSB0`）
- 用户有权限访问串口（将用户加入 `dialout` 组）：
  ```bash
  sudo usermod -a -G dialout $USER
  # 然后重新登录
  ```

## 相关资源

- [BreezySLAM GitHub](https://github.com/simondlevy/BreezySLAM)
- [RPLidar A1 规格](https://www.slamtec.com/en/Lidar/A1)
- [Jetson Orin Nano 官方文档](https://developer.nvidia.com/embedded/jetson-orin-nano)

