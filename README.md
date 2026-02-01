# TrackFlow - 视频分析系统

基于现代C++20的分布式视频分析系统，支持YOLO目标检测、ByteTrack多目标跟踪和地理坐标转换。

## 特性

- **YOLO目标检测**: 支持ONNX Runtime GPU加速，兼容OBB和HBB检测
- **ByteTrack跟踪**: 实现多目标跟踪，带卡尔曼滤波预测
- **地理坐标转换**: 透视变换+UTM投影，像素坐标→经纬度
- **WebSocket通信**: 高性能uWebSockets，支持Base64图像传输
- **Pipeline架构**: 策略模式+责任链，易于扩展

## 目录结构

```
TrackFlow/
├── include/                 # 头文件
│   ├── core/               # 核心框架
│   ├── processors/         # 处理器
│   ├── network/            # 网络层
│   └── utils/              # 工具类
├── src/                    # 源文件
├── config/                 # 配置文件
├── models/                 # ONNX模型
├── scripts/                # 安装和构建脚本
└── third_party/            # 第三方库
```

## 快速开始

### 1. 安装依赖

```bash
# 系统依赖
sudo ./scripts/install_deps.sh

# ONNX Runtime GPU
./scripts/install_onnxruntime.sh

# uWebSockets
./scripts/install_uwebsockets.sh
```

### 2. 编译

```bash
./scripts/build.sh

# 或手动:
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 3. 运行

```bash
./build/yolo_edge_server -p 9001 -t 2

# 使用配置文件
./build/yolo_edge_server -c config/config.yaml
```

## WebSocket API

### 推理请求

```json
{
  "type": "infer",
  "request_id": "uuid-xxx",
  "session_id": "client-001",
  "frame_id": 42,
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "pipeline": ["decoder", "yolo", "tracker"],
  "config": {
    "yolo": {
      "model_path": "models/yolo_obb.onnx",
      "confidence": 0.5
    }
  }
}
```

### 响应

```json
{
  "type": "result",
  "request_id": "uuid-xxx",
  "frame_id": 42,
  "detections": [
    {
      "track_id": 1,
      "class_name": "car",
      "confidence": 0.92,
      "bbox": {
        "center": [500, 300],
        "size": [120, 60],
        "angle": 15.5
      }
    }
  ],
  "timing": {
    "decode_ms": 2.1,
    "infer_ms": 12.5,
    "track_ms": 0.8,
    "total_ms": 16.2
  }
}
```

## 可用处理器

| 名称 | 类型 | 说明 |
|------|------|------|
| `decoder` | ImageDecoder | Base64图像解码 |
| `yolo` | YoloDetector | YOLO目标检测 |
| `tracker` | ByteTracker | 多目标跟踪 |
| `geo_transform` | GeoTransformer | 坐标转换 |
| `undistort` | UndistortProcessor | 畸变校正 |

## 依赖

- C++20 编译器 (GCC 10+ / Clang 12+)
- CMake 3.20+
- OpenCV 4.x
- ONNX Runtime 1.17+ (GPU)
- uWebSockets
- spdlog
- nlohmann/json
- PROJ 9.x

## 许可证

MIT License
