# TrackFlow 项目交接文档

> **生成时间**: 2026-02-02 12:40 UTC
> **项目进度**: 85%
> **当前阶段**: Phase 7 - 功能测试

---

## 项目概述

TrackFlow 是一个分布式视频分析系统，支持：
- YOLO OBB 目标检测 (ONNX Runtime + GPU)
- ByteTrack 目标跟踪
- 地理坐标转换 (像素 → WGS84)
- WebSocket 实时通信

---

## 仓库信息

- **GitHub**: https://github.com/AbeginnerFLM/TrackFlow.git
- **VPS 路径**: `/projects/TrackFlow`
- **GPU 服务器路径**: `~/TrackFlow`

---

## 架构

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│   浏览器/客户端   │◄────────────────►│   VPS (公网)     │
└─────────────────┘                    │  142.171.65.88  │
                                       │  FRP Server     │
                                       └───────┬─────────┘
                                               │ FRP隧道
                                       ┌───────▼─────────┐
                                       │ GPU服务器 (内网) │
                                       │  yolo_edge_server│
                                       │  YOLO推理+跟踪   │
                                       └─────────────────┘
```

---

## 核心文件结构

```
/projects/TrackFlow/
├── CMakeLists.txt              # 构建配置
├── config/config.yaml          # 运行时配置
├── include/
│   ├── core/                   # 核心框架
│   │   ├── processing_context.hpp
│   │   ├── image_processor.hpp
│   │   ├── processing_pipeline.hpp
│   │   └── processor_factory.hpp
│   ├── processors/             # 处理器
│   │   ├── yolo_detector.hpp
│   │   ├── byte_tracker.hpp
│   │   └── geo_transformer.hpp
│   ├── network/ws_server.hpp
│   └── utils/
├── src/                        # 实现代码
├── scripts/
│   ├── install_deps.sh         # 安装系统依赖
│   ├── install_onnxruntime.sh  # 安装ONNX Runtime GPU
│   ├── install_uwebsockets.sh  # 安装uWebSockets
│   ├── install_frpc.sh         # 安装FRP客户端(GPU服务器)
│   ├── build.sh                # 编译脚本
│   └── convert_model.py        # PyTorch→ONNX转换
├── models/                     # 模型目录
│   └── yolo26.onnx            # YOLO模型(需转换)
├── third_party/                # 第三方库
│   ├── onnxruntime/           # ONNX Runtime
│   └── uWebSockets/           # WebSocket库
└── test.html                   # 推理测试页面
```

---

## 服务器配置

### VPS (142.171.65.88)

| 服务 | 端口 | 配置文件 |
|------|------|----------|
| FRP Server | 7000 | /opt/frp/frps.toml |
| FRP Dashboard | 7500 | admin:trackflow2026 |
| HTTP测试页 | 8080 | python -m http.server |
| WebSocket代理 | 9001 | 转发到GPU服务器 |

**FRP服务端配置** `/opt/frp/frps.toml`:
```toml
bindPort = 7000
auth.token = "TrackFlow@2026!Secure"
webServer.addr = "0.0.0.0"
webServer.port = 7500
webServer.user = "admin"
webServer.password = "trackflow2026"
```

**管理命令**:
```bash
systemctl status frps
systemctl restart frps
```

### GPU服务器

| 服务 | 端口 | 说明 |
|------|------|------|
| yolo_edge_server | 9001 | WebSocket推理服务 |
| FRP Client | - | 连接VPS:7000 |

**FRP客户端配置** `/opt/frp/frpc.toml`:
```toml
serverAddr = "142.171.65.88"
serverPort = 7000
auth.token = "TrackFlow@2026!Secure"

[[proxies]]
name = "trackflow-ws"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9001
remotePort = 9001
```

---

## 开发命令速查

### VPS 开发

```bash
cd /projects/TrackFlow

# 编译
./scripts/build.sh

# 运行
./build/yolo_edge_server -p 9001 -v

# 启动测试页HTTP服务
python3 -m http.server 8080

# 推送代码
git add . && git commit -m "message" && git push
```

### GPU 服务器部署

```bash
cd ~/TrackFlow

# 拉取最新代码
git pull

# 首次安装依赖
sudo ./scripts/install_deps.sh
./scripts/install_onnxruntime.sh
./scripts/install_uwebsockets.sh
sudo ./scripts/install_frpc.sh

# 转换模型(需要yolo26.pt)
python3 scripts/convert_model.py models/yolo26.pt

# 编译
./scripts/build.sh

# 运行
./build/yolo_edge_server -p 9001 -v
```

---

## WebSocket API

### 连接
```javascript
const ws = new WebSocket('ws://142.171.65.88:9001');
```

### Ping 测试
```json
{"type": "ping"}
// 响应: {"type": "pong"}
```

### 推理请求
```json
{
  "type": "infer",
  "request_id": "req_123",
  "frame_id": 1,
  "image": "data:image/jpeg;base64,...",
  "config": {
    "yolo": {
      "model_path": "models/yolo26.onnx",
      "confidence": 0.5,
      "nms_threshold": 0.45
    },
    "tracker": {
      "enabled": true
    }
  }
}
```

### 推理响应
```json
{
  "type": "result",
  "request_id": "req_123",
  "frame_id": 1,
  "detections": [
    {
      "track_id": 1,
      "class_id": 0,
      "class_name": "ship",
      "confidence": 0.92,
      "obb": [x1,y1, x2,y2, x3,y3, x4,y4],
      "geo": {"lat": 31.23, "lng": 121.47}
    }
  ],
  "timing": {
    "preprocess_ms": 5,
    "inference_ms": 25,
    "postprocess_ms": 3
  }
}
```

---

## 已完成阶段

- [x] Phase 1: 项目基础设施 (CMake, 依赖安装脚本)
- [x] Phase 2: 核心框架 (Pipeline, Factory, Context)
- [x] Phase 3: 工具类 (ThreadPool, Base64, Config)
- [x] Phase 4: 处理器 (YOLO, ByteTracker, GeoTransformer)
- [x] Phase 5: 网络通信 (WebSocket Server)
- [x] Phase 6: 编译部署 (VPS+GPU+FRP)

---

## 当前任务: Phase 7 功能测试

### 待完成

1. **测试真实推理** - 发送图片验证YOLO检测
2. **验证跟踪功能** - 多帧测试ByteTracker
3. **验证坐标转换** - GeoTransformer测试
4. **创建前端演示** - 完整的Web界面
5. **性能优化** - GPU利用率

### 测试方法

1. 打开 http://142.171.65.88:8080/test.html
2. 点击 Connect 连接 WebSocket
3. 拖拽图片上传
4. 点击 Run Inference

### 可能问题

1. **模型不存在**: 需要 `python3 scripts/convert_model.py models/yolo26.pt`
2. **CUDA错误**: 检查ONNX Runtime GPU版本和CUDA版本匹配
3. **FRP连接失败**: 检查auth.token是否一致

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| C++标准 | C++20 | 现代特性、简洁 |
| 推理后端 | ONNX Runtime | 跨平台、GPU支持 |
| WebSocket | uWebSockets | 高性能、轻量 |
| JSON | nlohmann/json | Header-only、易用 |
| 跟踪算法 | ByteTrack | 简单高效 |
| 坐标转换 | PROJ | 专业地理库 |
| 内网穿透 | FRP | 稳定可靠 |

---

## 联系方式

- **GitHub**: https://github.com/AbeginnerFLM/TrackFlow
- **VPS IP**: 142.171.65.88

---

**接手须知**: 项目主体已完成，当前需要在GPU服务器上测试真实推理。如果遇到问题，检查模型文件和CUDA环境。
