# TrackFlow 项目交接文档

> **更新时间**: 2026-02-04
> **项目进度**: 100%
> **当前阶段**: 已完成部署和调试

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
- **GPU 服务器路径**: `/home/xx509/TrackFlow`

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
├── include/                    # 头文件
├── src/                        # C++ 源码
│   ├── processors/
│   │   ├── yolo_detector.cpp   # YOLO 推理
│   │   ├── byte_tracker.cpp    # 目标跟踪
│   │   └── geo_transformer.cpp # 坐标转换
│   └── network/ws_server.cpp   # WebSocket 服务
├── scripts/
│   ├── build.sh                # 编译脚本
│   └── install_*.sh            # 依赖安装脚本
├── models/
│   ├── yolo26.pt               # PyTorch 模型
│   └── yolo26.onnx             # ONNX 模型 (1280x1280)
├── third_party/                # 第三方库
├── test_v4.html                # 浏览器测试页面
├── export_onnx.py              # ONNX 导出工具
└── debug.md                    # 调试日志
```

---

## 快速开始

### GPU 服务器部署

```bash
cd /home/xx509/TrackFlow

# 拉取代码
git fetch --all && git reset --hard origin/main

# 编译
./scripts/build.sh

# 启动服务
pkill -9 -f yolo_edge
nohup ./build/yolo_edge_server > server.log 2>&1 &
```

### 测试

1. 访问 http://142.171.65.88:8088/test_v4.html
2. 上传图片并点击 "Run Inference"

---

## 已解决的问题

详见 [debug.md](debug.md)

| 问题 | 解决方案 |
|------|----------|
| 大图导致崩溃 | 移除响应中的图片数据回显 |
| 检测框偏移 | 前端添加 uploadScale 缩放 |
| 置信度显示 100% | 识别 End-to-End 模型格式 |
| 检测框过大 | 使用 1280 分辨率导出 ONNX |
| GCC 编译崩溃 | 移除 spdlog，改用 fprintf |

---

## 服务器配置

### VPS (142.171.65.88)

| 服务 | 端口 | 说明 |
|------|------|------|
| FRP Server | 7000 | FRP 服务端 |
| HTTP 代理 | 8088 | 静态文件服务 |
| WebSocket | 9001 | 推理服务代理 |

### GPU 服务器 (内网)

| 服务 | 端口 | 说明 |
|------|------|------|
| yolo_edge_server | 9001 | WebSocket 推理 |

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 模型分辨率 | 1280x1280 | 4K 图像需要高分辨率输入 |
| 推理后端 | ONNX Runtime CPU | WSL 环境 CUDA 不稳定 |
| 日志库 | fprintf | 避免 spdlog 导致编译崩溃 |
| WebSocket | uWebSockets | 高性能、轻量 |

---

## 维护命令

```bash
# 查看服务状态
pgrep -f yolo_edge

# 重启服务
pkill -9 -f yolo_edge && nohup ./build/yolo_edge_server &

# 查看日志
tail -f server.log
```

---

**项目已完成部署，检测精度良好。如有问题请参考 debug.md。**
