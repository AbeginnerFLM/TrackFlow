# TrackFlow 项目交接文档

## 状态概览
- **核心系统**: C++ 后端功能完备，支持 GPU 加速 + 动态批处理推理
- **性能**: 推理延迟 ~28ms（内部），前端 v5 已按“连续播放 + 异步抽帧推理”方向优化，目标是稳定性、前端流畅度和并发吞吐
- **部署**: GPU 服务器（WSL）直接通过 Cloudflare Tunnel 对外暴露，**不再经过 VPS / FRP**

## 系统架构

### 方案A：Cloudflare Tunnel 远程访问（当前部署）

```text
用户浏览器 (任意网络)
    │
    ├── HTTPS (test_v4.html / test_v5.html)
    │
    └── WSS (/ws)
    ▼
Cloudflare Tunnel
    ▼
GPU 服务器 WSL
    ├── nginx (:8080)            # 静态文件 + WebSocket 反代
    └── yolo_edge_server (:9001) # C++ 推理服务
```

> Cloudflare Tunnel 负责公网接入与 TLS。性能判断请优先用本机直连结果，再参考远程效果。

### 方案B：本机直连（推荐用于性能测试）

```text
Windows 浏览器 (GPU 服务器本机)
    │
    ├── HTTP (test_v4.html / test_v5.html) ──→ WSL:8080 (nginx 静态文件)
    │
    └── WebSocket (/ws) ───────────────────→ WSL:8080 (nginx) ──→ WSL:9001 (yolo_edge_server)
```

> 本机直连无公网抖动，最适合定位前端采帧、编码、发送和后端推理的真实瓶颈。

### 推理管线架构

```text
前端 (MAX_INFLIGHT=4)
    │ 最多 4 帧并发飞行
    ▼
WebSocket Server (线程池)
    │
    ├── 阶段1: decode + YOLO (并行, 无锁)
    │     多帧可同时执行 decode 和 YOLO 推理
    │     BatchInferenceEngine 自动收集并发请求，合并为 batch GPU 推理
    │
    └── 阶段2: tracker (串行, 帧排序)
          wait_for_turn(frame_id) → ByteTracker.update() → advance_turn()
          确保 tracker 严格按帧序号处理，维持目标 ID 连续性
```

## 访问信息

| 资源 | 地址 | 认证 |
|------|------|------|
| 前端 v4 | Cloudflare Tunnel 域名上的 `/test_v4.html` | 按现网配置 |
| 前端 v5 | Cloudflare Tunnel 域名上的 `/test_v5.html` | 按现网配置 |
| WebSocket | 同源 `/ws`（页面自动按 `ws/wss` 拼接） | 按现网配置 |
| 前端 v4（本机） | `http://localhost:8080/test_v4.html` | 无 |
| 前端 v5（本机） | `http://localhost:8080/test_v5.html` | 无 |
| GPU SSH（直接） | `ssh 10.6.22.1` → `wsl` → `su - xf` | Windows: `4090`, xf: `123456xf` |
GPU SSH (从 VPS) | `ssh -p 9022 xf@localhost` | 密码: `123456xf` |

> 远程访问的具体公网域名由当前 Cloudflare Tunnel 配置决定；文档不再记录旧 VPS/FRP 地址。

## 快速操作

### 本机构建并启动服务

```bash
cd /projects/TrackFlow
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
nohup ./build/yolo_edge_server -c config/config.yaml > /dev/null 2>&1 &
```

### 仅重启服务（无代码变更）

```bash
killall -9 yolo_edge_server 2>/dev/null
cd /projects/TrackFlow
nohup ./build/yolo_edge_server -c config/config.yaml > /dev/null 2>&1 &
```

### nginx 配置（GPU WSL 本机）

配置文件: `/etc/nginx/sites-available/trackflow`

```nginx
server {
    listen 8080;
    root /projects/TrackFlow;

    location /ws {
        proxy_pass http://127.0.0.1:9001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

```bash
systemctl status nginx
systemctl restart nginx
```

### Cloudflare Tunnel 核查要点

- 确认 Tunnel 把外部域名转发到本机 `http://127.0.0.1:8080`
- 确认 `/ws` 路径允许 WebSocket upgrade
- 如出现远程偶发断流，先排查 tunnel/公网链路，再排查应用层

### 验证
1. 本机用 `http://localhost:8080/test_v5.html`，远程用当前 Tunnel 域名
2. 上传视频，点击 **Start** 开始推理
3. 观察 Stats 栏的 Latency / FPS / Objects
4. 做性能判断时优先看本机直连数据

## 关键注意事项

1. **静默模式**: 推理服务必须将输出重定向到 `/dev/null`。持续打印日志会明显拖慢推理
2. **性能排查顺序**: 先本机直连，再看 Cloudflare Tunnel 远程；否则容易把公网波动误判成推理瓶颈
3. **端口**:
   - `9001`: GPU WSL 本地推理服务端口
   - `8080`: nginx 本地端口（静态文件 + `/ws` 反代）
4. **模型文件**: 当前使用支持动态 batch 的 ONNX 模型；如更换模型需同步更新 `config/config.yaml`
5. **权限**: WSL 中 nginx 需要对 `/projects/TrackFlow` 以及页面文件有读取权限

## 前端 v5 现状（test_v5.html）

v5 在 v4 基础上新增了完整的坐标转换、车道检测和车辆分析功能。当前前端推理链路已按稳定性优先做了关键调整：

- 视频显示与推理采样解耦：**不再用 pause/play 做背压**
- 推理采样异步进行：按 `targetFps` 抽帧，`MAX_INFLIGHT=4` 控制飞行中请求数
- 结果返回后只更新最新状态，不再为每个 inflight 请求保存整帧 `ImageData`
- Canvas 改成 video + overlay 分层显示，减少因推理回包造成的画面抖动

### 布局

```text
┌─ TopBar ──────────────────────────────────────────────────┐
├──────────────────────────────┬──┬─────────────────────────┤
│                              │拖│  Tabs:                  │
│      Video Player            │拽│  · Vehicles (轨迹+属性) │
│   (video + canvas overlay)   │条│  · Settings (矩阵+校正) │
│                              │  │  · Lanes (车道配置)     │
├──────────────────────────────┤  │  · Data (CSV导出)      │
│  Timeline + Toolbar          │  │                         │
├──────────────────────────────┼──┴─────────────────────────┤
│  Stats (Latency/FPS/Objects) │                            │
├──────────────────────────────┴────────────────────────────┤
│  Log (可折叠)                                             │
└───────────────────────────────────────────────────────────┘
```

### 前端模块（纯 JS，无外部依赖）

| 模块 | 功能 | 对应原型 (`traj_keyong.py`) |
|------|------|-----------------------------|
| 坐标转换 | 像素→地面坐标 (单应性矩阵 3x3) | `image_to_ground()` / `cv2.perspectiveTransform` |
| 畸变校正 | Brown 模型迭代反畸变 | `cv2.undistortPoints` |
| UTM↔WGS84 | 纯公式实现，无需 pyproj | `utm_to_lonlat()` / pyproj |
| 车道检测 | ray-casting 点-多边形测试 | `cv2.pointPolygonTest` |
| 车辆跟踪面板 | 轨迹绘制 + 速度/方向/经纬度显示 | `update_vehicle_info()` |
| 数据导出 | CSV 格式（车辆数据 + 车道统计） | `export_data()` |

## 目录结构

```text
/projects/TrackFlow/          # 当前开发、构建与本机运行目录
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/main.cpp` | 入口点，命令行参数解析 |
| `src/processors/yolo_detector.cpp` | YOLO 推理 + 预处理/后处理 |
| `src/processors/batch_inference_engine.cpp` | 动态批处理推理引擎 |
| `src/processors/byte_tracker.cpp` | ByteTrack 多目标跟踪 |
| `src/network/ws_server.cpp` | WebSocket 服务，消息路由，并行管线调度 |
| `include/core/session.hpp` | 会话管理，per-session pipeline + 帧排序机制 |
| `config/config.yaml` | 运行时配置 |
| `test_v4.html` | 基础前端测试页 |
| `test_v5.html` | 增强版前端测试页（当前主优化对象） |
| `optimization.md` | 优化方案文档 |
| `debug.md` | 调试日志 |

## 故障排查

| 现象 | 排查步骤 |
|------|----------|
| 网页打不开 | 检查 GPU 服务器上 `systemctl status nginx` |
| WebSocket 连不上 | 1. 检查 `yolo_edge_server` 是否运行 2. 检查 nginx `/ws` 反代 3. 检查 Cloudflare Tunnel 健康状态 |
| 推理延迟高 | 1. 先本机直连复现 2. 检查浏览器采帧/编码负载 3. 检查 GPU 负载 (`nvidia-smi`) |
| 跟踪 ID 不稳定 | 1. 确认使用动态 batch 模型 2. 检查 frame_id 是否递增 |
| 远程偶发断流 | 1. 检查 Tunnel 与公网链路 2. 检查连接是否被中间链路回收 |
| 进程崩溃 | 查看服务日志或前台启动复现 |
