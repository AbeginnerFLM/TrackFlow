# TrackFlow 项目交接文档

## 状态概览
- **核心系统**: C++ 后端功能完备，支持 GPU 加速 + 动态批处理推理
- **性能**: 推理延迟 ~28ms (内部)，Batch 推理可合并多帧提升吞吐
- **部署**: GPU 服务器 (WSL) 通过 FRP 映射到 VPS，前端由 VPS nginx 提供

## 系统架构

```
用户浏览器
    │
    ├── HTTP (test_v4.html)    ──→  VPS:8080 (nginx 静态文件)
    │
    └── WebSocket (/ws)        ──→  VPS:8080 (nginx) ──→ VPS:9002 (FRP) ──→ GPU:9001

VPS (142.171.65.88)
    ├── FRP Server (:7000)       # 反向代理服务
    ├── nginx (:8080)            # 静态文件 + WebSocket 反代 (/ws → 127.0.0.1:9002)
    ├── FRP → GPU SSH (:9022)    # SSH 到 GPU 服务器
    └── FRP → GPU WS (:9002)    # WebSocket 推理转发 (内部)

GPU 服务器 (WSL on Windows)
    ├── yolo_edge_server (:9001) # C++ 推理服务
    └── FRP Client               # 连接到 VPS
```

### 推理管线架构

```
前端 (MAX_INFLIGHT=4)
    │ 同时发送最多4帧
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
| 前端页面 | `http://142.171.65.88:8080/test_v4.html` | 无 |
| WebSocket | `ws://142.171.65.88:8080/ws` (同源, nginx 反代) | 无 |
| GPU SSH (从 VPS) | `ssh -p 9022 xf@localhost` | 密码: `123456xf` |
| GPU SSH (直接) | `ssh 10.6.22.1` → `wsl` → `su - xf` | Windows: `4090`, xf: `123456xf` |
| FRP 仪表盘 | `http://142.171.65.88:7500` | admin / trackflow2026 |
| VPS SSH | `ssh -p 12345 root@142.171.65.88` | - |

## 快速操作

### 从 VPS 一键部署 (推荐)

```bash
# 1. 同步代码到 GPU 服务器
cd /projects/TrackFlow
sshpass -p '123456xf' rsync -avz \
  -e "ssh -o StrictHostKeyChecking=no -p 9022" \
  --exclude='.git' --exclude='build/' --exclude='.cache/' --exclude='third_party/' \
  --include='src/***' --include='include/***' --include='config/***' \
  --include='CMakeLists.txt' --include='test_v4.html' --exclude='*' \
  ./ xf@localhost:/home/xf/TrackFlow/

# 2. 杀死旧进程 + 重新编译 + 启动
sshpass -p '123456xf' ssh -p 9022 xf@localhost "
  cd /home/xf/TrackFlow &&
  killall -9 yolo_edge_server 2>/dev/null;
  cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j\$(nproc) &&
  cd .. && nohup ./build/yolo_edge_server -c config/config.yaml > /dev/null 2>&1 &
"

# 3. 验证
sshpass -p '123456xf' ssh -p 9022 xf@localhost "ps aux | grep yolo_edge_server | grep -v grep"
```

### 仅重启服务 (无代码变更)

```bash
sshpass -p '123456xf' ssh -p 9022 xf@localhost "
  killall -9 yolo_edge_server 2>/dev/null;
  cd /home/xf/TrackFlow &&
  nohup ./build/yolo_edge_server -c config/config.yaml > /dev/null 2>&1 &
"
```

### nginx 配置 (VPS 上)

配置文件: `/etc/nginx/sites-available/trackflow`

```nginx
server {
    listen 8080;
    root /projects/TrackFlow;
    location /ws {
        proxy_pass http://127.0.0.1:9002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

```bash
# 检查 nginx 是否运行
systemctl status nginx

# 重启 nginx
systemctl restart nginx
```

### 验证
1. 打开浏览器: `http://142.171.65.88:8080/test_v4.html`
2. 上传视频，点击 **Start** 开始推理
3. 检查 Stats 栏的 Latency / FPS / Objects 数据

## 关键注意事项

1. **静默模式**: 推理服务必须将输出重定向到 `/dev/null`。通过 SSH 打印到终端会导致严重延迟 (观察者效应)
2. **网络链路**: 客户端 → VPS → FRP → GPU 存在网络开销，高延迟通常是带宽问题而非推理瓶颈
3. **端口映射**:
   - `9001`: GPU 服务器本地监听端口
   - `9002`: VPS 上的 FRP 映射端口 (nginx 内部转发, 客户端不直接访问)
   - `9022`: VPS 上的 FRP SSH 端口 (映射到 GPU WSL 的 SSH)
   - `8080`: VPS 上的 nginx 端口 (前端页面 + WebSocket 反代)
4. **模型文件**: 当前使用 `yolo26_4batch.onnx`，支持动态 batch (N=1~4)。如更换模型需同时更新 `config/config.yaml` 中的 `model_path`

## 目录结构

```
/projects/TrackFlow/          # VPS 上的代码仓库 (开发 + 前端服务)
/home/xf/TrackFlow/           # GPU 服务器上的代码 (编译 + 运行)
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/main.cpp` | 入口点，命令行参数解析 |
| `src/processors/yolo_detector.cpp` | YOLO 推理 + 预处理/后处理 (thread_local 线程安全) |
| `src/processors/batch_inference_engine.cpp` | 动态批处理推理引擎 (单例, 合并多帧 GPU 推理) |
| `src/processors/byte_tracker.cpp` | ByteTrack 多目标跟踪 (Hungarian 匹配) |
| `src/network/ws_server.cpp` | WebSocket 服务，消息路由，并行管线调度 |
| `include/core/session.hpp` | 会话管理，per-session pipeline + 帧排序机制 |
| `include/core/processing_pipeline.hpp` | 管线框架，支持 `execute_range()` 分段执行 |
| `config/config.yaml` | 运行时配置 (端口、线程数、模型路径) |
| `test_v4.html` | 前端页面 (视频推理 + 可视化, MAX_INFLIGHT=4) |
| `export_onnx.py` | ONNX 导出脚本 (支持动态 batch) |
| `optimization.md` | 优化方案文档 (含完成状态标记) |
| `debug.md` | 调试日志 (历史问题及解决方案) |

## FRP 配置

**VPS 端** (`/opt/frp/frps.toml`):
```toml
bindPort = 7000
auth.token = "TrackFlow@2026!Secure"
```

**GPU 端** (`/etc/frp/frpc.toml`):
```toml
serverAddr = "142.171.65.88"
serverPort = 7000
auth.token = "TrackFlow@2026!Secure"

[[proxies]]
name = "gpu-ssh"
type = "tcp"
localPort = 2222
remotePort = 9022

[[proxies]]
name = "trackflow-ws"
type = "tcp"
localPort = 9001
remotePort = 9002
```

## 故障排查

| 现象 | 排查步骤 |
|------|----------|
| 网页打不开 | 检查 VPS 上 `systemctl status nginx` |
| WebSocket 连不上 | 1. 检查 GPU 上 `yolo_edge_server` 是否在运行 2. 检查 FRP 隧道是否正常 (`curl http://localhost:7500`) |
| 推理延迟高 | 1. 确认输出重定向到 /dev/null 2. 检查网络带宽 3. 检查 GPU 负载 (`nvidia-smi`) |
| 跟踪 ID 不稳定 | 1. 确认使用 dynamic batch 模型 2. 检查帧排序是否正常 (日志中 frame_id 应递增) |
| 进程崩溃 | 查看 `nohup.out` 或用 `journalctl` 查日志 |
| FRP 断连 | 在 GPU 上重启 `systemctl restart frpc` 或检查 VPS 上 `systemctl status frps` |
