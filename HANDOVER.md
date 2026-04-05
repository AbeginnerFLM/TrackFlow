# TrackFlow 项目交接文档

## 状态概览
- **核心系统**: C++ 后端功能完备，支持 GPU 加速 + 动态批处理推理
- **性能**: 推理延迟 ~28ms (内部)，Batch 推理可合并多帧提升吞吐
- **部署**: GPU 服务器 (WSL) 通过 FRP 映射到 VPS，前端由 VPS nginx 提供
- **结构更新**: 前端已迁移到 `frontend/`，`test_v4.html` 已删除，根目录 `test_v5.html` / `calibration.html` 仅保留兼容跳转

## 系统架构

### 方案A: 远程访问 (通过 VPS)

```text
用户浏览器 (任意网络)
    │
    ├── HTTP (frontend/inference.html) ──→ VPS:8080 (nginx 静态文件)
    │
    └── WebSocket (/ws)               ──→ VPS:8080 (nginx) ──→ VPS:9002 (FRP) ──→ GPU:9001

VPS (142.171.65.88)
    ├── FRP Server (:7000)      # 反向代理服务
    ├── nginx (:8080)           # 静态文件 + WebSocket 反代 (/ws → 127.0.0.1:9002)
    ├── FRP → GPU SSH (:9022)   # SSH 到 GPU 服务器
    └── FRP → GPU WS (:9002)    # WebSocket 推理转发 (内部)

GPU 服务器 (WSL on Windows)
    ├── yolo_edge_server (:9001) # C++ 推理服务
    └── FRP Client               # 连接到 VPS
```

> ⚠️ FRP 带宽约 20KB/s，帧传输延迟高，仅适合远程测试。

### 方案B: 本机直连 (推荐用于性能测试)

```text
Windows 浏览器 (GPU 服务器本机)
    │
    ├── HTTP (frontend/inference.html) ──→ WSL:8080 (nginx 静态文件)
    │
    └── WebSocket (/ws)               ──→ WSL:8080 (nginx) ──→ WSL:9001 (yolo_edge_server)

GPU 服务器 WSL
    ├── yolo_edge_server (:9001) # C++ 推理服务
    └── nginx (:8080)            # 静态文件 + WebSocket 反代 (/ws → 127.0.0.1:9001)
```

> 本机直连无网络瓶颈，延迟即为纯推理耗时。浏览器访问 `http://localhost:8080/frontend/inference.html`

### 推理管线架构

```text
前端 (MAX_INFLIGHT=4)
    │ 同时发送最多4帧
    ▼
WebSocket Server (线程池 + 限流)
    │
    ├── 阶段1: decode + YOLO (并行)
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
| 前端推理页 (远程) | `http://142.171.65.88:8080/frontend/inference.html` | 无 |
| 前端标定页 (远程) | `http://142.171.65.88:8080/frontend/calibration.html` | 无 |
| 前端推理页 (本机) | `http://localhost:8080/frontend/inference.html` | 无 |
| 前端标定页 (本机) | `http://localhost:8080/frontend/calibration.html` | 无 |
| WebSocket | `ws://<host>:8080/ws` (nginx 反代, 同源自动路由) | 无 |
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
  --include='frontend/***' --include='CMakeLists.txt' --include='test_v5.html' --include='calibration.html' --exclude='*' \
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

## nginx 配置

### GPU WSL 本机

配置文件: `/etc/nginx/sites-available/trackflow`

```nginx
server {
    listen 8080;
    root /home/xf/TrackFlow;

    location /ws {
        proxy_pass http://127.0.0.1:9001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### VPS 远程

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

## 验证清单

1. 打开浏览器：`http://142.171.65.88:8080/frontend/inference.html`
2. 上传视频，点击 **Start** 开始推理
3. 检查 Stats 栏的 End-to-end / Server total / Overhead / FPS / Objects 数据
4. 打开 `frontend/calibration.html` 检查页面是否能正常加载视频与导出文件
5. 检查 reset、重连、长视频连续推理是否稳定

## 关键注意事项

1. **静默模式**: 推理服务必须将输出重定向到 `/dev/null`。通过 SSH 打印到终端会导致严重延迟。
2. **网络链路**: 远程访问经 VPS→FRP→GPU，FRP 带宽约 20KB/s，高延迟是网络瓶颈而非推理瓶颈。本机直连可排除网络因素。
3. **端口映射**:
   - `9001`: GPU WSL 本地推理服务端口
   - `8080`: nginx 端口 (VPS 和 GPU WSL 均使用此端口)
   - `9002`: VPS 上的 FRP 映射端口 (VPS nginx 内部转发到此)
   - `9022`: VPS 上的 FRP SSH 端口
4. **模型文件**: 当前使用 `models/3class4batch.onnx`，如更换模型，更新 `config/config.yaml` 中的 `yolo.model_path`。
5. **本机 nginx 权限**: WSL 中 nginx 以 www-data 运行，需要 `/home/xf` 目录有 o+x 权限，前端文件有 o+r 权限：
   ```bash
   sudo chmod o+x /home/xf
   sudo chmod o+x /home/xf/TrackFlow
   sudo chmod -R o+r /home/xf/TrackFlow/frontend
   sudo chmod o+r /home/xf/TrackFlow/test_v5.html
   sudo chmod o+r /home/xf/TrackFlow/calibration.html
   ```

## 当前前端组织

- `frontend/inference.html`：模块化推理页
- `frontend/calibration.html`：模块化标定页
- `frontend/assets/js/shared/*`：共享 DOM / 日志 / 地理 / 下载工具
- `frontend/assets/js/inference/app.js`：推理页主逻辑
- `frontend/assets/js/calibration/app.js`：标定页主逻辑
- 根目录 `test_v5.html` / `calibration.html`：兼容跳转入口

## 当前后端健壮性设计

- 服务端配置采用“后端默认值 + 前端允许覆盖项”的模式
- pipeline 由服务端根据 feature flags 生成，不再信任任意前端顺序
- `infer_header` 与 binary 配对错误会返回显式错误
- 线程池与 batch engine 已增加队列上限
- 每个 session 限制 outstanding 请求数
- session reset/过期清理会避免直接破坏正在执行的 worker 所持对象
- 若同一 `session_id` 的有效配置变化，会自动重建 pipeline

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/main.cpp` | 启动参数解析、服务端默认配置构建 |
| `src/network/ws_server.cpp` | WebSocket 协议、配置校验、限流、pipeline 生成、session 管理 |
| `include/core/session.hpp` | 会话生命周期、请求配额、配置变更重建 |
| `include/utils/thread_pool.hpp` | 有界线程池 |
| `src/processors/batch_inference_engine.cpp` | 有界 batch 推理队列 |
| `config/config.yaml` | 端口、线程、YOLO、tracker、feature、limit 配置 |
| `frontend/` | 模块化前端页面与资源 |
| `optimization.md` | 优化方案文档 |
| `debug.md` | 调试日志 |

## 故障排查

| 现象 | 排查步骤 |
|------|----------|
| 网页打不开 | 检查 VPS 上 `systemctl status nginx` |
| WebSocket 连不上 | 1. 检查 GPU 上 `yolo_edge_server` 是否在运行 2. 检查 FRP 隧道是否正常 |
| 推理延迟高 | 1. 确认输出重定向到 /dev/null 2. 检查网络带宽 3. 检查 GPU 负载 (`nvidia-smi`) |
| 跟踪 ID 不稳定 | 1. 确认 reset 与 frame_id 正常 2. 检查 session 是否因配置变化被重建 |
| 进程崩溃 | 查看 `nohup.out` 或直接前台运行查看 stderr |
| FRP 断连 | 在 GPU 上重启 `systemctl restart frpc` 或检查 VPS 上 `systemctl status frps` |
