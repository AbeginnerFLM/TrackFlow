# TrackFlow 项目交接文档

## 🟢 状态概览
- **核心系统**: C++ 后端功能完备，支持 GPU 加速。
- **性能**: 推理延迟 ~28ms (内部)。端到端延迟取决于网络带宽。
- **部署**: 已通过 FRP 部署在 GPU 服务器 (WSL) 上以供远程访问。

## 🔑 访问信息
- **GPU 服务器**: `127.0.0.1` (通过 SSH 隧道)
- **VPS**: `142.171.65.88`
- **前端 URL**: `http://142.171.65.88:8088/test_v4.html`

## 🛠️ 快速开始

### 1. 启动支持服务 (如果未运行)
**在 GPU 服务器上:**
```bash
# 启动前端 (Web 服务器)
cd /home/xx509/TrackFlow
nohup python3 -m http.server 8088 > /dev/null 2>&1 &

# 确保防火墙已开放
sudo ufw allow 8088
sudo ufw allow 9002
```

### 2. 启动推理服务器 (生产环境)
**在 GPU 服务器上:**
```bash
cd /home/xx509/TrackFlow

# 终止现有实例
killall -9 yolo_edge_server

# 以静默模式启动 (对性能至关重要)
nohup ./build/yolo_edge_server -c config/config.yaml > /dev/null 2>&1 &
```

### 3. 验证
1. 打开浏览器: `http://142.171.65.88:8088/test_v4.html`
2. 点击 **Connect** (应连接到 `ws://142.171.65.88:9002`)
3. 上传图片进行测试。


## ⚠️ 关键注意事项
1. **静默模式**: 必须将输出重定向到 `/dev/null` 或文件。通过 SSH 打印到控制台 (stdout/stderr) 会导致巨大的延迟 (观察者效应)。
2. **网络带宽**: 高延迟 (>1s) 目前是由于网络上传速度 (客户端 -> VPS -> GPU) 造成的。
3. **端口**:
   - `9001`: 内部本地端口 (服务器监听此处)
   - `9002`: 外部 FRP 端口 (客户端连接此处)

## 📁 关键文件
- `src/main.cpp`: 入口点。
- `src/processors/yolo_detector.cpp`: 核心推理逻辑。
- `src/network/ws_server.cpp`: WebSocket 处理。
- `config/config.yaml`: 运行时配置。
