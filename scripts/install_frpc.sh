#!/bin/bash
# GPU服务器 FRP 客户端安装脚本
# 将此脚本复制到GPU服务器执行

set -e

echo "=========================================="
echo "  FRP 客户端安装脚本 (GPU服务器)"
echo "=========================================="

# 创建目录
mkdir -p /opt/frp
cd /opt/frp

# 下载 FRP
echo "[1/4] 下载 FRP..."
wget -q https://github.com/fatedier/frp/releases/download/v0.61.1/frp_0.61.1_linux_amd64.tar.gz
tar -xzf frp_0.61.1_linux_amd64.tar.gz --strip-components=1

# 创建配置文件
echo "[2/4] 创建配置文件..."
cat > frpc.toml << 'EOF'
serverAddr = "142.171.65.88"
serverPort = 7000

# 认证令牌（必须与服务端一致）
auth.token = "TrackFlow@2026!Secure"

[[proxies]]
name = "trackflow-ws"
type = "tcp"
localIP = "127.0.0.1"
localPort = 9001
remotePort = 9001
EOF

# 创建 systemd 服务
echo "[3/4] 创建 systemd 服务..."
cat > /etc/systemd/system/frpc.service << 'EOF'
[Unit]
Description=FRP Client Service
After=network.target

[Service]
Type=simple
ExecStart=/opt/frp/frpc -c /opt/frp/frpc.toml
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
echo "[4/4] 启动服务..."
systemctl daemon-reload
systemctl enable frpc
systemctl start frpc

echo ""
echo "=========================================="
echo "  FRP 客户端安装完成!"
echo "=========================================="
echo ""
echo "状态检查: systemctl status frpc"
echo "查看日志: journalctl -u frpc -f"
