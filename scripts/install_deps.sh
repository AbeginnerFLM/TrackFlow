#!/bin/bash
# TrackFlow 依赖安装脚本
# 用法: sudo ./scripts/install_deps.sh

set -e

echo "=========================================="
echo "  TrackFlow 依赖安装脚本"
echo "=========================================="

# 检查是否root权限
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 更新包列表
echo ""
echo "[1/6] 更新包列表..."
apt-get update

# 安装基础工具
echo ""
echo "[2/6] 安装基础编译工具..."
apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    curl

# 安装OpenCV
echo ""
echo "[3/6] 安装 OpenCV..."
apt-get install -y libopencv-dev

# 安装spdlog
echo ""
echo "[4/6] 安装 spdlog..."
apt-get install -y libspdlog-dev

# 安装nlohmann-json
echo ""
echo "[5/6] 安装 nlohmann-json..."
apt-get install -y nlohmann-json3-dev

# 安装PROJ
echo ""
echo "[6/6] 安装 PROJ..."
apt-get install -y libproj-dev

# 安装SSL和压缩库 (uWebSockets需要)
echo ""
echo "[额外] 安装 SSL和压缩库..."
apt-get install -y libssl-dev zlib1g-dev libuv1-dev

echo ""
echo "=========================================="
echo "  系统依赖安装完成!"
echo "=========================================="
echo ""
echo "接下来需要手动安装:"
echo "  1. ONNX Runtime (GPU版本)"
echo "  2. uWebSockets"
echo ""
echo "运行: ./scripts/install_onnxruntime.sh"
echo "运行: ./scripts/install_uwebsockets.sh"
