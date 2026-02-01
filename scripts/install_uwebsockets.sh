#!/bin/bash
# uWebSockets 安装脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"

echo "=========================================="
echo "  安装 uWebSockets"
echo "=========================================="

mkdir -p "$THIRD_PARTY_DIR"
cd "$THIRD_PARTY_DIR"

# 克隆 uWebSockets (如果不存在)
if [ ! -d "uWebSockets" ]; then
    echo ""
    echo "[1/3] 克隆 uWebSockets..."
    git clone --recurse-submodules https://github.com/uNetworking/uWebSockets.git
else
    echo "[1/3] uWebSockets 已存在,更新子模块..."
    cd uWebSockets
    git submodule update --init --recursive
    cd ..
fi

# 编译 uSockets
echo ""
echo "[2/3] 编译 uSockets..."
cd uWebSockets/uSockets

# 清理之前的构建
make clean 2>/dev/null || true

# 编译 (启用SSL支持)
make WITH_OPENSSL=1 WITH_LIBUV=0

echo ""
echo "[3/3] 验证..."
if [ -f "uSockets.a" ]; then
    echo "✓ uSockets.a 编译成功"
else
    echo "✗ 编译失败"
    exit 1
fi

cd "$THIRD_PARTY_DIR"

echo ""
echo "=========================================="
echo "  uWebSockets 安装完成!"
echo "=========================================="
echo ""
echo "安装路径: $THIRD_PARTY_DIR/uWebSockets"
