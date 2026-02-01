#!/bin/bash
# TrackFlow 构建脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# 默认构建类型
BUILD_TYPE="${1:-Release}"

echo "=========================================="
echo "  TrackFlow 构建脚本"
echo "  构建类型: $BUILD_TYPE"
echo "=========================================="

# 创建构建目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行CMake
echo ""
echo "[1/2] 配置项目..."
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..

# 编译
echo ""
echo "[2/2] 编译..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "  构建完成!"
echo "=========================================="
echo ""
echo "可执行文件: $BUILD_DIR/yolo_edge_server"
echo ""
echo "运行: ./build/yolo_edge_server --help"
