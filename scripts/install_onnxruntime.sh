#!/bin/bash
# ONNX Runtime GPU 安装脚本
# 适用于 CUDA 12.x

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"

# ONNX Runtime 版本 (支持CUDA 12)
ORT_VERSION="1.17.0"
ORT_PACKAGE="onnxruntime-linux-x64-gpu-${ORT_VERSION}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PACKAGE}.tgz"

echo "=========================================="
echo "  安装 ONNX Runtime GPU ${ORT_VERSION}"
echo "=========================================="

mkdir -p "$THIRD_PARTY_DIR"
cd "$THIRD_PARTY_DIR"

# 下载
if [ ! -f "${ORT_PACKAGE}.tgz" ]; then
    echo ""
    echo "[1/3] 下载 ONNX Runtime..."
    wget -q --show-progress "$ORT_URL"
else
    echo "[1/3] 已存在,跳过下载"
fi

# 解压
echo ""
echo "[2/3] 解压..."
tar -xzf "${ORT_PACKAGE}.tgz"

# 创建符号链接
echo ""
echo "[3/3] 创建符号链接..."
rm -f onnxruntime
ln -sf "$ORT_PACKAGE" onnxruntime

echo ""
echo "=========================================="
echo "  ONNX Runtime 安装完成!"
echo "=========================================="
echo ""
echo "安装路径: $THIRD_PARTY_DIR/onnxruntime"
echo ""
echo "如需系统级安装,执行:"
echo "  sudo cp -r $THIRD_PARTY_DIR/onnxruntime/lib/* /usr/local/lib/"
echo "  sudo cp -r $THIRD_PARTY_DIR/onnxruntime/include/* /usr/local/include/"
echo "  sudo ldconfig"
