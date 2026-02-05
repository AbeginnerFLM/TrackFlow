#!/bin/bash
# GPU 环境诊断脚本
# 用于在 GPU 服务器上诊断 CUDA/cuDNN 问题

echo "=========================================="
echo "  TrackFlow GPU 环境诊断"
echo "=========================================="
echo ""

# 1. 检查 NVIDIA 驱动
echo "[1/5] 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    echo "✓ NVIDIA 驱动已安装"
else
    echo "✗ nvidia-smi 不可用，请检查驱动安装"
fi
echo ""

# 2. 检查 CUDA 版本
echo "[2/5] 检查 CUDA Toolkit..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo "✓ CUDA Toolkit 已安装"
else
    echo "⚠ nvcc 不在 PATH 中，检查常见路径..."
    if [ -f /usr/local/cuda/bin/nvcc ]; then
        /usr/local/cuda/bin/nvcc --version | grep "release"
        echo "CUDA 路径: /usr/local/cuda"
    else
        echo "✗ CUDA Toolkit 未找到"
    fi
fi
echo ""

# 3. 检查 cuDNN
echo "[3/5] 检查 cuDNN..."
CUDNN_H_PATHS=(
    "/usr/include/cudnn.h"
    "/usr/include/cudnn_version.h"
    "/usr/local/cuda/include/cudnn.h"
    "/usr/local/cuda/include/cudnn_version.h"
)
CUDNN_FOUND=0
for path in "${CUDNN_H_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "找到 cuDNN 头文件: $path"
        grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" "$path" 2>/dev/null | head -3
        CUDNN_FOUND=1
        break
    fi
done
if [ $CUDNN_FOUND -eq 0 ]; then
    echo "✗ cuDNN 头文件未找到"
fi
echo ""

# 4. 检查 ONNX Runtime CUDA 库
echo "[4/5] 检查 ONNX Runtime CUDA 库..."
ORT_LIB="$HOME/TrackFlow/third_party/onnxruntime/lib"
if [ -d "$ORT_LIB" ]; then
    echo "ONNX Runtime 库目录: $ORT_LIB"
    ls -lh "$ORT_LIB"/*.so 2>/dev/null || echo "没有找到 .so 文件"
    if [ -f "$ORT_LIB/libonnxruntime_providers_cuda.so" ]; then
        echo "✓ CUDA provider 库存在"
    else
        echo "✗ CUDA provider 库不存在"
    fi
else
    echo "✗ ONNX Runtime 目录不存在: $ORT_LIB"
fi
echo ""

# 5. 简单 CUDA 测试
echo "[5/5] 测试 CUDA 可访问性..."
python3 -c "
import os
try:
    # 首先尝试 PyTorch
    import torch
    if torch.cuda.is_available():
        print(f'✓ PyTorch CUDA 可用')
        print(f'  设备: {torch.cuda.get_device_name(0)}')
        print(f'  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('✗ PyTorch CUDA 不可用')
except ImportError:
    print('⚠ PyTorch 未安装，跳过')

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f'ONNX Runtime providers: {providers}')
    if 'CUDAExecutionProvider' in providers:
        print('✓ ONNX Runtime CUDA 可用')
    else:
        print('✗ CUDAExecutionProvider 不在可用列表中')
except ImportError:
    print('⚠ onnxruntime 未安装，跳过')
except Exception as e:
    print(f'✗ ONNX Runtime 错误: {e}')
" 2>&1

echo ""
echo "=========================================="
echo "  诊断完成"
echo "=========================================="
echo ""
echo "如果 CUDA 不可用，可能的原因："
echo "  1. WSL2 需要 Windows 端的 NVIDIA GPU 驱动支持"
echo "  2. 需要安装 CUDA Toolkit 和 cuDNN"
echo "  3. 环境变量 LD_LIBRARY_PATH 需要包含 CUDA 库路径"
echo ""
echo "建议操作："
echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
