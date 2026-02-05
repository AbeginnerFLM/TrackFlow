#!/bin/bash
# Check CUDA library dependencies
echo "=== ONNX Runtime CUDA Provider Dependencies ==="
ldd ~/TrackFlow/third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so 2>&1 | grep -E "cuda|cudnn|not found" || echo "No cuda/cudnn/missing libs found"

echo ""
echo "=== Check for missing libraries ==="
ldd ~/TrackFlow/third_party/onnxruntime/lib/libonnxruntime_providers_cuda.so 2>&1 | grep "not found" || echo "No missing libraries"

echo ""
echo "=== Test CUDA shared library loading ==="
LD_LIBRARY_PATH=~/TrackFlow/third_party/onnxruntime/lib:$LD_LIBRARY_PATH \
    ldd ~/TrackFlow/build/yolo_edge_server 2>&1 | grep -E "cuda|cudnn|onnx|not found" || echo "No cuda/cudnn/onnx/missing libs found"
