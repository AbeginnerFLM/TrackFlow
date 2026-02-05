#!/usr/bin/env python3
"""Test ONNX Runtime CUDA availability"""
import sys
import os

# Add ONNX Runtime lib path
ort_lib = os.path.expanduser("~/TrackFlow/third_party/onnxruntime/lib")
if os.path.exists(ort_lib):
    sys.path.insert(0, ort_lib)
    os.environ["LD_LIBRARY_PATH"] = ort_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if "CUDAExecutionProvider" in providers:
        print("✓ CUDA provider is AVAILABLE")
        
        # Try to create a session with CUDA
        model_path = os.path.expanduser("~/TrackFlow/models/yolo26.onnx")
        if os.path.exists(model_path):
            try:
                sess_options = ort.SessionOptions()
                sess = ort.InferenceSession(
                    model_path,
                    sess_options,
                    providers=["CUDAExecutionProvider"]
                )
                print("✓ Successfully created CUDA session")
                print(f"  Model: {model_path}")
            except Exception as e:
                print(f"✗ Failed to create CUDA session: {e}")
        else:
            print(f"⚠ Model not found: {model_path}")
    else:
        print("✗ CUDA provider NOT available")
        print("  This is likely because CUDA/cuDNN is not properly installed")
        
except ImportError as e:
    print(f"✗ Cannot import onnxruntime: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
