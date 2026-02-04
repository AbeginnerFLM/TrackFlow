"""
正确的YOLO ONNX导出脚本

用法：
    python export_onnx.py --model your_model.pt --imgsz 640

这个脚本确保ONNX模型的坐标输出与原始PyTorch模型一致
"""

from ultralytics import YOLO
import argparse

def export_onnx(model_path, imgsz=640):
    """
    导出YOLO模型为ONNX格式
    
    Args:
        model_path: .pt模型文件路径
        imgsz: 输入图像尺寸
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Exporting to ONNX with imgsz={imgsz}...")
    
    # 关键参数说明：
    # - format="onnx": 导出格式
    # - imgsz: 输入尺寸，必须与推理时一致
    # - half=False: 不使用FP16，保证数值精度
    # - simplify=True: 简化ONNX图结构
    # - opset=12: ONNX算子集版本
    # - dynamic=False: 不使用动态batch（更稳定）
    
    success = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,
        simplify=True,
        opset=12,
        dynamic=False,  # 固定batch size
    )
    
    if success:
        onnx_path = model_path.replace('.pt', '.onnx')
        print(f"✓ Export successful: {onnx_path}")
        print("\n重要提示:")
        print("1. 确保C++推理时输入尺寸也是", imgsz)
        print("2. 预处理必须使用letterbox（保持宽高比）")
        print("3. 后处理坐标变换公式: (coord - padding) / scale")
        return onnx_path
    else:
        print("✗ Export failed")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    
    args = parser.parse_args()
    export_onnx(args.model, args.imgsz)
