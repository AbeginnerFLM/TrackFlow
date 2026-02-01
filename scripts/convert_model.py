#!/usr/bin/env python3
"""
YOLO模型转换脚本
将 PyTorch (.pt) 模型转换为 ONNX 格式
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='转换YOLO模型为ONNX格式')
    parser.add_argument('input', type=str, help='输入的.pt模型路径')
    parser.add_argument('-o', '--output', type=str, help='输出的.onnx模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset版本 (默认: 17)')
    parser.add_argument('--simplify', action='store_true', help='简化ONNX模型')
    parser.add_argument('--half', action='store_true', help='使用FP16')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 模型文件不存在: {input_path}")
        sys.exit(1)
    
    output_path = args.output or input_path.with_suffix('.onnx')
    
    print(f"输入模型: {input_path}")
    print(f"输出模型: {output_path}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"ONNX opset: {args.opset}")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 需要安装 ultralytics 库")
        print("运行: pip install ultralytics")
        sys.exit(1)
    
    print("\n加载模型...")
    model = YOLO(str(input_path))
    
    print("导出为ONNX...")
    model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half
    )
    
    # 检查输出
    expected_output = input_path.with_suffix('.onnx')
    if expected_output.exists():
        if str(expected_output) != str(output_path):
            expected_output.rename(output_path)
        print(f"\n✓ 模型导出成功: {output_path}")
        print(f"  大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n✗ 导出失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
