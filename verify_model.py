import cv2
import os
import sys
import numpy as np

model_path = "models/yolo26.onnx"
if not os.path.exists(model_path):
    model_path = "build/models/yolo26.onnx"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

print(f"Loading model from {model_path} using OpenCV DNN...")
try:
    net = cv2.dnn.readNetFromONNX(model_path)
    print("Model loaded successfully!")
    
    layer_names = net.getLayerNames()
    print(f"Total layers: {len(layer_names)}")
    
    out_layers = net.getUnconnectedOutLayersNames()
    print(f"Output layers: {out_layers}")
    
    # Try a dummy inference to see output shape
    # YOLO input is usually 640x640
    blob = cv2.dnn.blobFromImage(np.zeros((640, 640, 3), np.uint8), 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    
    for i, out in enumerate(outs):
        print(f"Output {i} ({out_layers[i]}) shape: {out.shape}")
        
    print("Verification complete.")

except Exception as e:
    print(f"Failed to load/run model: {e}")
    sys.exit(1)
