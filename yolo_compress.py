import os
import time
import numpy as np
from ultralytics import YOLO

# Make a folder for YOLO models
YOLO_DIR = os.path.expanduser('~/mini_adas/yolo_models')
os.makedirs(YOLO_DIR, exist_ok=True)

# Load the FP32 PyTorch model
print("Loading YOLOv8n (PyTorch)...")
model = YOLO('yolov8n.pt')

# Export to multiple formats - Ultralytics handles the conversion
print("\n--- Exporting to TFLite FP32 ---")
model.export(format='tflite', imgsz=320)  # 320x320 input - good for Pi speed

print("\n--- Exporting to TFLite FP16 ---")
model.export(format='tflite', imgsz=320, half=True)

print("\n--- Exporting to TFLite INT8 ---")
model.export(format='tflite', imgsz=320, int8=True)

# Show what we got
print("\n--- Generated TFLite files ---")
import glob
for f in sorted(glob.glob('yolov8n_saved_model/*.tflite')):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f}: {size_mb:.2f} MB")
