"""
Compress the fine-tuned DriveIndia YOLO model to TFLite (FP32 / FP16 / INT8).
Outputs go to ~/mini_adas/yolov8n_driveindia_saved_model/
"""
import os
from ultralytics import YOLO

PT_PATH = os.path.expanduser('~/mini_adas/yolov8n_driveindia_best.pt')

print("Exporting fine-tuned model: FP32, FP16, INT8 to TFLite...\n")

# FP32
print("[1/3] FP32...")
model = YOLO(PT_PATH)
model.export(format='tflite', imgsz=320)

# FP16
print("\n[2/3] FP16...")
model = YOLO(PT_PATH)
model.export(format='tflite', imgsz=320, half=True)

# INT8
print("\n[3/3] INT8...")
model = YOLO(PT_PATH)
model.export(format='tflite', imgsz=320, int8=True)

# Show files
import glob
print("\n--- Generated files ---")
for f in sorted(glob.glob('yolov8n_driveindia_best_saved_model/*.tflite')):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f}: {size_mb:.2f} MB")

