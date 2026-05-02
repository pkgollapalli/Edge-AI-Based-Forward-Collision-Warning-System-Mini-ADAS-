import os, glob, time
import numpy as np
from ultralytics import YOLO

YOLO_DIR = os.path.expanduser('~/mini_adas/pothole_saved_model')
PT_PATH = os.path.expanduser('~/mini_adas/weights/best.pt')

print("Loading pothole .pt model...")
model = YOLO(PT_PATH)

print("\n--- Exporting pothole to TFLite FP32 ---")
model.export(format='tflite', imgsz=320)

print("\n--- Re-loading and exporting FP16 ---")
model = YOLO(PT_PATH)
model.export(format='tflite', imgsz=320, half=True)

print("\n--- Re-loading and exporting INT8 ---")
model = YOLO(PT_PATH)
model.export(format='tflite', imgsz=320, int8=True)

print("\n--- Generated files ---")
import os
saved_dir = os.path.dirname(PT_PATH).replace('weights', 'best_saved_model')
# Ultralytics outputs go to <model_name>_saved_model/ in CWD
for d in glob.glob('best_saved_model'):
    for f in sorted(glob.glob(f'{d}/*.tflite')):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f}: {size_mb:.2f} MB")

# Also list current dir
for f in sorted(glob.glob('best_saved_model/*.tflite')):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {f}: {size_mb:.2f} MB")