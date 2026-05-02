import os
import glob
from ultralytics import YOLO
import time

# YOLOv8n: nano version - smallest, fastest YOLOv8 (~6 MB)
print("Loading YOLOv8n (will auto-download on first run, ~6 MB)...")
model = YOLO('yolov8n.pt')

# All 80 COCO classes the model knows
print(f"\nClasses model recognizes ({len(model.names)} total):")
print(list(model.names.values()))

# Danger classes for ADAS
DANGER_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'traffic light', 'stop sign'
}
print(f"\nDanger classes for ADAS: {sorted(DANGER_CLASSES)}")

# Test on our existing images
print("\n" + "="*70)
print("Testing on real images we downloaded earlier:")
print("="*70)

all_imgs = sorted(glob.glob('test_images/vehicles/*.jpg')) + \
           sorted(glob.glob('test_images/non_vehicles/*.jpg'))

for img_path in all_imgs:
    t0 = time.time()
    results = model(img_path, verbose=False)[0]
    elapsed_ms = (time.time() - t0) * 1000
    
    detected = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        if conf > 0.3:
            detected.append(f"{cls_name}({conf:.2f})")
    
    danger_present = any(d.split('(')[0] in DANGER_CLASSES for d in detected)
    flag = "DANGER" if danger_present else "safe"
    
    fname = os.path.basename(img_path)
    print(f"{fname:<15} [{elapsed_ms:6.1f} ms] {flag:<7} -> {detected if detected else '(nothing)'}")
