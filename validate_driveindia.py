"""
Validate YOLOv8n COCO + Pothole models on DriveIndia validation set.
Compares predictions against ground-truth labels.
"""
import os
import glob
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image

VAL_IMG = os.path.expanduser('~/mini_adas/driveindia_val/images_2500')
VAL_LBL = os.path.expanduser('~/mini_adas/driveindia_val/labels_2500')

# DriveIndia 24-class list (order from paper)
DRIVEINDIA_NAMES = [
    'person', 'animal', 'bicycle', 'car', 'motorcycle', 'bus',
    'commercial_vehicle', 'truck', 'autorickshaw', 'ambulance',
    'police_vehicle', 'tractor', 'pushcart', 'construction_vehicle',
    'pothole', 'route_board', 'traffic_sign', 'traffic_light',
    'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
    'marked_bump', 'zebra_crossing'
]

# Map DriveIndia classes that COCO YOLOv8n can also detect
# COCO names (some): person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7),
#                    traffic light(9), stop sign(11)
DRIVE_TO_COCO = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorcycle': 'motorcycle',
    'bus': 'bus',
    'truck': 'truck',
    'traffic_light': 'traffic light',
    # autorickshaw → COCO has no equivalent (key gap!)
    # pothole → handled by separate pothole model
}

# How many images to test (start small, expand if fast)
N_IMAGES = 2500

# Pick a sample
all_imgs = sorted(glob.glob(f'{VAL_IMG}/*.jpg'))
sample = all_imgs[:N_IMAGES]
print(f"Validating on {len(sample)} of {len(all_imgs)} DriveIndia images\n")

# Load YOLOv8n COCO + pothole
print("Loading models...")
yolo = YOLO('yolov8n.pt')
pothole = YOLO(os.path.expanduser('~/mini_adas/weights/best.pt'))
print()

# Stats
gt_class_count = defaultdict(int)        # ground truth occurrences
pred_class_count = defaultdict(int)      # predicted occurrences
correct_per_class = defaultdict(int)     # images where correctly detected
images_with_gt_class = defaultdict(int)  # images that have at least one gt of class

total_lat = 0
for i, img_path in enumerate(sample):
    # Load ground truth labels
    fname = os.path.basename(img_path).replace('.jpg', '.txt')
    lbl_path = os.path.join(VAL_LBL, fname)
    gt_classes = set()
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(DRIVEINDIA_NAMES):
                        gt_classes.add(DRIVEINDIA_NAMES[cls_id])
                        gt_class_count[DRIVEINDIA_NAMES[cls_id]] += 1

    for cls in gt_classes:
        images_with_gt_class[cls] += 1

    # Run YOLO COCO
    t0 = time.time()
    yolo_res = yolo(img_path, verbose=False, conf=0.3)[0]
    pothole_res = pothole(img_path, verbose=False, conf=0.4)[0]
    total_lat += time.time() - t0

    pred_classes_coco = set()
    for box in yolo_res.boxes:
        coco_name = yolo.names[int(box.cls[0])]
        pred_classes_coco.add(coco_name)
        pred_class_count[coco_name] += 1

    pothole_present = len(pothole_res.boxes) > 0
    if pothole_present:
        pred_class_count['pothole_pred'] += 1

    # Cross-check: did our model detect classes that exist in ground truth?
    for drive_class, coco_class in DRIVE_TO_COCO.items():
        if drive_class in gt_classes and coco_class in pred_classes_coco:
            correct_per_class[drive_class] += 1
    if 'pothole' in gt_classes and pothole_present:
        correct_per_class['pothole'] += 1

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(sample)}...")

print(f"\n--- Validation results on {len(sample)} DriveIndia images ---\n")
print(f"Avg latency: {total_lat*1000/len(sample):.0f} ms/image (laptop CPU)\n")

print(f"{'DriveIndia class':<25} {'GT images':>10} {'Recall (our model)':>20}")
print("-" * 60)
for drive_class in list(DRIVE_TO_COCO.keys()) + ['pothole']:
    gt_imgs = images_with_gt_class.get(drive_class, 0)
    correct = correct_per_class.get(drive_class, 0)
    recall = correct / gt_imgs * 100 if gt_imgs > 0 else 0
    print(f"{drive_class:<25} {gt_imgs:>10} {recall:>18.1f}%")

print(f"\n--- Indian-only classes our model CANNOT detect ---")
indian_only = ['autorickshaw', 'commercial_vehicle', 'pushcart', 'tractor',
               'construction_vehicle', 'route_board', 'traffic_sign',
               'temp_barrier', 'traffic_cone', 'rumblestrips',
               'unmarked_bump', 'marked_bump', 'zebra_crossing',
               'animal', 'ambulance', 'police_vehicle']
for c in indian_only:
    gt_imgs = images_with_gt_class.get(c, 0)
    print(f"  {c:<25} {gt_imgs:>4} GT images")
