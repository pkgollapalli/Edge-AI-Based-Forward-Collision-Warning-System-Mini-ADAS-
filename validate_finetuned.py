"""
Validate FINE-TUNED YOLOv8n (DriveIndia) on 2500 val images.
Compares per-class recall against COCO baseline.
"""
import os
import glob
import time
from collections import defaultdict
from ultralytics import YOLO

VAL_IMG = os.path.expanduser('~/mini_adas/driveindia_val/images_2500')
VAL_LBL = os.path.expanduser('~/mini_adas/driveindia_val/labels_2500')
MODEL_PT = os.path.expanduser('~/mini_adas/yolov8n_driveindia_best.pt')

CLASS_NAMES = [
    'person', 'animal', 'bicycle', 'car', 'motorcycle', 'bus',
    'commercial_vehicle', 'truck', 'autorickshaw', 'ambulance',
    'police_vehicle', 'tractor', 'pushcart', 'construction_vehicle',
    'pothole', 'route_board', 'traffic_sign', 'traffic_light',
    'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
    'marked_bump', 'zebra_crossing',
    'cls_24', 'cls_25', 'cls_26', 'cls_27'
]

print(f"Loading fine-tuned model from {MODEL_PT}...")
model = YOLO(MODEL_PT)
print(f"Model classes ({len(model.names)}): {list(model.names.values())[:8]}...")

all_imgs = sorted(glob.glob(f'{VAL_IMG}/*.jpg'))[:2500]
print(f"\nValidating on {len(all_imgs)} images\n")

# Track stats per class
gt_per_class = defaultdict(int)
correct_per_class = defaultdict(int)
images_with_gt_class = defaultdict(int)
total_lat = 0

for i, img_path in enumerate(all_imgs):
    fname = os.path.basename(img_path).replace('.jpg', '.txt')
    lbl_path = os.path.join(VAL_LBL, fname)
    
    # Load ground truth class IDs in this image
    gt_classes = set()
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(CLASS_NAMES):
                        gt_classes.add(cls_id)
                        gt_per_class[CLASS_NAMES[cls_id]] += 1
    for c in gt_classes:
        images_with_gt_class[CLASS_NAMES[c]] += 1
    
    # Predict
    t0 = time.time()
    results = model(img_path, verbose=False, conf=0.3)[0]
    total_lat += time.time() - t0
    
    pred_classes = set()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id < len(CLASS_NAMES):
            pred_classes.add(cls_id)
    
    # Did we correctly detect each ground-truth class?
    for cls_id in gt_classes:
        if cls_id in pred_classes:
            correct_per_class[CLASS_NAMES[cls_id]] += 1
    
    if (i + 1) % 250 == 0:
        print(f"  Processed {i+1}/{len(all_imgs)}...")

# Print results
print(f"\n--- Fine-tuned model results on {len(all_imgs)} DriveIndia images ---")
print(f"Avg latency: {total_lat*1000/len(all_imgs):.0f} ms/image (laptop CPU)\n")

print(f"{'Class':<25} {'GT images':>10} {'Recall':>10}")
print("-" * 50)

for cls_name in CLASS_NAMES:
    gt = images_with_gt_class.get(cls_name, 0)
    correct = correct_per_class.get(cls_name, 0)
    if gt > 0:
        recall = correct / gt * 100
        print(f"{cls_name:<25} {gt:>10} {recall:>9.1f}%")

# Save results to file
import json
results_dict = {
    'model': 'fine_tuned_driveindia',
    'images_tested': len(all_imgs),
    'avg_latency_ms': total_lat*1000/len(all_imgs),
    'per_class': {
        cls: {
            'gt_images': images_with_gt_class.get(cls, 0),
            'correct': correct_per_class.get(cls, 0),
            'recall_pct': correct_per_class.get(cls, 0) / max(images_with_gt_class.get(cls, 1), 1) * 100
        }
        for cls in CLASS_NAMES if images_with_gt_class.get(cls, 0) > 0
    }
}
with open(os.path.expanduser('~/mini_adas/finetuned_results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)
print("\nSaved to finetuned_results.json")
