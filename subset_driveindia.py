"""
Pick a balanced 5000-image subset of DriveIndia train1.
Prioritizes images containing rare Indian-specific classes.
"""
import os
import shutil
import glob
import random
from collections import defaultdict

random.seed(42)

SRC_DIR = os.path.expanduser('~/mini_adas/driveindia_train1/train1')
DST_DIR = os.path.expanduser('~/mini_adas/driveindia_subset')
SUBSET_SIZE = 5000

DRIVEINDIA_NAMES = [
    'person', 'animal', 'bicycle', 'car', 'motorcycle', 'bus',
    'commercial_vehicle', 'truck', 'autorickshaw', 'ambulance',
    'police_vehicle', 'tractor', 'pushcart', 'construction_vehicle',
    'pothole', 'route_board', 'traffic_sign', 'traffic_light',
    'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
    'marked_bump', 'zebra_crossing'
]

# Rare classes we want to over-represent
RARE_CLASSES = {1, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23}  # animal, autorickshaw, ambulance, etc.

# Index all images by which classes they contain
print("Indexing labels...")
all_imgs = sorted(glob.glob(f'{SRC_DIR}/images/*.jpg'))
print(f"Total images: {len(all_imgs)}")

img_classes = {}  # img_path -> set of class IDs
class_to_imgs = defaultdict(list)  # class_id -> list of img_paths

for img_path in all_imgs:
    fname = os.path.basename(img_path).replace('.jpg', '.txt')
    lbl_path = f'{SRC_DIR}/labels/{fname}'
    if not os.path.exists(lbl_path):
        continue
    classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    img_classes[img_path] = classes
    for c in classes:
        class_to_imgs[c].append(img_path)

print(f"\nClass distribution in train1:")
for c_id, imgs in sorted(class_to_imgs.items()):
    name = DRIVEINDIA_NAMES[c_id] if c_id < len(DRIVEINDIA_NAMES) else f'cls_{c_id}'
    print(f"  {c_id:>2} {name:<25} {len(imgs):>6} images")

# Selection: take ALL images with rare classes + random sample from rest
selected = set()
for rare_id in RARE_CLASSES:
    for img in class_to_imgs.get(rare_id, []):
        selected.add(img)
print(f"\n{len(selected)} images selected for rare class coverage")

remaining = [img for img in all_imgs if img not in selected]
random.shuffle(remaining)
needed = SUBSET_SIZE - len(selected)
if needed > 0:
    selected.update(remaining[:needed])
selected = list(selected)[:SUBSET_SIZE]
random.shuffle(selected)

print(f"Final subset size: {len(selected)}")

# Copy
os.makedirs(f'{DST_DIR}/images', exist_ok=True)
os.makedirs(f'{DST_DIR}/labels', exist_ok=True)
print(f"\nCopying to {DST_DIR}...")
for i, img_path in enumerate(selected):
    fname = os.path.basename(img_path)
    shutil.copy(img_path, f'{DST_DIR}/images/{fname}')
    lbl_fname = fname.replace('.jpg', '.txt')
    src_lbl = f'{SRC_DIR}/labels/{lbl_fname}'
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, f'{DST_DIR}/labels/{lbl_fname}')
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(selected)}")

# Final stats
final_class_count = defaultdict(int)
for img in selected:
    for c in img_classes.get(img, set()):
        final_class_count[c] += 1

print("\nFinal subset class distribution:")
for c_id in sorted(final_class_count.keys()):
    name = DRIVEINDIA_NAMES[c_id] if c_id < len(DRIVEINDIA_NAMES) else f'cls_{c_id}'
    print(f"  {c_id:>2} {name:<25} {final_class_count[c_id]:>5} images")

print(f"\nDone! Subset at: {DST_DIR}")
print(f"Size: ~{int(SUBSET_SIZE * 0.3)} MB")
