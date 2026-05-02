"""
Prepares driveindia_subset for Kaggle upload:
- Splits 5000 images into train (80%) / val (20%)
- Creates required folder structure for YOLO training
- Writes data.yaml with class names
"""
import os
import shutil
import glob
import random
import yaml

random.seed(42)

SRC = os.path.expanduser('~/mini_adas/driveindia_subset')
DST = os.path.expanduser('~/mini_adas/driveindia_kaggle')

# Clean and recreate
if os.path.exists(DST):
    shutil.rmtree(DST)
for split in ['train', 'val']:
    os.makedirs(f'{DST}/images/{split}', exist_ok=True)
    os.makedirs(f'{DST}/labels/{split}', exist_ok=True)

# Get all images, split 80/20
all_imgs = sorted(glob.glob(f'{SRC}/images/*.jpg'))
random.shuffle(all_imgs)
split_idx = int(len(all_imgs) * 0.8)
train_imgs = all_imgs[:split_idx]
val_imgs = all_imgs[split_idx:]

print(f"Train: {len(train_imgs)} images")
print(f"Val:   {len(val_imgs)} images")

for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
    for img in imgs:
        fname = os.path.basename(img)
        shutil.copy(img, f'{DST}/images/{split}/{fname}')
        lbl_src = img.replace('/images/', '/labels/').replace('.jpg', '.txt')
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, f'{DST}/labels/{split}/{os.path.basename(lbl_src)}')

# Class names - 28 classes total based on what's in the data
# Use known names from paper for 0-23, generic for 24-27
CLASS_NAMES = [
    'person', 'animal', 'bicycle', 'car', 'motorcycle', 'bus',
    'commercial_vehicle', 'truck', 'autorickshaw', 'ambulance',
    'police_vehicle', 'tractor', 'pushcart', 'construction_vehicle',
    'pothole', 'route_board', 'traffic_sign', 'traffic_light',
    'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
    'marked_bump', 'zebra_crossing',
    'cls_24', 'cls_25', 'cls_26', 'cls_27'
]

# Write data.yaml
yaml_data = {
    'path': '/kaggle/input/driveindia-subset',  # Kaggle dataset mount path
    'train': 'images/train',
    'val': 'images/val',
    'names': {i: name for i, name in enumerate(CLASS_NAMES)}
}

with open(f'{DST}/data.yaml', 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"\nKaggle-ready dataset at: {DST}")
print(f"Total size: ~1.5 GB")
print(f"Folder structure:")
print(f"  {DST}/")
print(f"    images/train/  ({len(train_imgs)} files)")
print(f"    images/val/    ({len(val_imgs)} files)")
print(f"    labels/train/")
print(f"    labels/val/")
print(f"    data.yaml")
