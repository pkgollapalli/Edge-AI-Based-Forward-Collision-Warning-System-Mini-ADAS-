import os
import time
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

tflite = tf.lite

YOLO_DIR = os.path.expanduser('~/mini_adas/yolov8n_saved_model')
TEST_DIR = os.path.expanduser('~/mini_adas/test_images')

# Load class names from the original YOLO
from ultralytics import YOLO
ref = YOLO('yolov8n.pt')
NAMES = ref.names

DANGER = {'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
          'cat', 'dog', 'horse', 'sheep', 'cow', 'traffic light', 'stop sign'}

# Models to test (size order)
MODELS = [
    ('yolov8n_float32.tflite', 'FP32'),
    ('yolov8n_float16.tflite', 'FP16'),
    ('yolov8n_int8.tflite', 'INT8'),
]

# Load all test images
all_imgs = sorted(glob.glob(os.path.join(TEST_DIR, 'vehicles/*.jpg'))) + \
           sorted(glob.glob(os.path.join(TEST_DIR, 'non_vehicles/*.jpg')))
print(f"Loaded {len(all_imgs)} test images\n")

IMG_SIZE = 320
CONF_THRESH = 0.3

def preprocess(img_path, dtype):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    if dtype == np.int8:
        return (arr * 255 - 128).astype(np.int8)
    if dtype == np.uint8:
        return (arr * 255).astype(np.uint8)
    if dtype == np.float16:
        return arr.astype(np.float16)
    return arr.astype(np.float32)

def parse_yolo_output(output, conf_thresh, in_dtype):
    """YOLOv8 output: [1, 84, N] where 84 = 4 box + 80 classes."""
    if output.ndim == 3 and output.shape[1] == 84:
        out = output[0].T  # [N, 84]
    else:
        out = output.reshape(output.shape[0], -1, 84)[0]
    
    boxes = out[:, :4]
    class_scores = out[:, 4:]
    max_scores = class_scores.max(axis=1)
    cls_ids = class_scores.argmax(axis=1)
    
    detections = []
    for i in np.where(max_scores > conf_thresh)[0]:
        detections.append((NAMES[int(cls_ids[i])], float(max_scores[i])))
    return detections

print(f"{'Model':<8} {'Size MB':>8} {'Avg lat ms':>12} {'Detections (sample)':>30}")
print("-" * 65)

results = []
for fname, label in MODELS:
    path = os.path.join(YOLO_DIR, fname)
    size_mb = os.path.getsize(path) / 1024 / 1024
    
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    # Warmup
    sample = preprocess(all_imgs[0], inp['dtype'])
    interp.set_tensor(inp['index'], sample)
    interp.invoke()
    
    # Time across all images
    latencies = []
    danger_counts = 0
    sample_dets = []
    for img_path in all_imgs:
        sample = preprocess(img_path, inp['dtype'])
        t0 = time.time()
        interp.set_tensor(inp['index'], sample)
        interp.invoke()
        raw_out = interp.get_tensor(out['index'])
        latencies.append((time.time() - t0) * 1000)
        
        # Dequantize if needed
        if out['dtype'] == np.int8:
            scale, zp = out['quantization']
            if scale != 0:
                raw_out = (raw_out.astype(np.float32) - zp) * scale
        
        dets = parse_yolo_output(raw_out, CONF_THRESH, inp['dtype'])
        if any(d[0] in DANGER for d in dets):
            danger_counts += 1
        if 'v01' in img_path:
            sample_dets = dets[:3]  # Save v01 detections as a sanity sample
    
    avg_lat = float(np.median(latencies))
    sample_str = ', '.join(f'{n}({c:.2f})' for n, c in sample_dets[:3]) or '(none)'
    print(f"{label:<8} {size_mb:>8.2f} {avg_lat:>12.1f} {sample_str:>30}")
    results.append({'model': label, 'size_mb': size_mb, 'lat_ms': avg_lat, 'danger_flagged': danger_counts})

print(f"\nDanger-class detection across {len(all_imgs)} images:")
for r in results:
    print(f"  {r['model']}: flagged danger in {r['danger_flagged']}/{len(all_imgs)} images")

import json
with open(os.path.expanduser('~/mini_adas/yolo_compression_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to yolo_compression_results.json")