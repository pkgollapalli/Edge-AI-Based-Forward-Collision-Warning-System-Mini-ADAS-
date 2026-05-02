import os, glob, time
import numpy as np
import tensorflow as tf
from PIL import Image

tflite = tf.lite

POTHOLE_DIR = os.path.expanduser('~/mini_adas/weights/best_saved_model')
TEST_DIR = os.path.expanduser('~/mini_adas/test_images')

MODELS = [
    ('best_float32.tflite', 'FP32'),
    ('best_float16.tflite', 'FP16'),
    ('best_int8.tflite', 'INT8'),
]

IMG_SIZE = 320
CONF_THRESH = 0.30

def preprocess(img_path, dtype):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    if dtype == np.int8:
        return (arr * 255 - 128).astype(np.int8)
    if dtype == np.float16:
        return arr.astype(np.float16)
    return arr.astype(np.float32)

def parse_pothole_output(output, conf_thresh):
    """Pothole model: [1, 5, N] = 4 box + 1 class score."""
    if output.ndim == 3 and output.shape[1] == 5:
        out = output[0].T
    else:
        out = output.reshape(output.shape[0], -1, 5)[0]
    scores = out[:, 4]
    return [float(s) for s in scores if s > conf_thresh]

# Test images: mix of vehicles, non-vehicles, potholes
test_imgs = (sorted(glob.glob(os.path.join(TEST_DIR, 'vehicles/*.jpg'))) +
             sorted(glob.glob(os.path.join(TEST_DIR, 'non_vehicles/*.jpg'))) +
             sorted(glob.glob(os.path.join(TEST_DIR, 'potholes/*.jpg'))))
print(f"Loaded {len(test_imgs)} test images\n")

print(f"{'Model':<8} {'Size MB':>8} {'Avg lat ms':>12} {'Pothole hits':>14} {'Sample (potholes folder)':<35}")
print("-" * 85)

results = []
for fname, label in MODELS:
    path = os.path.join(POTHOLE_DIR, fname)
    size_mb = os.path.getsize(path) / 1024 / 1024
    
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    # Warmup
    sample = preprocess(test_imgs[0], inp['dtype'])
    interp.set_tensor(inp['index'], sample)
    interp.invoke()
    
    latencies = []
    pothole_hits_total = 0
    pothole_hits_in_pothole_folder = 0
    sample_str = ''
    
    for img_path in test_imgs:
        sample = preprocess(img_path, inp['dtype'])
        t0 = time.time()
        interp.set_tensor(inp['index'], sample)
        interp.invoke()
        raw = interp.get_tensor(out['index'])
        latencies.append((time.time() - t0) * 1000)
        
        if out['dtype'] == np.int8:
            scale, zp = out['quantization']
            if scale != 0:
                raw = (raw.astype(np.float32) - zp) * scale
        
        scores = parse_pothole_output(raw, CONF_THRESH)
        if scores:
            pothole_hits_total += 1
            if 'potholes' in img_path:
                pothole_hits_in_pothole_folder += 1
                if not sample_str and 'download (2)' in img_path:
                    sample_str = ', '.join(f'{s:.2f}' for s in scores[:3])
    
    avg_lat = float(np.median(latencies))
    n_pothole_imgs = len(glob.glob(os.path.join(TEST_DIR, 'potholes/*.jpg')))
    print(f"{label:<8} {size_mb:>8.2f} {avg_lat:>12.1f} {pothole_hits_in_pothole_folder}/{n_pothole_imgs:<3}     "
          f"{sample_str:<35}")
    results.append({
        'model': label,
        'size_mb': size_mb,
        'lat_ms': avg_lat,
        'recall_potholes': pothole_hits_in_pothole_folder / n_pothole_imgs,
        'total_hits': pothole_hits_total,
    })

import json
with open(os.path.expanduser('~/mini_adas/pothole_compression_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to pothole_compression_results.json")