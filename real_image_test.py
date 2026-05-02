import os
import numpy as np
import tensorflow as tf
from PIL import Image
import glob

tflite = tf.lite

MODELS_DIR = os.path.expanduser('~/mini_adas/models_extracted/models')
TEST_DIR = os.path.expanduser('~/mini_adas/test_images')
IMG_SIZE = 96

# Load all images
def load_images(folder, true_label):
    items = []
    for path in sorted(glob.glob(os.path.join(folder, '*.jpg'))):
        img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        items.append((os.path.basename(path), arr, true_label))
    return items

veh_imgs = load_images(os.path.join(TEST_DIR, 'vehicles'), 1)
non_imgs = load_images(os.path.join(TEST_DIR, 'non_vehicles'), 0)
all_imgs = veh_imgs + non_imgs
print(f"Loaded {len(veh_imgs)} vehicles + {len(non_imgs)} non-vehicles = {len(all_imgs)} total\n")

# Test each model
models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.tflite')])

print(f"{'Model':<28} {'Acc':>6} {'V_recall':>10} {'N_recall':>10}")
print("-" * 56)

results = {}

for fname in models:
    path = os.path.join(MODELS_DIR, fname)
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    correct = 0
    veh_correct = 0
    non_correct = 0
    veh_total = 0
    non_total = 0
    per_image = []
    
    for name, arr, true in all_imgs:
        sample = arr[np.newaxis, ...]
        if inp['dtype'] == np.int8:
            scale, zp = inp['quantization']
            if scale != 0:
                sample = (sample / scale + zp).round().astype(np.int8)
            else:
                sample = sample.astype(np.float32)
        elif inp['dtype'] == np.float16:
            sample = sample.astype(np.float16)
        else:
            sample = sample.astype(np.float32)
        
        interp.set_tensor(inp['index'], sample)
        interp.invoke()
        pred = interp.get_tensor(out['index'])[0]
        
        if out['dtype'] == np.int8:
            scale, zp = out['quantization']
            if scale != 0:
                pred = (pred.astype(np.float32) - zp) * scale
        
        pred_class = int(np.argmax(pred))
        per_image.append((name, true, pred_class, float(pred[1])))
        
        if pred_class == true:
            correct += 1
        if true == 1:
            veh_total += 1
            if pred_class == 1: veh_correct += 1
        else:
            non_total += 1
            if pred_class == 0: non_correct += 1
    
    acc = correct / len(all_imgs) * 100
    v_recall = veh_correct / veh_total * 100 if veh_total else 0
    n_recall = non_correct / non_total * 100 if non_total else 0
    
    print(f"{fname:<28} {acc:>5.1f}% {v_recall:>9.1f}% {n_recall:>9.1f}%")
    results[fname] = per_image

# Show detailed predictions for the winning model
print("\n--- Per-image breakdown (m8_student_int8) ---")
print(f"{'Image':<15} {'True':>5} {'Pred':>5} {'P(vehicle)':>12} {'Result':>8}")
for name, true, pred, p_veh in results['m8_student_int8.tflite']:
    result = "OK" if pred == true else "WRONG"
    label = "vehicle" if true == 1 else "non_veh"
    print(f"{name:<15} {label:>5} {pred:>5} {p_veh:>12.4f} {result:>8}")
