import os
import numpy as np
import tensorflow as tf
tflite = tf.lite

MODELS_DIR = os.path.expanduser('~/mini_adas/models_extracted/models')
IMG_SIZE = 96

# Find all .tflite files
tflite_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.tflite')])
print(f"Found {len(tflite_files)} TFLite models")

# Generate one fake test image (random pixels in 0-1 float range)
np.random.seed(42)
fake_image = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

print(f"\n{'Model':<32} {'Input dtype':<10} {'Pred class':<10} {'Confidence':<10}")
print("-" * 65)

for fname in tflite_files:
    path = os.path.join(MODELS_DIR, fname)
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    
    # Match the input dtype
    if inp['dtype'] == np.int8:
        scale, zp = inp['quantization']
        sample = (fake_image / scale + zp).round().astype(np.int8) if scale != 0 else fake_image.astype(np.float32)
    elif inp['dtype'] == np.float16:
        sample = fake_image.astype(np.float16)
    else:
        sample = fake_image.astype(np.float32)
    
    interp.set_tensor(inp['index'], sample)
    interp.invoke()
    pred = interp.get_tensor(out['index'])[0]
    
    # Dequantize output if needed
    if out['dtype'] == np.int8:
        scale, zp = out['quantization']
        if scale != 0:
            pred = (pred.astype(np.float32) - zp) * scale
    
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))
    dtype_str = str(inp['dtype']).split("'")[1].split('.')[-1] if "'" in str(inp['dtype']) else str(inp['dtype'])
    print(f"{fname:<32} {dtype_str:<10} {cls:<10} {conf:<10.4f}")