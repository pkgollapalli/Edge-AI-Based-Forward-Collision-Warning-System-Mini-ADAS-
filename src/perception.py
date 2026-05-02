"""
Perception module for Mini ADAS - 3-model ensemble.
1. YOLOv8n COCO INT8 (general: car, person, motorcycle, truck, bus, etc.)
2. YOLOv8n DriveIndia INT8 (Indian: autorickshaw, police_vehicle, etc.)
3. Pothole YOLOv8s INT8 (potholes only)
Union of detections from all three. Best coverage.
"""
import os
import time
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

YOLO_COCO_PATH = os.path.expanduser('~/mini_adas/yolov8n_saved_model/yolov8n_int8.tflite')
YOLO_INDIA_PATH = os.path.expanduser('~/mini_adas/yolov8n_driveindia_best_saved_model/yolov8n_driveindia_best_int8.tflite')
POTHOLE_PATH = os.path.expanduser('~/mini_adas/weights/best_saved_model/best_int8.tflite')

IMG_SIZE = 320
CONF_THRESH_COCO = 0.30
CONF_THRESH_INDIA = 0.45  # higher to suppress bicycle bias
POTHOLE_CONF_THRESH = 0.40

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

INDIA_NAMES = [
    'person', 'animal', 'bicycle', 'car', 'motorcycle', 'bus',
    'commercial_vehicle', 'truck', 'autorickshaw', 'ambulance',
    'police_vehicle', 'tractor', 'pushcart', 'construction_vehicle',
    'pothole', 'route_board', 'traffic_sign', 'traffic_light',
    'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
    'marked_bump', 'zebra_crossing',
    'cls_24', 'cls_25', 'cls_26', 'cls_27'
]

# India-specific classes COCO can't see
INDIA_ONLY = {'autorickshaw', 'commercial_vehicle', 'ambulance', 'police_vehicle',
              'tractor', 'pushcart', 'construction_vehicle', 'route_board',
              'temp_barrier', 'traffic_cone', 'rumblestrips', 'unmarked_bump',
              'marked_bump', 'zebra_crossing', 'animal'}

DANGER_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'cat', 'dog', 'horse', 'cow', 'autorickshaw', 'commercial_vehicle',
    'ambulance', 'police_vehicle', 'tractor', 'pushcart',
    'construction_vehicle', 'pothole', 'animal'
}


def _load_tflite(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def _preprocess_for(inp_meta, frame_rgb_uint8):
    if frame_rgb_uint8.shape[:2] != (IMG_SIZE, IMG_SIZE):
        frame_rgb_uint8 = np.array(
            Image.fromarray(frame_rgb_uint8).resize((IMG_SIZE, IMG_SIZE))
        )
    arr = frame_rgb_uint8.astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    if inp_meta['dtype'] == np.int8:
        return (arr * 255 - 128).astype(np.int8)
    if inp_meta['dtype'] == np.float16:
        return arr.astype(np.float16)
    return arr.astype(np.float32)


def _dequant_output(out_meta, raw):
    if out_meta['dtype'] == np.int8:
        scale, zp = out_meta['quantization']
        if scale != 0:
            return (raw.astype(np.float32) - zp) * scale
    return raw


class Perception:
    def __init__(self):
        self.coco, self.c_in, self.c_out = _load_tflite(YOLO_COCO_PATH)
        print(f"[Perception] COCO YOLO loaded ({os.path.getsize(YOLO_COCO_PATH)/1024:.0f} KB)")
        self.india, self.i_in, self.i_out = _load_tflite(YOLO_INDIA_PATH)
        print(f"[Perception] India YOLO loaded ({os.path.getsize(YOLO_INDIA_PATH)/1024:.0f} KB)")
        self.pothole, self.p_in, self.p_out = _load_tflite(POTHOLE_PATH)
        print(f"[Perception] Pothole loaded ({os.path.getsize(POTHOLE_PATH)/1024:.0f} KB)")

    def _yolo_postprocess(self, raw, out_meta, names, conf_thresh, n_classes):
        raw = _dequant_output(out_meta, raw)
        n_chan = 4 + n_classes
        if raw.ndim == 3 and raw.shape[1] == n_chan:
            out = raw[0].T
        else:
            out = raw.reshape(raw.shape[0], -1, n_chan)[0]
        cls_scores = out[:, 4:]
        max_scores = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)
        detections = []
        for i in np.where(max_scores > conf_thresh)[0]:
            cls_id = int(cls_ids[i])
            if cls_id < len(names):
                detections.append((names[cls_id], float(max_scores[i])))
        return detections

    def _pothole_postprocess(self, raw):
        raw = _dequant_output(self.p_out, raw)
        if raw.ndim == 3 and raw.shape[1] == 5:
            out = raw[0].T
        else:
            out = raw.reshape(raw.shape[0], -1, 5)[0]
        scores = out[:, 4]
        return [float(s) for s in scores if s > POTHOLE_CONF_THRESH]

    def infer(self, frame_rgb_uint8):
        # COCO YOLO
        csamp = _preprocess_for(self.c_in, frame_rgb_uint8)
        t0 = time.time()
        self.coco.set_tensor(self.c_in['index'], csamp)
        self.coco.invoke()
        craw = self.coco.get_tensor(self.c_out['index'])
        coco_lat = (time.time() - t0) * 1000
        coco_dets = self._yolo_postprocess(craw, self.c_out, COCO_NAMES, CONF_THRESH_COCO, 80)

        # India YOLO - only keep India-specific classes (avoid bicycle bias on common classes)
        isamp = _preprocess_for(self.i_in, frame_rgb_uint8)
        t1 = time.time()
        self.india.set_tensor(self.i_in['index'], isamp)
        self.india.invoke()
        iraw = self.india.get_tensor(self.i_out['index'])
        india_lat = (time.time() - t1) * 1000
        india_all = self._yolo_postprocess(iraw, self.i_out, INDIA_NAMES, CONF_THRESH_INDIA, 28)
        india_dets = [d for d in india_all if d[0] in INDIA_ONLY]

        # Pothole
        psamp = _preprocess_for(self.p_in, frame_rgb_uint8)
        t2 = time.time()
        self.pothole.set_tensor(self.p_in['index'], psamp)
        self.pothole.invoke()
        praw = self.pothole.get_tensor(self.p_out['index'])
        pothole_lat = (time.time() - t2) * 1000
        pothole_scores = self._pothole_postprocess(praw)

        detections = coco_dets + india_dets
        for s in pothole_scores:
            detections.append(('pothole', s))

        has_danger = any(d[0] in DANGER_CLASSES for d in detections)
        return {
            'detections': detections,
            'has_danger': has_danger,
            'pothole_present': len(pothole_scores) > 0,
            'latency_ms': coco_lat + india_lat + pothole_lat,
            'coco_latency_ms': coco_lat,
            'india_latency_ms': india_lat,
            'pothole_latency_ms': pothole_lat,
        }


if __name__ == '__main__':
    import glob
    p = Perception()
    test_imgs = (sorted(glob.glob(os.path.expanduser('~/mini_adas/test_images/vehicles/*.jpg')))[:3] +
                 sorted(glob.glob(os.path.expanduser('~/mini_adas/test_images/potholes/*.jpg')))[:3])
    print(f"\nTesting on {len(test_imgs)} mixed images:")
    print(f"{'Image':<35} {'Lat':>6} {'COCO':>6} {'INDIA':>6} {'Hole':>6} {'Result':>8} Detections")
    print("-" * 110)
    for path in test_imgs:
        img = np.array(Image.open(path).convert('RGB'))
        r = p.infer(img)
        flag = 'DANGER' if r['has_danger'] else 'safe'
        names = [n for n, _ in r['detections'][:4]]
        label = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
        print(f"{label:<35} {r['latency_ms']:>5.0f}ms {r['coco_latency_ms']:>5.0f}ms {r['india_latency_ms']:>5.0f}ms {r['pothole_latency_ms']:>5.0f}ms {flag:>8} {names}")