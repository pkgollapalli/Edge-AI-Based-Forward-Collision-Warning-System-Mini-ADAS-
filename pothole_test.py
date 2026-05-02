import os, glob, time
from ultralytics import YOLO

# Direct download from Hugging Face
print("Loading pothole-detection model from Hugging Face...")
pothole_model = YOLO("https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt")
print(f"Model loaded. Classes: {pothole_model.names}")

# Download a few public pothole images for testing
import urllib.request
POTHOLE_URLS = [
    "https://images.unsplash.com/photo-1592963103120-c4076014fa9d?w=640",  # pothole-like
    "https://images.unsplash.com/photo-1601584115197-04ecc0da31d8?w=640",  # damaged road
]

print("\nDownloading test pothole images...")
for i, url in enumerate(POTHOLE_URLS):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            with open(f'test_images/potholes/p{i+1:02d}.jpg', 'wb') as f:
                f.write(r.read())
        print(f"  OK: p{i+1:02d}.jpg")
    except Exception as e:
        print(f"  FAIL: {e}")

# Test on existing road images we already have + new pothole images
test_imgs = sorted(glob.glob('test_images/vehicles/*.jpg')) + \
            sorted(glob.glob('test_images/non_vehicles/*.jpg')) + \
            sorted(glob.glob('test_images/potholes/*.jpg'))

print(f"\nTesting pothole model on {len(test_imgs)} images:")
print(f"{'Image':<25} {'Lat':>6} {'Detections':>40}")
print("-" * 75)

for img_path in test_imgs:
    t0 = time.time()
    results = pothole_model(img_path, verbose=False)[0]
    elapsed_ms = (time.time() - t0) * 1000

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf > 0.25:
            detections.append(f'pothole({conf:.2f})')

    fname = os.path.basename(img_path)
    folder = os.path.basename(os.path.dirname(img_path))
    label = f"{folder}/{fname}"
    detected_str = ', '.join(detections[:3]) if detections else '(none)'
    print(f"{label:<25} {elapsed_ms:>5.0f}ms {detected_str:>40}")