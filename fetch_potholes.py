import urllib.request, os
os.makedirs('test_images/potholes', exist_ok=True)

# Real pothole images from a public GitHub repo (the original dataset README has these)
URLS = [
    "https://raw.githubusercontent.com/anuragxel/Pothole-Detection-using-YOLOv8/main/sample_images/sample1.jpg",
    "https://raw.githubusercontent.com/anuragxel/Pothole-Detection-using-YOLOv8/main/sample_images/sample2.jpg",
    "https://raw.githubusercontent.com/arpy8/Pothole_Detection_YOLOv8/main/test_imgs/test_image1.jpg",
    "https://raw.githubusercontent.com/arpy8/Pothole_Detection_YOLOv8/main/test_imgs/test_image2.jpg",
]

for i, url in enumerate(URLS):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
            with open(f'test_images/potholes/p{i+1:02d}.jpg', 'wb') as f:
                f.write(data)
        print(f"OK: p{i+1:02d}.jpg ({len(data)//1024} KB)")
    except Exception as e:
        print(f"FAIL p{i+1:02d}: {e}")

import glob
got = glob.glob('test_images/potholes/*.jpg')
print(f"\nGot {len(got)} pothole images")
