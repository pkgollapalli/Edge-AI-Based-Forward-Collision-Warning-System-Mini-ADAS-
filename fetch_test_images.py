import os
import urllib.request
import time

# Free CC0 images from Picsum + free vehicle photos from Unsplash
VEHICLE_URLS = [
    "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=400",  # car
    "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",  # car
    "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=400",  # car
    "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400",  # car
    "https://images.unsplash.com/photo-1542362567-b07e54358753?w=400",     # car
    "https://images.unsplash.com/photo-1485291571150-772bcfc10da5?w=400",  # bus
    "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=400",     # truck
    "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400",  # car
    "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=400",  # car
    "https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?w=400",  # car
]

NON_VEHICLE_URLS = [
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # forest
    "https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=400",  # ocean
    "https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=400",  # mountain
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # mountain
    "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=400",  # nature
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=400",  # nature
    "https://images.unsplash.com/photo-1444930694458-01babe71870e?w=400",  # animal
    "https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=400",     # cat
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400",     # building
    "https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?w=400",  # snow
]

def download(url, path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            with open(path, 'wb') as f:
                f.write(r.read())
        print(f"  OK: {path}")
        return True
    except Exception as e:
        print(f"  FAIL: {path} ({e})")
        return False

print("Downloading vehicle images...")
for i, url in enumerate(VEHICLE_URLS):
    download(url, f'test_images/vehicles/v{i+1:02d}.jpg')
    time.sleep(0.3)

print("\nDownloading non-vehicle images...")
for i, url in enumerate(NON_VEHICLE_URLS):
    download(url, f'test_images/non_vehicles/n{i+1:02d}.jpg')
    time.sleep(0.3)

veh = len([f for f in os.listdir('test_images/vehicles') if f.endswith('.jpg')])
non = len([f for f in os.listdir('test_images/non_vehicles') if f.endswith('.jpg')])
print(f"\nDone. Got {veh} vehicle, {non} non-vehicle images.")