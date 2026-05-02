"""
Mini ADAS - Main pipeline.
Wires: Perception (camera + YOLO) -> Fusion -> Alert
Modes:
  --mode sim         : laptop testing with simulated camera + ultrasonic
  --mode pi          : real Pi camera + HC-SR04 + GPIO alerts
  --scenario approach: laptop sim of a vehicle approaching
  --scenario walk    : laptop sim of a pedestrian crossing
  --scenario static  : static wall (no danger) test
"""
import os
import sys
import time
import argparse
import glob
import json
import numpy as np
from PIL import Image

from src.perception import Perception
from src.distance import DistanceSensor
from src.fusion import fuse
from src.alert import AlertSystem


def get_sim_frame(scenario, step):
    """Return a synthetic test frame based on scenario + step."""
    test_dir = os.path.expanduser('~/mini_adas/test_images')
    if scenario in ('approach', 'walk'):
        veh_imgs = sorted(glob.glob(os.path.join(test_dir, 'vehicles/*.jpg')))
        path = veh_imgs[step % len(veh_imgs)]
    else:
        non_imgs = sorted(glob.glob(os.path.join(test_dir, 'non_vehicles/*.jpg')))
        path = non_imgs[step % len(non_imgs)]
    return np.array(Image.open(path).convert('RGB'))


def get_sim_distance(scenario, step):
    """Simulated distance trajectory in cm."""
    if scenario == 'approach':
        # Vehicle approaching: 200 -> 20 cm over 10 steps
        traj = [200, 180, 150, 110, 80, 60, 40, 25, 15, 10]
    elif scenario == 'walk':
        # Pedestrian crossing: stays around 80-100 cm
        traj = [120, 110, 95, 80, 75, 80, 90, 110, 130, 150]
    else:
        # Static: no movement
        traj = [60] * 10
    return traj[step % len(traj)]


def run_simulation(scenario='approach', steps=10):
    print(f"\n{'='*70}")
    print(f"MINI ADAS - SIMULATION MODE  (scenario: {scenario})")
    print(f"{'='*70}")

    perception = Perception()
    distance = DistanceSensor(mode='sim')
    alert = AlertSystem(mode='sim')

    log = []
    print(f"\n{'step':>4} {'lvl':>5} {'dist':>6} {'speed':>7} {'objects':<25} {'lat':>5}")
    print("-" * 70)

    for step in range(steps):
        # 1. Get frame + ground truth distance
        frame = get_sim_frame(scenario, step)
        sim_dist = get_sim_distance(scenario, step)
        distance.set_sim_distance(sim_dist)

        # 2. Run perception + distance
        per = perception.infer(frame)
        dist = distance.read()

        # 3. Fuse + alert
        decision = fuse(per, dist)
        alert.set_level(decision['level'])

        # 4. Print + log
        objs = ','.join(decision['detected_classes'][:2]) or '-'
        print(f"{step:>4} {decision['level']:>5} "
              f"{decision['distance_cm']:>5.0f}cm {decision['closing_speed_cmps']:>6.0f} "
              f"{objs:<25} {per['latency_ms']:>4.0f}ms")

        log.append({
            'step': step,
            'level': decision['level'],
            'distance_cm': decision['distance_cm'],
            'closing_speed_cmps': decision['closing_speed_cmps'],
            'ttc_s': decision['ttc_s'],
            'detected_classes': decision['detected_classes'],
            'reason': decision['reason'],
            'latency_ms': per['latency_ms'],
        })
        time.sleep(0.2)

    # Save log
    log_path = os.path.expanduser(f'~/mini_adas/logs/sim_{scenario}.json')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved to {log_path}")

    alert.cleanup()
    distance.cleanup()


def run_pi():
    """Real Pi mode - to be used on Raspberry Pi only."""
    print("="*70)
    print("MINI ADAS - PI MODE  (real camera + ultrasonic + LEDs)")
    print("="*70)

    perception = Perception()
    distance = DistanceSensor(mode='pi')
    alert = AlertSystem(mode='pi')

    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"size": (320, 320), "format": "RGB888"}))
        cam.start()
        time.sleep(0.5)
        print("[Camera] Pi camera started\n")
    except Exception as e:
        print(f"Camera init failed: {e}")
        return

    try:
        while True:
            frame = cam.capture_array()
            per = perception.infer(frame)
            dist = distance.read()
            decision = fuse(per, dist)
            alert.set_level(decision['level'])

            objs = ','.join(decision['detected_classes'][:2]) or '-'
            ttc_str = f"{decision['ttc_s']:4.1f}s" if decision['ttc_s'] else '  inf'
            print(f"\r[{decision['level']:5s}] "
                  f"dist={decision['distance_cm']:5.0f}cm "
                  f"closing={decision['closing_speed_cmps']:5.0f} "
                  f"ttc={ttc_str} "
                  f"obj={objs:<20} "
                  f"lat={per['latency_ms']:4.0f}ms", end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        alert.cleanup()
        distance.cleanup()
        cam.stop()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sim', 'pi'], default='sim')
    ap.add_argument('--scenario', choices=['approach', 'walk', 'static'], default='approach')
    ap.add_argument('--steps', type=int, default=10)
    args = ap.parse_args()

    if args.mode == 'pi':
        run_pi()
    else:
        run_simulation(args.scenario, args.steps)