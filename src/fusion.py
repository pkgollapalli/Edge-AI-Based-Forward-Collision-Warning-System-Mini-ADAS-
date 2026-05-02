"""
Fusion module for Mini ADAS.
Combines perception (what is in front) + distance (how far + closing speed)
to produce one of three risk levels: SAFE / WARN / BRAKE.

Decision logic (rule-based, defensible in Q&A):
  - BRAKE: danger object detected AND (distance<50cm OR TTC<0.8s)
  - WARN:  danger object detected AND (distance<150cm OR TTC<2s)
  - SAFE:  otherwise

TTC = Time To Collision = distance / closing_speed (when closing_speed > 0)
"""

# Thresholds (tunable)
BRAKE_DIST_CM = 50.0
WARN_DIST_CM = 150.0
BRAKE_TTC_S = 0.8
WARN_TTC_S = 2.0
MIN_CLOSING_FOR_TTC = 5.0  # cm/s; below this, object isn't really approaching

LEVELS = ['SAFE', 'WARN', 'BRAKE']


def compute_ttc(distance_cm, closing_speed_cmps):
    """Time To Collision in seconds. Positive closing_speed = approaching."""
    if closing_speed_cmps <= MIN_CLOSING_FOR_TTC:
        return float('inf')
    return distance_cm / closing_speed_cmps


def fuse(perception_result, distance_result):
    """
    perception_result: dict from Perception.infer()
    distance_result:   dict from DistanceSensor.read()
    Returns: dict with 'level' (str), 'reason' (str), 'ttc_s' (float),
             'has_danger' (bool), and forwarded sensor values.
    """
    has_danger = perception_result['has_danger']
    distance_cm = distance_result['distance_cm']
    closing = distance_result['closing_speed_cmps']
    ttc = compute_ttc(distance_cm, closing)

    # Default
    level = 'SAFE'
    reason = 'no danger detected'

    if has_danger:
        if distance_cm < BRAKE_DIST_CM or ttc < BRAKE_TTC_S:
            level = 'BRAKE'
            if distance_cm < BRAKE_DIST_CM:
                reason = f'danger at {distance_cm:.0f}cm (<{BRAKE_DIST_CM:.0f})'
            else:
                reason = f'danger TTC={ttc:.2f}s (<{BRAKE_TTC_S})'
        elif distance_cm < WARN_DIST_CM or ttc < WARN_TTC_S:
            level = 'WARN'
            if distance_cm < WARN_DIST_CM:
                reason = f'danger at {distance_cm:.0f}cm'
            else:
                reason = f'danger TTC={ttc:.2f}s'
        else:
            level = 'SAFE'
            reason = f'danger present but far ({distance_cm:.0f}cm)'
    else:
        # No danger object - even if distance is low, do not alert
        # (would prevent false alarms from walls, your own desk, etc.)
        reason = f'no danger object (dist={distance_cm:.0f}cm)'

    detected_names = [n for n, _ in perception_result.get('detections', [])][:3]

    return {
        'level': level,
        'reason': reason,
        'ttc_s': ttc if ttc != float('inf') else None,
        'has_danger': has_danger,
        'distance_cm': distance_cm,
        'closing_speed_cmps': closing,
        'detected_classes': detected_names,
    }


# Self-test - simulate full ADAS scenarios
if __name__ == '__main__':
    print("Scenario A: car approaching from 200cm to 20cm")
    print("-" * 70)
    distances_speeds = [
        (200, 0),     # far, not closing yet
        (180, 100),   # closing slowly
        (150, 150),   # WARN territory
        (100, 200),   # WARN
        (60, 250),    # close to BRAKE
        (40, 200),    # BRAKE
        (20, 100),    # BRAKE
    ]
    for d, s in distances_speeds:
        per = {'has_danger': True, 'detections': [('car', 0.92)], 'latency_ms': 10}
        dist = {'distance_cm': d, 'closing_speed_cmps': s, 'timestamp': 0}
        out = fuse(per, dist)
        print(f"  dist={d:3d}cm speed={s:4d}cm/s -> {out['level']:5s}: {out['reason']}")

    print("\nScenario B: no danger object (wall) at 30cm - must NOT alert")
    print("-" * 70)
    per = {'has_danger': False, 'detections': [], 'latency_ms': 10}
    dist = {'distance_cm': 30, 'closing_speed_cmps': 100, 'timestamp': 0}
    out = fuse(per, dist)
    print(f"  dist=30cm speed=100cm/s -> {out['level']:5s}: {out['reason']}")

    print("\nScenario C: pedestrian crossing - person detected close")
    print("-" * 70)
    per = {'has_danger': True, 'detections': [('person', 0.81)], 'latency_ms': 10}
    dist = {'distance_cm': 80, 'closing_speed_cmps': 50, 'timestamp': 0}
    out = fuse(per, dist)
    print(f"  pedestrian at 80cm closing 50cm/s -> {out['level']:5s}: {out['reason']}")