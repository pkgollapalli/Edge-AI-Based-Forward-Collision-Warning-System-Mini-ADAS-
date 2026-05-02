"""
Distance sensor module for Mini ADAS.
Two modes:
  - 'sim': simulated distance for laptop testing (manually set or scripted)
  - 'pi': real HC-SR04 ultrasonic via RPi.GPIO (only works on Pi)
Also tracks closing speed by keeping last N distance readings.
"""
import time
from collections import deque


class DistanceSensor:
    def __init__(self, mode='sim', trig_pin=23, echo_pin=24, history_size=5):
        self.mode = mode
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.history = deque(maxlen=history_size)  # (timestamp, distance_cm)
        self._sim_distance = 200.0  # default starting distance for sim mode

        if mode == 'pi':
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(trig_pin, GPIO.OUT)
                GPIO.setup(echo_pin, GPIO.IN)
                GPIO.output(trig_pin, False)
                time.sleep(0.1)  # settle
                print(f"[Distance] Pi mode ready (TRIG={trig_pin}, ECHO={echo_pin})")
            except ImportError:
                raise RuntimeError("RPi.GPIO not available. Use mode='sim' on laptop.")
        else:
            print(f"[Distance] Sim mode ready (manually set self._sim_distance)")

    def set_sim_distance(self, distance_cm):
        """Sim mode only - set the next distance reading."""
        self._sim_distance = max(2.0, min(400.0, distance_cm))

    def _read_pi(self):
        """Real HC-SR04 ping read. Returns distance in cm."""
        self.GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10 microseconds pulse
        self.GPIO.output(self.trig_pin, False)
        timeout_start = time.time()
        pulse_start = pulse_end = timeout_start
        while self.GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.04:
                return 400.0  # timeout = no echo = far
        while self.GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - timeout_start > 0.04:
                return 400.0
        duration_us = (pulse_end - pulse_start) * 1_000_000
        distance_cm = duration_us / 58.0
        return max(2.0, min(400.0, distance_cm))

    def read(self):
        """Returns: {'distance_cm': float, 'closing_speed_cmps': float, 'timestamp': float}"""
        ts = time.time()
        if self.mode == 'pi':
            dist = self._read_pi()
        else:
            dist = self._sim_distance
        
        self.history.append((ts, dist))
        
        # Compute closing speed (positive = approaching)
        closing_speed = 0.0
        if len(self.history) >= 2:
            t0, d0 = self.history[0]
            t1, d1 = self.history[-1]
            dt = t1 - t0
            if dt > 0:
                closing_speed = (d0 - d1) / dt  # cm/s, positive = closing
        
        return {'distance_cm': dist, 'closing_speed_cmps': closing_speed, 'timestamp': ts}

    def cleanup(self):
        if self.mode == 'pi' and hasattr(self, 'GPIO'):
            self.GPIO.cleanup()


# Self-test - simulate hand approaching camera
if __name__ == '__main__':
    print("\nSimulating hand approaching from 200 cm to 10 cm over 5 seconds...")
    sensor = DistanceSensor(mode='sim')
    
    distances = [200, 150, 110, 80, 50, 30, 15, 10, 10, 10]
    for d in distances:
        sensor.set_sim_distance(d)
        r = sensor.read()
        print(f"  dist={r['distance_cm']:6.1f} cm | closing_speed={r['closing_speed_cmps']:7.1f} cm/s")
        time.sleep(0.2)