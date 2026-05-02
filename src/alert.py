"""
Alert module for Mini ADAS.
Two modes:
  - 'sim': prints colored text to console (for laptop testing)
  - 'pi':  drives real GPIO LEDs + buzzer on Raspberry Pi
"""
import time


# Default GPIO pins on Pi (BCM numbering)
PIN_GREEN = 5
PIN_YELLOW = 22
PIN_RED = 27
PIN_BUZZER = 17

# ANSI color codes for sim mode console output
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_RED = '\033[91m'
ANSI_RESET = '\033[0m'
ANSI_BOLD = '\033[1m'


class AlertSystem:
    def __init__(self, mode='sim'):
        self.mode = mode
        self.last_level = None
        self.last_buzz_time = 0

        if mode == 'pi':
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                for pin in [PIN_GREEN, PIN_YELLOW, PIN_RED, PIN_BUZZER]:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, False)
                print("[Alert] Pi mode ready (GPIO LEDs + buzzer)")
            except ImportError:
                raise RuntimeError("RPi.GPIO not available. Use mode='sim' on laptop.")
        else:
            print("[Alert] Sim mode ready (console output only)")

    def set_level(self, level):
        """level: 'SAFE' | 'WARN' | 'BRAKE'"""
        if self.mode == 'pi':
            self.GPIO.output(PIN_GREEN, level == 'SAFE')
            self.GPIO.output(PIN_YELLOW, level == 'WARN')
            self.GPIO.output(PIN_RED, level == 'BRAKE')
            # Buzzer pattern
            now = time.time()
            if level == 'BRAKE':
                self.GPIO.output(PIN_BUZZER, True)  # continuous
            elif level == 'WARN':
                # Beep once every 0.5s
                if now - self.last_buzz_time > 0.5:
                    self.GPIO.output(PIN_BUZZER, True)
                    self.last_buzz_time = now
                elif now - self.last_buzz_time > 0.1:
                    self.GPIO.output(PIN_BUZZER, False)
            else:
                self.GPIO.output(PIN_BUZZER, False)
        else:
            # Sim mode - only print when level changes (avoid console spam)
            if level != self.last_level:
                color = {'SAFE': ANSI_GREEN, 'WARN': ANSI_YELLOW, 'BRAKE': ANSI_RED}[level]
                symbol = {'SAFE': '[ OK  ]', 'WARN': '[WARN ]', 'BRAKE': '[BRAKE]'}[level]
                print(f"  {color}{ANSI_BOLD}{symbol}{ANSI_RESET} alert level changed to {level}")

        self.last_level = level

    def cleanup(self):
        if self.mode == 'pi' and hasattr(self, 'GPIO'):
            for pin in [PIN_GREEN, PIN_YELLOW, PIN_RED, PIN_BUZZER]:
                self.GPIO.output(pin, False)
            self.GPIO.cleanup()


# Self-test - cycle through alert levels
if __name__ == '__main__':
    alert = AlertSystem(mode='sim')
    print("\nCycling SAFE -> WARN -> BRAKE -> SAFE")
    for level in ['SAFE', 'WARN', 'BRAKE', 'WARN', 'SAFE']:
        alert.set_level(level)
        time.sleep(0.3)
    print("\nDone.")