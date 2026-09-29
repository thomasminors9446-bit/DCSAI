import time
import pyvjoy

AXIS_MIN = 0x0001
AXIS_MAX = 0x7FFF

joystick = pyvjoy.VJoyDevice(1)

def set_throttle(percent):
    percent = max(0.0, min(percent, 1.0))
    value = int(AXIS_MIN + percent * (AXIS_MAX - AXIS_MIN))
    joystick.set_axis(pyvjoy.HID_USAGE_Z, value)
    print(f"Throttle command: {percent * 100:.0f}%")

try:
    set_throttle(0.30)
    time.sleep(2)

    set_throttle(0.70)
    time.sleep(2)

    set_throttle(0.50)
    time.sleep(2)

finally:
    set_throttle(0.50)
    print("Finished at 50% throttle.")