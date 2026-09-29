import time
import pyvjoy

j = pyvjoy.VJoyDevice(1)

try:
    print("Centering all axes...")
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)

    time.sleep(2)

    print("Moving X axis fully left...")
    j.set_axis(pyvjoy.HID_USAGE_X, 0x0001)
    time.sleep(2)

    print("Moving X axis fully right...")
    j.set_axis(pyvjoy.HID_USAGE_X, 0x7FFF)
    time.sleep(2)

    print("Centering X axis...")
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)

finally:
    print("Done.")