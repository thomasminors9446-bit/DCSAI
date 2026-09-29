import math
import socket
import time

import pyvjoy

HOST = "127.0.0.1"
PORT = 5005

DEVICE_ID = 1

TARGET_ROLL_RAD = 0.0
TARGET_PITCH_RAD = 0.0

ROLL_KP = 0.8
ROLL_KD = 0.20
PITCH_KP = 0.7
PITCH_KD = 0.15

MAX_AILERON = 0.25
MAX_ELEVATOR = 0.20

INVERT_AILERON = False
INVERT_ELEVATOR = False

AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x7FFF


def clamp(value, low, high):
    return max(low, min(value, high))


def normalized_to_vjoy(command):
    command = clamp(command, -1.0, 1.0)
    return int(AXIS_CENTER + command * (AXIS_MAX - AXIS_CENTER))


def set_axis(joystick, axis, command):
    joystick.set_axis(axis, normalized_to_vjoy(command))


def parse_packet(data):
    text = data.decode("utf-8").strip()
    values = [float(value) for value in text.split(",")]

    if len(values) != 7:
        raise ValueError(f"Expected 7 values, got {len(values)}")

    time_s, pitch, roll, yaw, altitude, ias, vertical_speed = values

    return {
        "time_s": time_s,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "altitude": altitude,
        "ias": ias,
        "vertical_speed": vertical_speed,
    }


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
sock.settimeout(0.5)

joystick = pyvjoy.VJoyDevice(DEVICE_ID)

set_axis(joystick, pyvjoy.HID_USAGE_X, 0.0)
set_axis(joystick, pyvjoy.HID_USAGE_Y, 0.0)

previous_roll_error = 0.0
previous_pitch_error = 0.0
previous_time = time.monotonic()

print("Roll + pitch hold is armed.")
print("Waiting for DCS telemetry on UDP port 5005.")
print("Start the Su-25T airborne mission.")
print("Press Ctrl+C to stop and center controls.")

try:
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except socket.timeout:
            continue

        telemetry = parse_packet(data)

        now = time.monotonic()
        dt = max(now - previous_time, 0.001)
        previous_time = now

        roll = telemetry["roll"]
        pitch = telemetry["pitch"]

        roll_error = TARGET_ROLL_RAD - roll
        roll_error_rate = (roll_error - previous_roll_error) / dt
        previous_roll_error = roll_error

        aileron = ROLL_KP * roll_error + ROLL_KD * roll_error_rate
        aileron = clamp(aileron, -MAX_AILERON, MAX_AILERON)

        pitch_error = TARGET_PITCH_RAD - pitch
        pitch_error_rate = (pitch_error - previous_pitch_error) / dt
        previous_pitch_error = pitch_error

        elevator = PITCH_KP * pitch_error + PITCH_KD * pitch_error_rate
        elevator = clamp(elevator, -MAX_ELEVATOR, MAX_ELEVATOR)

        if INVERT_AILERON:
            aileron = -aileron

        if INVERT_ELEVATOR:
            elevator = -elevator

        set_axis(joystick, pyvjoy.HID_USAGE_X, aileron)
        set_axis(joystick, pyvjoy.HID_USAGE_Y, elevator)

        print(
            f"roll={math.degrees(roll):+6.1f} deg  "
            f"ail={aileron:+.3f}  "
            f"pitch={math.degrees(pitch):+6.1f} deg  "
            f"elev={elevator:+.3f}"
        )

except KeyboardInterrupt:
    print("\nStopping autopilot.")

finally:
    set_axis(joystick, pyvjoy.HID_USAGE_X, 0.0)
    set_axis(joystick, pyvjoy.HID_USAGE_Y, 0.0)
    sock.close()
    print("Aileron and elevator centered. Done.")