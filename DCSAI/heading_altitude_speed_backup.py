import math
import socket
import time

import pyvjoy

HOST = "127.0.0.1"
PORT = 5005

DEVICE_ID = 1

TARGET_HEADING_DEG = 180
TARGET_ALTITUDE_M = 3000.0
TARGET_IAS_MPS = 138.9

HEADING_KP = 1.2
MAX_TARGET_ROLL_RAD = math.radians(25.0)

ROLL_KP = 0.8
ROLL_KD = 0.20

PITCH_KP = 0.7
PITCH_KD = 0.15

ALTITUDE_KP = 0.0015
MAX_TARGET_PITCH_RAD = math.radians(10.0)

THROTTLE_TRIM = 0.75
AIRSPEED_KP = 0.003
AIRSPEED_KI = 0.00015
MAX_THROTTLE_CHANGE_PER_SEC = 0.08

MAX_AILERON = 0.25
MAX_ELEVATOR = 0.20

INVERT_AILERON = False
INVERT_ELEVATOR = False
INVERT_THROTTLE = False

AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x7FFF


def clamp(value, low, high):
    return max(low, min(value, high))


def wrap_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalized_to_vjoy(command):
    command = clamp(command, -1.0, 1.0)
    return int(AXIS_CENTER + command * (AXIS_MAX - AXIS_CENTER))


def throttle_to_vjoy(command):
    command = clamp(command, 0.0, 1.0)
    return int(AXIS_MIN + command * (AXIS_MAX - AXIS_MIN))


def set_axis(joystick, axis, command):
    joystick.set_axis(axis, normalized_to_vjoy(command))


def set_throttle(joystick, command):
    joystick.set_axis(
        pyvjoy.HID_USAGE_Z,
        throttle_to_vjoy(command),
    )


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
set_throttle(joystick, THROTTLE_TRIM)

previous_roll_error = 0.0
previous_pitch_error = 0.0
previous_time = time.monotonic()

speed_integral = 0.0
throttle_command = THROTTLE_TRIM

target_heading_rad = math.radians(TARGET_HEADING_DEG)

print("Heading + altitude + airspeed hold is armed.")
print(f"Target heading: {TARGET_HEADING_DEG:.0f} degrees")
print(f"Target altitude: {TARGET_ALTITUDE_M:.0f} m")
print(
    f"Target airspeed: {TARGET_IAS_MPS:.1f} m/s "
    f"({TARGET_IAS_MPS * 3.6:.0f} km/h)"
)
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
        yaw = telemetry["yaw"]
        altitude = telemetry["altitude"]
        ias = telemetry["ias"]

        heading_error = wrap_angle_rad(target_heading_rad - yaw)

        target_roll = HEADING_KP * heading_error
        target_roll = clamp(
            target_roll,
            -MAX_TARGET_ROLL_RAD,
            MAX_TARGET_ROLL_RAD,
        )

        roll_error = target_roll - roll
        roll_error_rate = (roll_error - previous_roll_error) / dt
        previous_roll_error = roll_error

        aileron = ROLL_KP * roll_error + ROLL_KD * roll_error_rate
        aileron = clamp(aileron, -MAX_AILERON, MAX_AILERON)

        altitude_error = TARGET_ALTITUDE_M - altitude

        target_pitch = ALTITUDE_KP * altitude_error
        target_pitch = clamp(
            target_pitch,
            -MAX_TARGET_PITCH_RAD,
            MAX_TARGET_PITCH_RAD,
        )

        pitch_error = target_pitch - pitch
        pitch_error_rate = (pitch_error - previous_pitch_error) / dt
        previous_pitch_error = pitch_error

        elevator = PITCH_KP * pitch_error + PITCH_KD * pitch_error_rate
        elevator = clamp(elevator, -MAX_ELEVATOR, MAX_ELEVATOR)

        airspeed_error = TARGET_IAS_MPS - ias

        candidate_integral = clamp(
            speed_integral + airspeed_error * dt,
            -100.0,
            100.0,
        )

        desired_throttle = (
            THROTTLE_TRIM
            + AIRSPEED_KP * airspeed_error
            + AIRSPEED_KI * candidate_integral
        )
        desired_throttle = clamp(desired_throttle, 0.0, 1.0)

        max_throttle_step = MAX_THROTTLE_CHANGE_PER_SEC * dt
        throttle_delta = clamp(
            desired_throttle - throttle_command,
            -max_throttle_step,
            max_throttle_step,
        )

        throttle_command = clamp(
            throttle_command + throttle_delta,
            0.0,
            1.0,
        )

        if abs(desired_throttle - throttle_command) < 0.20:
            speed_integral = candidate_integral

        if INVERT_AILERON:
            aileron = -aileron

        if INVERT_ELEVATOR:
            elevator = -elevator

        throttle_output = throttle_command

        if INVERT_THROTTLE:
            throttle_output = 1.0 - throttle_output

        set_axis(joystick, pyvjoy.HID_USAGE_X, aileron)
        set_axis(joystick, pyvjoy.HID_USAGE_Y, elevator)
        set_throttle(joystick, throttle_output)

        heading_deg = math.degrees(yaw) % 360.0
        heading_error_deg = math.degrees(heading_error)

        print(
            f"hdg={heading_deg:6.1f} deg  "
            f"hdg_err={heading_error_deg:+6.1f} deg  "
            f"bank_tgt={math.degrees(target_roll):+5.1f} deg  "
            f"roll={math.degrees(roll):+5.1f} deg  "
            f"alt={altitude:7.1f} m  "
            f"IAS={ias * 3.6:6.1f} km/h  "
            f"thr={throttle_command * 100:5.1f}%"
        )

except KeyboardInterrupt:
    print("\nStopping autopilot.")

finally:
    set_axis(joystick, pyvjoy.HID_USAGE_X, 0.0)
    set_axis(joystick, pyvjoy.HID_USAGE_Y, 0.0)
    set_throttle(joystick, THROTTLE_TRIM)
    sock.close()
    print("Aileron and elevator centered. Throttle returned to trim.")