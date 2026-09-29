import csv
import math
import socket
import time
from pathlib import Path

import pyvjoy

HOST = "127.0.0.1"
PORT = 5005
DEVICE_ID = 1

TARGET_HEADING_DEG = 272.0
TARGET_ALTITUDE_M = 3000.0
TARGET_IAS_MPS = 138.9

HEADING_KP = 0.8
MAX_TARGET_ROLL_RAD = math.radians(15.0)

ROLL_KP = 0.65
ROLL_KD = 0.12
MAX_AILERON = 0.20

ALTITUDE_KP = 0.00045
VERTICAL_SPEED_KD = 0.025
ALTITUDE_KI = 0.00003
ALTITUDE_DEADBAND_M = 5.0
MAX_ELEVATOR = 0.12
MAX_ELEVATOR_CHANGE_PER_SEC = 0.20

THROTTLE_TRIM = 0.75
AIRSPEED_KP = 0.002
AIRSPEED_KI = 0.00005
MAX_THROTTLE_CHANGE_PER_SEC = 0.04

INVERT_AILERON = False
INVERT_ELEVATOR = False
INVERT_THROTTLE = False

AXIS_MIN = 0x0001
AXIS_CENTER = 0x4000
AXIS_MAX = 0x7FFF

LOG_PATH = Path("logs/autopilot_training_data.csv")


def clamp(value, low, high):
    return max(low, min(value, high))


def move_toward(current, target, max_change):
    return current + clamp(target - current, -max_change, max_change)


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
    joystick.set_axis(pyvjoy.HID_USAGE_Z, throttle_to_vjoy(command))


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


LOG_FIELDS = [
    "time_s",
    "pitch_rad",
    "roll_rad",
    "yaw_rad",
    "altitude_m",
    "ias_mps",
    "vertical_speed_mps",
    "target_heading_deg",
    "target_altitude_m",
    "target_ias_mps",
    "heading_error_rad",
    "target_roll_rad",
    "altitude_error_m",
    "altitude_integral",
    "airspeed_error_mps",
    "aileron_cmd",
    "elevator_cmd",
    "throttle_cmd",
]


LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
sock.settimeout(0.5)

joystick = pyvjoy.VJoyDevice(DEVICE_ID)

set_axis(joystick, pyvjoy.HID_USAGE_X, 0.0)
set_axis(joystick, pyvjoy.HID_USAGE_Y, 0.0)
set_throttle(joystick, THROTTLE_TRIM)

previous_roll_error = 0.0
previous_time = time.monotonic()
previous_elevator = 0.0

altitude_integral = 0.0
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
print(f"Logging training data to: {LOG_PATH}")
print("Altitude error is recalculated from live DCS telemetry every update.")
print("Waiting for DCS telemetry on UDP port 5005.")
print("Start the Su-25T airborne mission.")
print("Press Ctrl+C to stop and center controls.")

try:
    with LOG_PATH.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()

        while True:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue

            telemetry = parse_packet(data)

            now = time.monotonic()
            dt = clamp(now - previous_time, 0.01, 0.15)
            previous_time = now

            roll = telemetry["roll"]
            yaw = telemetry["yaw"]
            altitude = telemetry["altitude"]
            ias = telemetry["ias"]
            vertical_speed = telemetry["vertical_speed"]

            heading_error = wrap_angle_rad(target_heading_rad - yaw)
            target_roll = clamp(
                HEADING_KP * heading_error,
                -MAX_TARGET_ROLL_RAD,
                MAX_TARGET_ROLL_RAD,
            )

            roll_error = target_roll - roll
            roll_error_rate = (roll_error - previous_roll_error) / dt
            previous_roll_error = roll_error

            aileron = ROLL_KP * roll_error + ROLL_KD * roll_error_rate
            aileron = clamp(aileron, -MAX_AILERON, MAX_AILERON)

            altitude_error = TARGET_ALTITUDE_M - altitude

            if abs(altitude_error) <= ALTITUDE_DEADBAND_M:
                altitude_error_for_control = 0.0
            else:
                altitude_error_for_control = altitude_error

            candidate_altitude_integral = clamp(
                altitude_integral + altitude_error_for_control * dt,
                -500.0,
                500.0,
            )

            requested_elevator = (
                ALTITUDE_KP * altitude_error_for_control
                - VERTICAL_SPEED_KD * vertical_speed
                + ALTITUDE_KI * candidate_altitude_integral
            )
            requested_elevator = clamp(
                requested_elevator,
                -MAX_ELEVATOR,
                MAX_ELEVATOR,
            )

            elevator = move_toward(
                previous_elevator,
                requested_elevator,
                MAX_ELEVATOR_CHANGE_PER_SEC * dt,
            )
            previous_elevator = elevator

            if abs(requested_elevator) < MAX_ELEVATOR * 0.95:
                altitude_integral = candidate_altitude_integral

            airspeed_error = TARGET_IAS_MPS - ias

            candidate_speed_integral = clamp(
                speed_integral + airspeed_error * dt,
                -80.0,
                80.0,
            )

            desired_throttle = clamp(
                THROTTLE_TRIM
                + AIRSPEED_KP * airspeed_error
                + AIRSPEED_KI * candidate_speed_integral,
                0.0,
                1.0,
            )

            throttle_command = move_toward(
                throttle_command,
                desired_throttle,
                MAX_THROTTLE_CHANGE_PER_SEC * dt,
            )

            if abs(desired_throttle - throttle_command) < 0.10:
                speed_integral = candidate_speed_integral

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

            writer.writerow(
                {
                    "time_s": telemetry["time_s"],
                    "pitch_rad": telemetry["pitch"],
                    "roll_rad": roll,
                    "yaw_rad": yaw,
                    "altitude_m": altitude,
                    "ias_mps": ias,
                    "vertical_speed_mps": vertical_speed,
                    "target_heading_deg": TARGET_HEADING_DEG,
                    "target_altitude_m": TARGET_ALTITUDE_M,
                    "target_ias_mps": TARGET_IAS_MPS,
                    "heading_error_rad": heading_error,
                    "target_roll_rad": target_roll,
                    "altitude_error_m": altitude_error,
                    "altitude_integral": altitude_integral,
                    "airspeed_error_mps": airspeed_error,
                    "aileron_cmd": aileron,
                    "elevator_cmd": elevator,
                    "throttle_cmd": throttle_command,
                }
            )
            log_file.flush()

            heading_deg = math.degrees(yaw) % 360.0

            print(
                f"hdg={heading_deg:6.1f} deg  "
                f"alt={altitude:7.1f} m  "
                f"target={TARGET_ALTITUDE_M:7.1f} m  "
                f"err={altitude_error:+7.1f} m  "
                f"vs={vertical_speed:+5.2f} m/s  "
                f"elev={elevator:+.3f}  "
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