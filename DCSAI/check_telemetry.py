import csv
import math
from pathlib import Path

path = Path("logs/dcsai_telemetry.csv")

with path.open(newline="") as file:
    rows = list(csv.DictReader(file))

print(f"Rows recorded: {len(rows)}")

if not rows:
    raise SystemExit("No telemetry rows found.")

roll_values = [float(row["roll_rad"]) for row in rows]
altitude_values = [float(row["altitude_m"]) for row in rows]
airspeed_values = [float(row["ias_mps"]) for row in rows]

max_abs_roll = max(abs(value) for value in roll_values)

print(f"Roll range: {min(roll_values):.3f} to {max(roll_values):.3f} rad")
print(f"Maximum bank: {math.degrees(max_abs_roll):.1f} degrees")
print(f"Altitude range: {min(altitude_values):.1f} to {max(altitude_values):.1f} m")
print(f"IAS range: {min(airspeed_values):.1f} to {max(airspeed_values):.1f} m/s")