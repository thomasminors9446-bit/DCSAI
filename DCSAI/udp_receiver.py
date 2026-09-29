import socket

HOST = "127.0.0.1"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening for DCS telemetry on UDP {HOST}:{PORT}")
print("Start an airborne DCS mission. Press Ctrl+C here to stop.")

try:
    while True:
        data, address = sock.recvfrom(1024)
        text = data.decode("utf-8").strip()
        values = [float(value) for value in text.split(",")]

        time_s, pitch, roll, yaw, altitude, ias, vertical_speed = values

        print(
            f"t={time_s:7.2f}  "
            f"roll={roll:+.3f} rad  "
            f"alt={altitude:7.1f} m  "
            f"IAS={ias:6.1f} m/s"
        )

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    sock.close()