"""
gps_logger.py

Thin wrapper around a NEO-6M GPS module over serial/UART, parsing NMEA
sentences for a lat/lon fix. Returns None if no fix is available within
the configured timeout -- capture should proceed and log without GPS
rather than blocking indefinitely.
"""

import time
import serial
import pynmea2


class GPSLogger:
    def __init__(self, port: str = "/dev/serial0", baud_rate: int = 9600,
                 fix_timeout_s: float = 10.0):
        self.fix_timeout_s = fix_timeout_s
        self._ser = None
        try:
            self._ser = serial.Serial(port, baud_rate, timeout=1.0)
        except Exception as e:
            print(f"[GPSLogger] Could not open {port}: {e}. GPS tagging disabled.")

    def get_fix(self):
        """Returns (lat, lon) or (None, None) if no fix within timeout."""
        if self._ser is None:
            return (None, None)

        start = time.monotonic()
        while (time.monotonic() - start) < self.fix_timeout_s:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                try:
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        return (msg.latitude, msg.longitude)
                except pynmea2.ParseError:
                    continue

        return (None, None)

    def close(self):
        if self._ser:
            self._ser.close()
