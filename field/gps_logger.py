"""
gps_logger.py

Thin wrapper around a NEO-6M GPS module over serial/UART, parsing NMEA
sentences for a lat/lon fix. Returns None if no fix is available within
the configured timeout: capture should proceed and log without GPS
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

        # The module streams NMEA continuously between captures, so the OS
        # buffer holds sentences that are many seconds old. Drop them and
        # read fresh so the fix reflects where we are now, not where we were.
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

        start = time.monotonic()
        while (time.monotonic() - start) < self.fix_timeout_s:
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            # GGA and RMC both carry a position; the NEO-6M interleaves
            # them, so accepting either roughly halves time-to-fix here.
            if line[:6] in ("$GPGGA", "$GNGGA", "$GPRMC", "$GNRMC"):
                try:
                    msg = pynmea2.parse(line)
                    # Both sentence types flag fix validity; a void fix can
                    # still carry stale coordinates, so check it explicitly.
                    valid = (getattr(msg, "status", "A") == "A"
                             and int(getattr(msg, "gps_qual", 1) or 0) > 0)
                    if valid and msg.latitude and msg.longitude:
                        return (msg.latitude, msg.longitude)
                except (pynmea2.ParseError, ValueError):
                    continue

        return (None, None)

    def close(self):
        if self._ser:
            self._ser.close()
