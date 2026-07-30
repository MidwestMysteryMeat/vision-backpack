"""
queue_writer.py

Writes each capture record to the queue -- either directly to the external
drive if mounted, or to a local buffer that gets flushed once the drive is
available. Records are the unit of work the desktop-side pipeline consumes.

Record format (one JSON file + one image file per capture):
    {timestamp}.json:
        {
            "timestamp": <unix ts>,
            "gps_lat": <float|null>,
            "gps_lon": <float|null>,
            "object_tags": [{"real_label": ..., "fantasy_label": ..., "confidence": ...}, ...],
            "gait_descriptor": <string|null>,   # text label only -- see gait_estimator.py
            "image_path": "{timestamp}.jpg"
        }
    {timestamp}.jpg: the ANONYMIZED frame only -- never the raw capture.
"""

import os
import json
import time
import cv2


class QueueWriter:
    def __init__(self, local_buffer_dir: str, external_drive_mount: str,
                 queue_subdir: str = "queue/", flush_on_capture: bool = True):
        self.local_buffer_dir = local_buffer_dir
        self.external_drive_mount = external_drive_mount
        self.queue_subdir = queue_subdir
        self.flush_on_capture = flush_on_capture
        os.makedirs(self.local_buffer_dir, exist_ok=True)

    def _drive_available(self) -> bool:
        return os.path.ismount(self.external_drive_mount)

    def _target_dir(self) -> str:
        if self.flush_on_capture and self._drive_available():
            target = os.path.join(self.external_drive_mount, self.queue_subdir)
        else:
            target = self.local_buffer_dir
        os.makedirs(target, exist_ok=True)
        return target

    def write_record(self, anonymized_frame, gps_lat, gps_lon, object_tags: list,
                      gait_descriptor: str = None):
        ts = time.time()
        ts_str = f"{ts:.3f}".replace(".", "_")
        target_dir = self._target_dir()

        image_filename = f"{ts_str}.jpg"
        image_path = os.path.join(target_dir, image_filename)
        cv2.imwrite(image_path, anonymized_frame)

        record = {
            "timestamp": ts,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
            "object_tags": [
                {"real_label": t.real_label, "fantasy_label": t.fantasy_label,
                 "confidence": t.confidence}
                for t in object_tags
            ],
            "gait_descriptor": gait_descriptor,
            "image_path": image_filename,
        }

        json_path = os.path.join(target_dir, f"{ts_str}.json")
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        return json_path

    def flush_buffer_to_drive(self):
        """Move any locally buffered records over to the external drive
        once it becomes available (e.g. plugged in mid-session)."""
        if not self._drive_available():
            return 0

        target = os.path.join(self.external_drive_mount, self.queue_subdir)
        os.makedirs(target, exist_ok=True)

        moved = 0
        for filename in os.listdir(self.local_buffer_dir):
            src = os.path.join(self.local_buffer_dir, filename)
            dst = os.path.join(target, filename)
            if os.path.isfile(src):
                os.rename(src, dst)
                moved += 1
        return moved
