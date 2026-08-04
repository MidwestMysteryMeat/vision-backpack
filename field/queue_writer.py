"""
queue_writer.py

Writes each capture record to the queue, either directly to the external
drive if mounted, or to a local buffer that gets flushed once the drive is
available. Records are the unit of work the desktop-side pipeline consumes.

Record format (one JSON file + one image file per capture):
    {timestamp}.json:
        {
            "timestamp": <unix ts>,
            "gps_lat": <float|null>,
            "gps_lon": <float|null>,
            "object_tags": [{"real_label": ..., "fantasy_label": ..., "confidence": ...}, ...],
            "gait_descriptor": <string|null>,   # text label only, see gait_estimator.py
            "image_path": "{timestamp}.jpg"
        }
    {timestamp}.jpg: the ANONYMIZED frame only, never the raw capture.
"""

import logging
import os
import json
import shutil
import time
import uuid
import cv2

logger = logging.getLogger("vb.queue")


def _move_across_filesystems(src: str, dest: str):
    """Move a file even when src and dest are on different filesystems.

    os.replace() raises EXDEV across devices, which is exactly the layout
    here: the local buffer lives on the SD card and the queue lives on the
    external drive. Copy to a hidden temp name on the destination, then
    rename into place so the desktop side never sees a half-copied file.
    """
    try:
        os.replace(src, dest)
        return
    except OSError:
        pass
    tmp = os.path.join(os.path.dirname(dest), f".{os.path.basename(dest)}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)
    os.remove(src)


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
        record_id = uuid.uuid4().hex
        target_dir = self._target_dir()

        image_filename = f"{record_id}.jpg"
        image_path = os.path.join(target_dir, image_filename)
        image_tmp = os.path.join(target_dir, f".{record_id}.tmp.jpg")
        if not cv2.imwrite(image_tmp, anonymized_frame):
            raise IOError(f"Could not write anonymized image: {image_path}")
        os.replace(image_tmp, image_path)

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

        json_path = os.path.join(target_dir, f"{record_id}.json")
        json_tmp = os.path.join(target_dir, f".{record_id}.tmp.json")
        try:
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(json_tmp, json_path)
        except Exception:
            if os.path.exists(json_tmp):
                os.remove(json_tmp)
            if os.path.exists(image_path):
                os.remove(image_path)
            raise

        return json_path

    def flush_buffer_to_drive(self):
        """Move any locally buffered records over to the external drive
        once it becomes available (e.g. plugged in mid-session).

        Fail-soft per record: one unmovable record (drive yanked mid-copy,
        permission hiccup) must not crash the capture loop or block the
        rest of the buffer from flushing.
        """
        if not self._drive_available():
            return 0

        target = os.path.join(self.external_drive_mount, self.queue_subdir)
        os.makedirs(target, exist_ok=True)

        moved = 0
        names = {name.rsplit(".", 1)[0] for name in os.listdir(self.local_buffer_dir)
                 if name.endswith((".json", ".jpg"))}
        for record_id in names:
            json_src = os.path.join(self.local_buffer_dir, f"{record_id}.json")
            image_src = os.path.join(self.local_buffer_dir, f"{record_id}.jpg")
            if not (os.path.isfile(json_src) and os.path.isfile(image_src)):
                continue
            image_dest = os.path.join(target, f"{record_id}.jpg")
            try:
                # Image first, JSON last: the desktop side treats the JSON
                # as the record's marker and checks for the companion image,
                # so a record must never appear in the queue image-less.
                _move_across_filesystems(image_src, image_dest)
                try:
                    _move_across_filesystems(json_src, os.path.join(target, f"{record_id}.json"))
                except OSError:
                    # Put the image back so the buffered pair stays intact
                    # and the record retries on the next flush.
                    _move_across_filesystems(image_dest, image_src)
                    raise
                moved += 1
            except OSError as e:
                logger.warning("Could not flush record %s: %s", record_id, e)
        return moved
