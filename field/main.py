"""
main.py (field)

Runs on the worn Pi 5. Orchestrates the capture loop:

    every N seconds (or on button press):
        grab frame
        anonymize faces (before anything touches disk)
        tag objects (coarse, on-device)
        get GPS fix (best-effort, non-blocking beyond timeout)
        write record to queue

Kept simple and low-compute on purpose: all the heavy lifting
(lore generation, mapping) happens later on the desktop side.
"""

import logging
import time
import yaml
import cv2

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("vb.main")

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logger.info("RPi.GPIO not available, running in desktop simulation mode "
                "(no button/LED support).")

from anonymizer import FaceAnonymizer
from object_mapper import ObjectMapper
from gait_estimator import GaitEstimator
from gps_logger import GPSLogger
from queue_writer import QueueWriter


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_gpio(button_pin: int, led_pin: int):
    if not HAS_GPIO:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(led_pin, GPIO.OUT)


def blink_led(led_pin: int, times: int = 1, duration: float = 0.1):
    if not HAS_GPIO:
        return
    for _ in range(times):
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(duration)


def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    button_pin = cfg["capture"]["manual_button_gpio"]
    led_pin = cfg["capture"]["status_led_gpio"]
    setup_gpio(button_pin, led_pin)

    anon_cfg = cfg["anonymization"]
    cascade_paths = anon_cfg.get("face_cascade_paths") or [anon_cfg["face_cascade_path"]]
    anonymizer = FaceAnonymizer(
        cascade_path=cascade_paths,
        min_face_size_px=anon_cfg["min_face_size_px"],
    )
    object_mapper = ObjectMapper(
        model_path=cfg["object_tagging"]["model_path"],
        confidence_threshold=cfg["object_tagging"]["confidence_threshold"],
        fantasy_map_path=cfg["object_tagging"]["fantasy_map_path"],
    )
    gait_cfg = cfg.get("gait", {})
    gait_estimator = None
    if gait_cfg.get("enabled", False):
        gait_estimator = GaitEstimator(
            model_path=gait_cfg["model_path"],
            confidence_threshold=gait_cfg.get("confidence_threshold", 0.3),
            thresholds=gait_cfg.get("thresholds"),
        )
    gps = GPSLogger(
        port=cfg["gps"]["serial_port"],
        baud_rate=cfg["gps"]["baud_rate"],
        fix_timeout_s=cfg["gps"]["fix_timeout_s"],
    )
    queue = QueueWriter(
        local_buffer_dir=cfg["storage"]["local_buffer_dir"],
        external_drive_mount=cfg["storage"]["external_drive_mount"],
        queue_subdir=cfg["storage"]["queue_subdir"],
        flush_on_capture=cfg["storage"]["flush_to_drive_on_capture"],
    )

    cap = cv2.VideoCapture(cfg["camera"]["device_index"])
    # resolution lives under capture: in the shipped config.yaml
    resolution = cfg["capture"]["resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    if not cap.isOpened():
        raise RuntimeError("Could not open camera device.")

    interval = cfg["capture"]["interval_seconds"]
    logger.info("Field capture running. Auto-interval: %ss. Ctrl+C to stop.", interval)

    last_capture_time = 0.0
    last_flush_time = 0.0
    button_was_down = False
    try:
        while True:
            now = time.monotonic()
            # Edge-triggered: fire once per press, not once per loop tick
            # while the button is held down.
            button_down = HAS_GPIO and GPIO.input(button_pin) == GPIO.LOW
            button_pressed = button_down and not button_was_down
            button_was_down = button_down
            due_for_auto_capture = (now - last_capture_time) >= interval

            if button_pressed or due_for_auto_capture:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                anonymized = anonymizer.anonymize(frame)
                # Tag the original frame, not the masked copy
                object_tags = object_mapper.tag_frame(frame)
                lat, lon = gps.get_fix()

                gait_descriptor = None
                if gait_estimator is not None and any(
                        t.real_label == "person" for t in object_tags):
                    # Only spend the extra burst/inference cost when a
                    # person is actually in frame. Burst frames live only
                    # in this local list, discarded once describe_burst()
                    # returns, per the ephemeral-descriptor design.
                    burst = []
                    for _ in range(gait_cfg.get("burst_frame_count", 15)):
                        ok, burst_frame = cap.read()
                        if ok:
                            burst.append(burst_frame)
                    gait_descriptor = gait_estimator.describe_burst(burst)
                    del burst  # explicit: nothing from the burst survives this scope

                queue.write_record(anonymized, lat, lon, object_tags, gait_descriptor)
                blink_led(led_pin, times=1)

                last_capture_time = now

            # Periodically try to flush any locally buffered records if the
            # drive has since been mounted (e.g. reconnected mid-hike).
            # Throttled: no need to hit the mount point every loop tick.
            if (now - last_flush_time) >= 10.0:
                queue.flush_buffer_to_drive()
                last_flush_time = now

            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        cap.release()
        gps.close()
        if HAS_GPIO:
            GPIO.cleanup()


if __name__ == "__main__":
    run()
