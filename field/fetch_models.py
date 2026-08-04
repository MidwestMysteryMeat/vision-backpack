"""
fetch_models.py

Fetches the three model files the field unit needs into field/models/,
matching the paths already set in config.yaml:

    models/haarcascade_frontalface_default.xml   (face detection)
    models/mobilenet_ssd_coco.tflite             (object tagging)
    models/movenet_lightning.tflite              (gait pose estimation)

The Haar cascade is copied from the installed OpenCV package (it ships
with opencv-python), falling back to the OpenCV GitHub mirror. The two
TFLite models are downloaded from their published hosting. Run once on
the Pi (or anywhere, then copy models/ over):

    cd field
    python fetch_models.py

Safe to re-run: files that already exist are skipped unless --force.
"""

import argparse
import io
import logging
import os
import shutil
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("vb.models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
               "data/haarcascades/haarcascade_frontalface_default.xml")

# Classic quantized MobileNet-SSD COCO export; the zip's detect.tflite is
# the model this project's SSD output decoding was written against.
SSD_ZIP_URL = ("https://storage.googleapis.com/download.tensorflow.org/models/"
               "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")
SSD_NAME = "mobilenet_ssd_coco.tflite"

MOVENET_URL = ("https://tfhub.dev/google/lite-model/movenet/singlepose/"
               "lightning/3?lite-format=tflite")
MOVENET_NAME = "movenet_lightning.tflite"


def _download(url: str) -> bytes:
    logger.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "vision-backpack-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_cascade(dest: str):
    try:
        import cv2
        bundled = os.path.join(cv2.data.haarcascades, CASCADE_NAME)
        if os.path.isfile(bundled):
            shutil.copyfile(bundled, dest)
            logger.info("Copied Haar cascade from installed OpenCV.")
            return
    except (ImportError, AttributeError):
        pass
    with open(dest, "wb") as f:
        f.write(_download(CASCADE_URL))
    logger.info("Downloaded Haar cascade.")


def fetch_ssd(dest: str):
    payload = _download(SSD_ZIP_URL)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tflite")]
        if not names:
            raise RuntimeError("No .tflite file inside the SSD model zip")
        with zf.open(names[0]) as src, open(dest, "wb") as f:
            shutil.copyfileobj(src, f)
    logger.info("Extracted %s from SSD zip.", os.path.basename(dest))


def fetch_movenet(dest: str):
    with open(dest, "wb") as f:
        f.write(_download(MOVENET_URL))
    logger.info("Downloaded MoveNet Lightning.")


def main():
    parser = argparse.ArgumentParser(description="Fetch field-unit model files")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    jobs = [
        (CASCADE_NAME, fetch_cascade),
        (SSD_NAME, fetch_ssd),
        (MOVENET_NAME, fetch_movenet),
    ]
    failures = 0
    for name, fetch in jobs:
        dest = os.path.join(MODELS_DIR, name)
        if os.path.isfile(dest) and not args.force:
            logger.info("%s already present, skipping (use --force to refresh).", name)
            continue
        tmp = dest + ".part"
        try:
            fetch(tmp)
            os.replace(tmp, dest)
            logger.info("%s -> %s (%d bytes)", name, dest, os.path.getsize(dest))
        except Exception as e:
            failures += 1
            logger.error("Failed to fetch %s: %s", name, e)
            if os.path.exists(tmp):
                os.remove(tmp)

    if failures:
        raise SystemExit(f"{failures} model(s) failed to fetch; see log above.")
    logger.info("All models in place.")


if __name__ == "__main__":
    main()
