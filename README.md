# RPG Vision Backpack

See `SPEC_SHEET.md` for the full hardware BOM, power budget, architecture,
and phased implementation plan. This README covers software setup for
both halves of the system.

## Field unit (Pi 5, 16GB, worn)

```bash
cd field
pip install -r ../requirements-field.txt --break-system-packages
```

Download a Haar cascade face model (`haarcascade_frontalface_default.xml`,
ships with OpenCV), a MobileNet-SSD COCO TFLite model, and a MoveNet
Lightning TFLite model into `field/models/` (same model class already
used in the wildlife camera project's `camera_detector.py` for object
detection; MoveNet Lightning is the pose model gait descriptors run on).

Set `storage.external_drive_mount` in `field/config.yaml` to wherever the
shared 4TB drive mounts on the Pi, and `gps.serial_port` to match your
NEO-6M wiring (`/dev/serial0` is the default UART on most Pi models).

```bash
python main.py
```

Runs continuously: auto-captures every `capture.interval_seconds`, or on
button press (GPIO pin set in config). Ctrl+C to stop.

**Session length:** this is a short-session build (roughly 2-3 hrs on a
20,000mAh bank). See `SPEC_SHEET.md` §5 for the power tradeoff behind
going with the Pi 5 over the original Pi Zero W concept. Bring spare
battery banks for longer outings.

**Gait descriptors:** when `gait.enabled` is true in config and a person
is detected in frame, a short burst of frames is analyzed on-device to
produce a text descriptor (e.g. "brisk, rigid stride"). This is
ephemeral by design. See `gait_estimator.py`'s module docstring and
`SPEC_SHEET.md` §7 for why raw pose/keypoint data is never persisted or
matched between captures. Only the text label reaches the queue.

**Before field use:** verify anonymization is working correctly. Capture
a test frame with a face in it and confirm the output has a procedural
overlay, not the raw face. Do this every time you change camera hardware
or anonymizer settings, not just once.

## Desktop processing (Phase 2, run at home)

```bash
cd desktop
pip install -r ../requirements-desktop.txt
```

Plug in the field unit's external drive, confirm `input.drive_queue_dir`
in `desktop/config.yaml` points at its mounted `queue/` folder.

Set `llm.provider` to `local_llama_server` (talks to `llama-server`
running on any self-hosted machine over the LAN, no API key needed; set
`llm.local_llama_server.host` to that machine's actual IP) or `ollama`
(requires a local Ollama server running the configured model). For the
local_llama_server path, start `llama-server` there first:

```bash
llama-server -m /path/to/qwen-model.gguf --host 0.0.0.0 --port 8080 -ngl 99
```

```bash
python main.py
```

This clusters the session's GPS-tagged records into zones, generates
NPCs/fixtures/lore per zone via the configured LLM, renders a fantasy map
of the walked route (route path drawn under geographically-scaled zone
markers), and exports everything into a SQLite database matching the
existing MMO world-database schema (zones, NPCs, fixtures, lore; pairs
with the quest hook and location fixture systems from prior
world-building pipeline work).

Zone IDs are derived from geographic position, not run order, so
processing multiple sessions accumulates zones in the world database:
walking a new area adds zones, re-walking a known area regenerates just
those zones.

Processed records get moved out of the drive's active queue automatically
unless you pass `--keep-queue`.

## Project status

Pre-prototype. `object_mapper.py`'s TFLite inference call is a stub, and
so is `gait_estimator.py`'s `_run_pose_model()`. Wire both up once the
respective models are exported (see the wildlife camera project's
`camera_detector.py` for the same inference pattern already working
elsewhere). Everything else runs end-to-end, including fallback paths if
tagging or gait isn't wired up yet: you can test the full capture,
anonymize, GPS, queue, desktop, lore, map, SQLite pipeline with empty
tags and no gait descriptors before either model exists.

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See
`LICENSE`. Copyright (C) 2026 MidwestMysteryMeat. See `NOTICE` for
attribution requirements and the network-use clause.
