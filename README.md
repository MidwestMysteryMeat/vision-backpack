# RPG Vision Backpack

See `SPEC_SHEET.md` for the full hardware BOM, power budget, architecture,
and phased implementation plan. This README covers software setup for
both halves of the system.

## Field unit (Pi 5 16GB worn; Pi 4 4GB budget build also supported)

```bash
cd field
pip install -r ../requirements-field.txt --break-system-packages
```

Fetch the model files (frontal + profile Haar cascades for faces,
MobileNet-SSD COCO for object tagging, MoveNet Lightning for gait pose)
into `field/models/`:

```bash
python fetch_models.py
```

The cascade is copied from the installed OpenCV package when possible;
the two TFLite models are downloaded from their published hosting.
Re-running skips files already present (`--force` to refresh).

Set `storage.external_drive_mount` in `field/config.yaml` to wherever the
shared 4TB drive mounts on the Pi, and `gps.serial_port` to match your
NEO-6M wiring (`/dev/serial0` is the default UART on most Pi models).

```bash
python main.py
```

Runs continuously: auto-captures every `capture.interval_seconds`, or on
button press (GPIO pin set in config). Ctrl+C to stop.

**Session length:** the Pi 5 is a short-session build (roughly 2-3 hrs
on a 20,000mAh bank); the Pi 4 4GB budget build stretches the same bank
to roughly 3.5-4.5 hrs at the cost of slower inference per capture. See
`SPEC_SHEET.md` §5 for the power tradeoff and §2.1b for the Pi 4 build.
Bring spare battery banks for longer outings.

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

Pre-prototype, but the software pipeline is complete: object tagging,
gait pose, and face detection have all been validated against the real
model files fetched by `fetch_models.py` (person/chair/ball detection,
gait descriptor generation, and face masking confirmed on test imagery).
TFLite inference falls back from `tflite_runtime` (the Pi target) to
`ai-edge-litert` or full TensorFlow, so the field code can be exercised
on a desktop too. Fallback paths still hold if a model file is missing:
capture, anonymize, GPS, queue, desktop, lore, map, and SQLite export
all run with empty tags and no gait descriptors.

What's left is hardware: Pi bring-up (camera, GPS wiring, drive mount),
the on-camera anonymization verification described above, gait threshold
tuning on real captures, and SPEC_SHEET §4 phase 8 field hardening.

Run the test suite from the repo root:

```bash
python -m unittest discover -s tests
```

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See
`LICENSE`. Copyright (C) 2026 MidwestMysteryMeat. See `NOTICE` for
attribution requirements and the network-use clause.
