# RPG Vision Backpack - Spec Sheet

**Version:** 0.1 (concept spec)
**Status:** Pre-prototype planning

## 1. System Overview

A wearable field-capture rig that turns real-world surroundings into fantasy
MMO content. Two-phase architecture:

- **Phase 1, Field capture (worn):** a Raspberry Pi 5 (16GB) with camera
  does on-device processing (anonymization, coarse object tagging, and a
  lightweight ephemeral gait descriptor, see §7) and writes
  timestamped, GPS-tagged records to a queue on an external drive.
  Upgraded from an original Pi Zero W concept specifically to support
  on-device pose estimation for gait. The accepted tradeoff is short
  session length (roughly 2-3 hrs) rather than all-day runtime; see §5.
- **Phase 2, Desktop processing (home):** the drive gets plugged into a
  local GPU-capable machine, where a local LLM via llama-server (or Ollama)
  turns the raw tagged captures into structured JSON: NPCs, locations, and
  lore fragments that feed directly into the MMO's world database.

**Core privacy rule, non-negotiable:** no raw identifiable image data is
ever stored or transmitted. Faces are detected and replaced with procedural
fantasy overlays at capture time, on-device, before anything is written
to disk. Nothing identifying leaves the field unit. This extends to
gait too: pose data is processed in-memory into a short descriptive text
label (e.g. "brisk, rigid stride"), and the underlying keypoint data is
discarded immediately. No gait signature is ever stored or matched against
later. This is a biometric identifier just like a face, and it gets the
same discard-immediately rule; see §7 for why storing a matchable gait
signature would defeat the purpose of the anonymizer entirely.

---

## 2. Hardware - Bill of Materials

Prices are rough 2026 estimates (USD) and should be treated as ballpark figures, not quotes: verify current prices before ordering, since component costs and part availability shift over time and these numbers may already be out of date.

### 2.1 Field unit (worn), Pi 5 16GB build

| Component | Example part | Est. cost |
|---|---|---|
| Compute | Raspberry Pi 5, 16GB | $120 |
| Active cooling | Official Pi 5 Active Cooler (fan + heatsink) | $5 |
| Camera | Pi Camera Module 3 (or NoIR for low-light) | $25-35 |
| Microphone (optional, ambient audio tagging) | USB lavalier mic or I2S mic HAT | $10-15 |
| GPS module | NEO-6M GPS module (UART) | $10 |
| Battery | 20,000mAh USB-C PD power bank (65W+ output) | $45-60 |
| Storage | 4TB portable external SSD/HDD (shared, not per-unit) | $80-120 |
| microSD (OS + local queue buffer) | 64GB A2 | $10 |
| Enclosure/mount | Larger weatherproof project box + backpack strap mount (bulkier than a Zero-class build) | $20-30 |
| Physical start/stop button | GPIO tactile button | $2 |
| Status LED | GPIO LED | $1 |
| **Field unit total** | | **~$318-388** (excluding shared drive) |

*Gait pose model (MoveNet Lightning or similar, tflite) is free/open weights: no added hardware cost, just the compute headroom the Pi 5 provides.*

### 2.2 Desktop processing station

Assumes an existing desktop with a GPU (RTX 3070 Ti class or better) is
already available, so no dedicated purchase is needed if reusing existing
hardware. If building from scratch, budget accordingly for a GPU-capable
machine; that's outside this spec's scope since it's shared infrastructure.

| Component | Notes |
|---|---|
| GPU-capable desktop | Existing hardware, RTX 3070 Ti or similar (8GB+ VRAM) |
| LLM inference | Local (Ollama + 7B-13B model) or API-based (Anthropic API), configurable |
| External drive dock/reader | To ingest the field unit's storage drive | $15-20 |

---

## 3. Software Architecture

```
vision_backpack/
├── field/                      # Runs on the Pi 5 (worn unit)
│   ├── main.py                 # Capture loop orchestration
│   ├── anonymizer.py           # On-device face detection + fantasy overlay
│   ├── object_mapper.py        # Real-world object -> fantasy equivalent tagging
│   ├── gps_logger.py           # NEO-6M GPS interface
│   ├── queue_writer.py         # Writes timestamped records to external drive
│   └── config.yaml
├── desktop/                    # Runs on the home desktop (Phase 2)
│   ├── main.py                 # Batch processing orchestration
│   ├── lore_generator.py       # LLM call: raw tags -> structured NPC/location/lore JSON
│   ├── map_builder.py          # Clusters GPS points into named fantasy zones, renders map
│   ├── schema.py               # Shared JSON schema for NPCs/locations/lore
│   └── config.yaml
├── requirements-field.txt      # Minimal deps for the Pi 5 field unit
├── requirements-desktop.txt    # Full deps including LLM client libs
└── README.md
```

### 3.1 Field-side pipeline (real-time, on-device, lightweight)

1. Periodic capture trigger (configurable interval, e.g. every 30-60s, or
   button-triggered for manual "capture this" moments).
2. Lightweight object detector (MobileNet-SSD COCO class, same model class
   used in the wildlife camera project) tags what's in frame. No heavy
   scene understanding is needed here, just coarse labels.
3. Face detector (Haar cascade or a small on-device face model, cheap
   enough to run in real time) locates any faces in frame.
4. Detected face regions get overlaid with a procedurally generated
   fantasy mask/pattern (not a photo-real replacement; the point is
   irreversible anonymization, not deepfaking) before the frame is ever
   written to disk.
5. Record written to the queue: `{timestamp, gps_lat, gps_lon, object_tags[],
   anonymized_image_path}`. No raw unmodified frame is ever persisted.

### 3.2 Desktop-side pipeline (batch, heavy compute, runs later at home)

1. Read all queued records from the external drive.
2. For each record (or clustered batch of nearby records), call an LLM
   (local via Ollama, or the Anthropic API) with the object tags and
   coarse scene description to generate structured lore: NPC descriptions,
   location fixtures, quest hooks, matching the schema your existing MMO
   world-database work already uses (SQLite export, quest hook system,
   location fixtures).
3. GPS points get spatially clustered (grid-based or simple distance
   clustering) into named zones, preserving real-world geographic
   relationships while translating them into fantasy names/content.
4. Output: structured JSON per zone, plus a rendered fantasy map image
   that mirrors the real walked route.

---

## 4. Implementation Plan (phased)

| Phase | Goal | Exit criteria |
|---|---|---|
| 1. Field capture loop | Pi 5 captures + GPS-tags images on interval, no anonymization yet | Reliable capture over a test walk, battery life measured |
| 2. On-device anonymization | Face detection + fantasy overlay running in the capture loop | No raw faces ever hit disk, verified by manual review |
| 3. Object tagging | MobileNet-SSD tagging integrated, coarse labels attached to each record | Reasonable tag accuracy on test footage |
| 4. Gait descriptor | MoveNet pose burst -> text label, keypoints discarded immediately | Manual review confirms no pose/keypoint data ever persists to disk |
| 5. Queue + drive handoff | Records reliably written to external drive queue format | Desktop can read and parse a full field session |
| 6. Desktop lore generation | Local LLM turns tagged records + gait labels into structured NPC/location JSON | Output matches existing MMO world-database schema |
| 7. Map builder | GPS clustering + named zone generation + rendered map | Map visually reflects the real walked route |
| 8. Field hardening | Battery swap workflow, cooling validation, enclosure weatherproofing, button/LED UX | 2-3 hr field session survives without intervention |

---

## 5. Power Budget (field unit)

| Config | Draw | 20,000mAh bank runtime (approx) |
|---|---|---|
| Pi 5 (16GB) + camera + GPS, gait estimation active, no mic | ~6-8W @ 5V (idle-to-load average) | ~2-2.5 hrs |
| + microphone + heavier tagging load | ~7-9W @ 5V | ~1.5-2 hrs |

This is the accepted tradeoff of the Pi 5 upgrade: on-device pose
estimation for gait descriptors was prioritized over all-day runtime.
Practical implication: this is a short-session field unit (a scouting
walk, a couple hours out), not an all-day wearable. Bring a spare 20,000mAh
bank (or two) if a session needs to run longer, and swap rather than trying
to recharge in the field.

Solar isn't practical for this build. The Pi 5's 6-9W active draw needs a
30-40W panel and USB-C PD negotiation that doesn't fit a backpack form
factor. If all-day solar-assisted operation ever becomes the priority
again, the original Pi Zero 2 W path (no gait, motion/object tagging only)
is the fallback; see version history in this repo.

Active cooling (fan) draws additional current under sustained load, so
factor that into the runtime estimates above; they already assume the fan
is running.

---

## 6. Privacy / Ethics Checklist

- [ ] Verify anonymization happens before any disk write, not after. No raw frame should ever exist even transiently.
- [ ] Confirm anonymization is irreversible (procedural overlay, not a reversible blur/pixelation that could theoretically be undone).
- [ ] Confirm gait descriptors are text labels only. No keypoint arrays, pose vectors, or anything matchable against a future capture should ever be written to disk (see §7).
- [ ] No audio recording of identifiable speech content if mic is enabled. Ambient tagging only, not transcription of bystander conversations.
- [ ] Be mindful of local wiretapping/recording consent laws if audio capture is enabled in public spaces (varies by state; some require all-party consent).
- [ ] GPS data should be treated as sensitive. Encrypt at rest on the field drive if there's any chance of loss/theft.

---

## 7. Gait Descriptor Design (why it's ephemeral-only)

Gait is a biometric identifier, and in some ways a stronger one than a face
for this project's context, since it works at a distance and can't be
masked the way a face region can. Storing a matchable gait signature would
mean the same real bystander walking by on two different sessions could be
re-identified and linked, which defeats the entire point of the
anonymizer. That path is explicitly out of scope for this project.

**What actually gets built instead:**

1. On detecting a person in frame, capture a short burst (roughly 15
   frames, 1-2 seconds) rather than a single still.
2. Run a lightweight pose model (MoveNet Lightning, tflite, feasible in
   real time on the Pi 5) across the burst to get per-frame keypoints.
3. Derive simple, coarse metrics from the keypoint sequence in-memory:
   apparent walking speed (ankle/hip displacement across frames), stride
   regularity (variance in step timing), and posture (torso angle,
   upright vs. hunched).
4. Map those metrics to a small set of descriptive text labels (e.g.
   "brisk and rigid," "slow, shuffling gait," "steady, purposeful
   stride") using simple thresholding. No ML is needed for this step.
5. Discard the burst frames and all keypoint data immediately. Only the
   resulting text label is written to the record.

The text label feeds into the desktop-side lore prompt as flavor
material for NPC generation, the same role the object fantasy-tags
already play, just adding movement/personality texture. It is never
compared against, matched to, or linked with any other record.
