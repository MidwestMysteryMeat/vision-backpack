"""End-to-end desktop pipeline test: queue records in, SQLite world out.

Runs the real desktop main.run() against a temp drive layout with the
LLM stubbed, then runs a second session in a different area to prove
zones accumulate in the world database instead of overwriting.
"""

import importlib.util
import json
import os
import contextlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "desktop"))


def _load_desktop_main():
    path = os.path.join(os.path.dirname(__file__), "..", "desktop", "main.py")
    spec = importlib.util.spec_from_file_location("desktop_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desktop_main = _load_desktop_main()


def _fake_generate(self, fantasy_tags, zone_id, center_lat, center_lon,
                   source_timestamps, gait_descriptors=None):
    from schema import NPCRecord
    npc = NPCRecord(name=f"NPC of {zone_id}", role="wanderer",
                    description="test", zone_id=zone_id)
    return (f"Zone at {zone_id}", [npc], [], [])


class PipelineTests(unittest.TestCase):
    def _write_record(self, queue_dir: Path, name: str, lat, lon, ts=1000.0):
        img = queue_dir / f"{name}.jpg"
        # Minimal valid JPEG isn't needed; the loader only checks existence
        img.write_bytes(b"\xff\xd8\xff\xd9")
        (queue_dir / f"{name}.json").write_text(json.dumps({
            "timestamp": ts,
            "gps_lat": lat,
            "gps_lon": lon,
            "object_tags": [{"real_label": "dog", "fantasy_label": "hound",
                             "confidence": 0.8}],
            "gait_descriptor": None,
            "image_path": f"{name}.jpg",
        }))

    def _write_config(self, root: Path) -> str:
        cfg = f"""
input:
  drive_queue_dir: "{(root / 'queue').as_posix()}/"
  processed_dir: "{(root / 'processed').as_posix()}/"
  no_gps_dir: "{(root / 'no_gps').as_posix()}/"
  quarantine_dir: "{(root / 'quarantine').as_posix()}/"
llm:
  provider: "local_llama_server"
  local_llama_server: {{host: "http://unused", timeout_s: 1}}
lore_generation:
  output_dir: "{(root / 'lore').as_posix()}/"
map_builder:
  cluster_radius_meters: 150
  output_dir: "{(root / 'maps').as_posix()}/"
  map_image_size: [400, 300]
mmo_export:
  sqlite_db_path: "{(root / 'world.db').as_posix()}"
"""
        path = root / "config.yaml"
        path.write_text(cfg)
        return str(path)

    def test_sessions_accumulate_in_world_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue"
            queue.mkdir()
            cfg_path = self._write_config(root)

            with mock.patch.object(desktop_main.LoreGenerator,
                                   "generate_for_cluster", _fake_generate):
                # Session 1: two records in one area, one with no GPS fix
                self._write_record(queue, "a1", 41.5872, -87.6521)
                self._write_record(queue, "a2", 41.5873, -87.6522)
                self._write_record(queue, "nofix", None, None)
                desktop_main.run(cfg_path)

                # Session 2: a different area ~2.5km away
                self._write_record(queue, "b1", 41.6100, -87.6521, ts=2000.0)
                desktop_main.run(cfg_path)

            with contextlib.closing(sqlite3.connect(root / "world.db")) as conn:
                zones = conn.execute("SELECT zone_id FROM zones").fetchall()
                npcs = conn.execute("SELECT zone_id FROM npcs").fetchall()

            # Both sessions' zones coexist; session 2 did not clobber session 1
            self.assertEqual(len(zones), 2)
            self.assertEqual(len(npcs), 2)

            # Queue fully drained; records routed to the right places
            self.assertEqual(list(queue.glob("*.json")), [])
            self.assertEqual(len(list((root / "processed").glob("*.json"))), 3)
            self.assertEqual(len(list((root / "no_gps").glob("*.json"))), 1)
            self.assertTrue((root / "maps" / "session_map.png").exists())

    def test_reprocessing_same_area_replaces_not_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue"
            queue.mkdir()
            cfg_path = self._write_config(root)

            with mock.patch.object(desktop_main.LoreGenerator,
                                   "generate_for_cluster", _fake_generate):
                self._write_record(queue, "a1", 41.5872, -87.6521)
                desktop_main.run(cfg_path)
                # Walk the same spot again
                self._write_record(queue, "a3", 41.5872, -87.6521, ts=3000.0)
                desktop_main.run(cfg_path)

            with contextlib.closing(sqlite3.connect(root / "world.db")) as conn:
                zones = conn.execute("SELECT zone_id FROM zones").fetchall()
                npcs = conn.execute("SELECT zone_id FROM npcs").fetchall()
            self.assertEqual(len(zones), 1)
            self.assertEqual(len(npcs), 1)  # regenerated, not stacked

    def test_malformed_record_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / "queue"
            queue.mkdir()
            (queue / "bad.json").write_text("{not json")
            self._write_record(queue, "good", 41.5872, -87.6521)

            loaded = desktop_main.load_queued_records(
                str(queue), str(root / "quarantine"))

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0][0], "good.json")
            self.assertTrue((root / "quarantine" / "bad.json").exists())


if __name__ == "__main__":
    unittest.main()
