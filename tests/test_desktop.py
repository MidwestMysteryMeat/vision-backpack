import os
import sys
import tempfile
import unittest
from pathlib import Path

# desktop modules import each other as top-level modules (from schema import
# ...), matching how they run in production from inside desktop/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "desktop"))

from lore_generator import extract_json_object
from map_builder import (RawRecord, cluster_records, render_map,
                         stable_zone_id)


class LoreParsingTests(unittest.TestCase):
    def test_plain_json_parses(self):
        self.assertEqual(extract_json_object('{"zone_name": "Mistfen"}'),
                         {"zone_name": "Mistfen"})

    def test_fenced_json_parses(self):
        raw = '```json\n{"zone_name": "Mistfen", "npcs": []}\n```'
        self.assertEqual(extract_json_object(raw)["zone_name"], "Mistfen")

    def test_prose_wrapped_json_parses(self):
        raw = 'Here is the content you asked for:\n{"zone_name": "Mistfen"}\nEnjoy!'
        self.assertEqual(extract_json_object(raw)["zone_name"], "Mistfen")

    def test_braces_inside_strings_do_not_break_extraction(self):
        raw = 'note: {"zone_name": "The {Broken} Gate"} trailing'
        self.assertEqual(extract_json_object(raw)["zone_name"], "The {Broken} Gate")

    def test_garbage_returns_none(self):
        self.assertIsNone(extract_json_object("no json here"))
        self.assertIsNone(extract_json_object("[1, 2, 3]"))


class ZoneIdTests(unittest.TestCase):
    def test_same_place_same_id_across_sessions(self):
        a = stable_zone_id(41.5872, -87.6521, 150)
        b = stable_zone_id(41.5872, -87.6521, 150)
        self.assertEqual(a, b)

    def test_nearby_points_share_an_id(self):
        # ~20m apart, well inside a 150m-radius cell
        a = stable_zone_id(41.58720, -87.65210, 150)
        b = stable_zone_id(41.58738, -87.65210, 150)
        self.assertEqual(a, b)

    def test_distant_points_get_different_ids(self):
        a = stable_zone_id(41.5872, -87.6521, 150)
        b = stable_zone_id(41.6100, -87.6521, 150)  # ~2.5km north
        self.assertNotEqual(a, b)


class ClusteringTests(unittest.TestCase):
    def _record(self, lat, lon):
        return RawRecord(timestamp=0.0, gps_lat=lat, gps_lon=lon,
                         fantasy_tags=[])

    def test_two_distant_groups_form_two_clusters(self):
        records = [self._record(41.5872, -87.6521),
                   self._record(41.5873, -87.6522),
                   self._record(41.6100, -87.6521)]
        clusters = cluster_records(records, 150)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(len(c) for c in clusters), [1, 2])


class MapRenderTests(unittest.TestCase):
    def test_renders_map_with_route(self):
        zones = {
            "zone_a": {"name": "Mistfen", "center_lat": 41.5872,
                       "center_lon": -87.6521},
            "zone_b": {"name": "Emberwatch", "center_lat": 41.5900,
                       "center_lon": -87.6480},
        }
        route = [(41.5872, -87.6521), (41.5885, -87.6500), (41.5900, -87.6480)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "map.png"
            render_map(zones, (400, 300), str(out), route=route)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
