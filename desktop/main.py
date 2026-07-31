"""
main.py (desktop)

Phase 2 orchestration. Run this after plugging the field unit's drive in:

    1. Read all queued records from the drive.
    2. Cluster them spatially into zones.
    3. For each zone, call the LLM to generate NPCs/fixtures/lore.
    4. Render a fantasy map of the walked route.
    5. Export everything to the MMO world-database schema (SQLite).
    6. Move processed records out of the queue.
"""

import os
import json
import sqlite3
import argparse
import yaml

from schema import Zone
from lore_generator import LoreGenerator
from map_builder import RawRecord, cluster_records, render_map


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_queued_records(queue_dir: str):
    records = []
    for filename in sorted(os.listdir(queue_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(queue_dir, filename), "r") as f:
            data = json.load(f)

        fantasy_tags = [t["fantasy_label"] for t in data.get("object_tags", [])]
        gait_descriptor = data.get("gait_descriptor")
        records.append((filename, RawRecord(
            timestamp=data["timestamp"],
            gps_lat=data.get("gps_lat"),
            gps_lon=data.get("gps_lon"),
            fantasy_tags=fantasy_tags,
            gait_descriptors=[gait_descriptor] if gait_descriptor else [],
        )))
    return records


def export_to_sqlite(zones: list, db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS zones (
        zone_id TEXT PRIMARY KEY, name TEXT, center_lat REAL, center_lon REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS npcs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT, name TEXT,
        role TEXT, description TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fixtures (
        id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT, name TEXT,
        fixture_type TEXT, description TEXT, gps_lat REAL, gps_lon REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS lore (
        id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT, title TEXT, text TEXT)""")

    for zone in zones:
        conn.execute("INSERT OR REPLACE INTO zones VALUES (?, ?, ?, ?)",
                      (zone.zone_id, zone.name, zone.center_lat, zone.center_lon))
        for npc in zone.npcs:
            conn.execute("INSERT INTO npcs (zone_id, name, role, description) VALUES (?, ?, ?, ?)",
                          (zone.zone_id, npc.name, npc.role, npc.description))
        for fixture in zone.fixtures:
            conn.execute(
                "INSERT INTO fixtures (zone_id, name, fixture_type, description, gps_lat, gps_lon) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (zone.zone_id, fixture.name, fixture.fixture_type, fixture.description,
                 fixture.gps_lat, fixture.gps_lon))
        for lore_frag in zone.lore:
            conn.execute("INSERT INTO lore (zone_id, title, text) VALUES (?, ?, ?)",
                          (zone.zone_id, lore_frag.title, lore_frag.text))

    conn.commit()
    conn.close()


def run(config_path: str = "config.yaml", move_processed: bool = True):
    cfg = load_config(config_path)

    queue_dir = cfg["input"]["drive_queue_dir"]
    filenames_and_records = load_queued_records(queue_dir)
    if not filenames_and_records:
        print("[main] No queued records found. Nothing to process.")
        return

    filenames = [f for f, _ in filenames_and_records]
    records = [r for _, r in filenames_and_records]

    print(f"[main] Loaded {len(records)} records. Clustering...")
    clusters = cluster_records(records, cfg["map_builder"]["cluster_radius_meters"])
    print(f"[main] {len(clusters)} zones identified.")

    llm_cfg = cfg["llm"]
    generator = LoreGenerator(
        provider=llm_cfg["provider"],
        local_llama_server_cfg=llm_cfg.get("local_llama_server"),
        ollama_cfg=llm_cfg.get("ollama"),
    )

    zones = []
    zone_render_data = {}

    for i, cluster in enumerate(clusters):
        zone_id = f"zone_{i:04d}"
        all_tags = [tag for r in cluster for tag in r.fantasy_tags]
        all_gait = [g for r in cluster for g in r.gait_descriptors]
        center_lat = sum(r.gps_lat for r in cluster) / len(cluster)
        center_lon = sum(r.gps_lon for r in cluster) / len(cluster)
        timestamps = [r.timestamp for r in cluster]

        zone_name, npcs, fixtures, lore = generator.generate_for_cluster(
            all_tags, zone_id, center_lat, center_lon, timestamps, gait_descriptors=all_gait
        )

        zone = Zone(zone_id=zone_id, name=zone_name, center_lat=center_lat,
                    center_lon=center_lon, npcs=npcs, fixtures=fixtures, lore=lore)
        zones.append(zone)
        zone_render_data[zone_id] = {"name": zone_name, "center_lat": center_lat,
                                       "center_lon": center_lon}

        print(f"[main] Zone {zone_id}: '{zone_name}', "
              f"{len(npcs)} NPCs, {len(fixtures)} fixtures, {len(lore)} lore fragments")

    # Write raw JSON output
    os.makedirs(cfg["lore_generation"]["output_dir"], exist_ok=True)
    for zone in zones:
        out_path = os.path.join(cfg["lore_generation"]["output_dir"], f"{zone.zone_id}.json")
        with open(out_path, "w") as f:
            json.dump(zone.to_dict(), f, indent=2)

    # Render map
    os.makedirs(cfg["map_builder"]["output_dir"], exist_ok=True)
    map_path = os.path.join(cfg["map_builder"]["output_dir"], "session_map.png")
    render_map(zone_render_data, tuple(cfg["map_builder"]["map_image_size"]), map_path)
    print(f"[main] Map rendered to {map_path}")

    # Export to MMO world database
    export_to_sqlite(zones, cfg["mmo_export"]["sqlite_db_path"])
    print(f"[main] Exported to {cfg['mmo_export']['sqlite_db_path']}")

    # Move processed records out of the active queue
    if move_processed:
        processed_dir = cfg["input"]["processed_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        for filename in filenames:
            base = filename.rsplit(".", 1)[0]
            for ext in (".json", ".jpg"):
                src = os.path.join(queue_dir, base + ext)
                if os.path.exists(src):
                    os.rename(src, os.path.join(processed_dir, base + ext))
        print(f"[main] Moved {len(filenames)} record sets to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Backpack desktop processor")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--keep-queue", action="store_true",
                         help="Don't move records out of the queue after processing")
    args = parser.parse_args()
    run(args.config, move_processed=not args.keep_queue)
