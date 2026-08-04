"""
map_builder.py

Clusters GPS-tagged records into named zones based on physical proximity,
then renders a simple 2D map preserving the real walked route's geography
translated into fantasy zone markers.

Uses simple distance-based clustering rather than a full DBSCAN dependency.
Good enough at the scale of "one person's walking route," and it keeps the
desktop-side dependency list light.
"""

import math
from dataclasses import dataclass
from typing import List, Dict
from PIL import Image, ImageDraw


@dataclass
class RawRecord:
    timestamp: float
    gps_lat: float
    gps_lon: float
    fantasy_tags: List[str]
    gait_descriptors: List[str] = None

    def __post_init__(self):
        if self.gait_descriptors is None:
            self.gait_descriptors = []


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def stable_zone_id(center_lat: float, center_lon: float, radius_meters: float) -> str:
    """Derives a zone ID from geographic position rather than run order.

    Sequential IDs (zone_0000, zone_0001, ...) restart every session, so a
    second walk's zones overwrite the first walk's rows in the world
    database. Quantizing the centroid to a grid keyed by cluster radius
    gives the same place the same ID across sessions: revisiting an area
    regenerates that area, while new areas accumulate alongside old ones.
    """
    cell_deg = max(radius_meters, 1.0) * 2 / 111_320  # ~meters per degree latitude
    qlat = round(center_lat / cell_deg)
    qlon = round(center_lon / cell_deg)
    return f"zone_{qlat}x{qlon}"


def cluster_records(records: List[RawRecord], radius_meters: float) -> List[List[RawRecord]]:
    """Simple greedy proximity clustering: walk the list, assign each record
    to the first existing cluster within radius of its centroid, else start
    a new cluster."""
    clusters: List[List[RawRecord]] = []

    for record in records:
        if record.gps_lat is None or record.gps_lon is None:
            continue  # skip records with no GPS fix

        placed = False
        for cluster in clusters:
            centroid_lat = sum(r.gps_lat for r in cluster) / len(cluster)
            centroid_lon = sum(r.gps_lon for r in cluster) / len(cluster)
            if haversine_meters(record.gps_lat, record.gps_lon,
                                 centroid_lat, centroid_lon) <= radius_meters:
                cluster.append(record)
                placed = True
                break

        if not placed:
            clusters.append([record])

    return clusters


def render_map(zones: Dict[str, dict], image_size: tuple, output_path: str,
               route: List[tuple] = None):
    """
    Renders a simple top-down map: zone centroids plotted proportionally
    to their real GPS spread, labeled with fantasy zone names, with the
    walked route drawn as a path underneath. Uses an equirectangular
    projection (longitude scaled by cos(latitude)) with a uniform scale on
    both axes so the layout matches real geography instead of stretching
    to fill the canvas.

    route: optional list of (lat, lon) points in time order.
    """
    if not zones:
        return

    route = route or []
    points = ([(z["center_lat"], z["center_lon"]) for z in zones.values()]
              + [(lat, lon) for lat, lon in route])
    lats = [p[0] for p in points]
    mid_lat = (min(lats) + max(lats)) / 2
    lon_scale = math.cos(math.radians(mid_lat))

    xs = [p[1] * lon_scale for p in points]
    ys = lats
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width, height = image_size
    margin = 100
    # One scale for both axes, whichever dimension is the binding one
    scale = min((width - 2 * margin) / max(max_x - min_x, 1e-9),
                (height - 2 * margin) / max(max_y - min_y, 1e-9))
    # Center the drawn extent on the canvas
    off_x = (width - (max_x - min_x) * scale) / 2
    off_y = (height - (max_y - min_y) * scale) / 2

    def project(lat, lon):
        x = off_x + (lon * lon_scale - min_x) * scale
        y = off_y + (max_y - lat) * scale
        return (x, y)

    img = Image.new("RGB", (width, height), color=(20, 24, 20))
    draw = ImageDraw.Draw(img)

    if len(route) >= 2:
        draw.line([project(lat, lon) for lat, lon in route],
                  fill=(90, 80, 55), width=3)

    for zone_id, zone in zones.items():
        x, y = project(zone["center_lat"], zone["center_lon"])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(180, 140, 60))
        draw.text((x + 12, y - 6), zone.get("name", zone_id), fill=(230, 220, 190))

    img.save(output_path)
