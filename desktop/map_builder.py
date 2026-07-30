"""
map_builder.py

Clusters GPS-tagged records into named zones based on physical proximity,
then renders a simple 2D map preserving the real walked route's geography
translated into fantasy zone markers.

Uses simple distance-based clustering rather than a full DBSCAN dependency
-- good enough at the scale of "one person's walking route," and keeps the
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


def render_map(zones: Dict[str, dict], image_size: tuple, output_path: str):
    """
    Renders a simple top-down map: zone centroids plotted proportionally
    to their real GPS spread, labeled with fantasy zone names. This is a
    minimal renderer -- swap in a proper projection/tile system later if
    the walked area gets large enough that flat scaling distorts things.
    """
    if not zones:
        return

    lats = [z["center_lat"] for z in zones.values()]
    lons = [z["center_lon"] for z in zones.values()]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Avoid divide-by-zero for a single-point map
    lat_range = max(max_lat - min_lat, 1e-6)
    lon_range = max(max_lon - min_lon, 1e-6)

    width, height = image_size
    margin = 100
    img = Image.new("RGB", (width, height), color=(20, 24, 20))
    draw = ImageDraw.Draw(img)

    for zone_id, zone in zones.items():
        x = margin + ((zone["center_lon"] - min_lon) / lon_range) * (width - 2 * margin)
        y = margin + (1 - (zone["center_lat"] - min_lat) / lat_range) * (height - 2 * margin)

        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(180, 140, 60))
        draw.text((x + 12, y - 6), zone.get("name", zone_id), fill=(230, 220, 190))

    img.save(output_path)
