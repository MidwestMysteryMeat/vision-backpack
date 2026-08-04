"""
schema.py

Shared structures for the lore generation output. Kept as plain dataclasses
with to_dict() so both lore_generator.py and any SQLite export step can
serialize consistently.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class NPCRecord:
    name: str
    role: str                      # e.g. "merchant", "guard", "wanderer"
    description: str
    source_fantasy_tags: List[str] = field(default_factory=list)
    source_gait_traits: List[str] = field(default_factory=list)
    zone_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LocationFixture:
    name: str
    fixture_type: str              # e.g. "keep", "cottage", "ward line"
    description: str
    gps_lat: Optional[float]
    gps_lon: Optional[float]
    zone_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LoreFragment:
    title: str
    text: str
    related_zone_id: Optional[str] = None
    source_record_timestamps: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Zone:
    zone_id: str
    name: str
    center_lat: float
    center_lon: float
    npcs: List[NPCRecord] = field(default_factory=list)
    fixtures: List[LocationFixture] = field(default_factory=list)
    lore: List[LoreFragment] = field(default_factory=list)

    def to_dict(self):
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "npcs": [n.to_dict() for n in self.npcs],
            "fixtures": [f.to_dict() for f in self.fixtures],
            "lore": [frag.to_dict() for frag in self.lore],
        }
