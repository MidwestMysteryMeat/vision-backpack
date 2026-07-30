"""
object_mapper.py

Runs a lightweight on-device object detector (MobileNet-SSD, COCO classes --
same model class used in the wildlife camera project) and translates
detected real-world object labels into their fantasy equivalents using a
configurable mapping table.

This stays deliberately coarse -- the goal at capture time is tagging, not
scene understanding. The heavy lore-writing work happens later on the
desktop with a real LLM.
"""

import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class TaggedObject:
    real_label: str
    fantasy_label: str
    confidence: float
    bbox: tuple


DEFAULT_FANTASY_MAP = {
    "knife": "dagger",
    "dog": "hound",
    "cat": "familiar",
    "building": "keep",
    "house": "cottage",
    "car": "wagon",
    "bicycle": "steed",
    "person": "traveler",
    "backpack": "satchel",
    "bottle": "flask",
    "chair": "throne",
    "bench": "resting stone",
    "tree": "ancient tree",
    "umbrella": "warding canopy",
    "clock": "sundial",
    "fence": "ward line",
    "traffic light": "beacon",
}


def load_fantasy_map(path: str = None) -> dict:
    if path is None:
        return DEFAULT_FANTASY_MAP
    try:
        with open(path, "r") as f:
            custom_map = yaml.safe_load(f) or {}
        merged = dict(DEFAULT_FANTASY_MAP)
        merged.update(custom_map)
        return merged
    except FileNotFoundError:
        return DEFAULT_FANTASY_MAP


class ObjectMapper:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 fantasy_map_path: str = None):
        self.confidence_threshold = confidence_threshold
        self.fantasy_map = load_fantasy_map(fantasy_map_path)
        self.model = None
        try:
            import tflite_runtime.interpreter as tflite
            self.model = tflite.Interpreter(model_path=model_path)
            self.model.allocate_tensors()
        except Exception as e:
            print(f"[ObjectMapper] Model not loaded ({e}). Tagging disabled, "
                  f"records will have empty object_tags until a model is in place.")

    def _to_fantasy_label(self, real_label: str) -> str:
        return self.fantasy_map.get(real_label.lower(), real_label)

    def tag_frame(self, frame) -> List[TaggedObject]:
        """
        Returns coarse object tags for the frame. Placeholder-safe: if no
        model is loaded yet, returns an empty list rather than failing,
        so the capture loop can run end-to-end before a model exists.
        """
        if self.model is None:
            return []

        # Actual TFLite inference wiring goes here -- input tensor prep,
        # invoke(), output tensor parsing. Left as an integration point
        # since exact pre/post-processing depends on the specific
        # MobileNet-SSD export used.
        raise NotImplementedError(
            "Wire up TFLite inference here once a model is exported -- "
            "see the wildlife camera project's camera_detector.py for the "
            "same MobileNet-SSD inference pattern already in use."
        )
