"""
object_mapper.py

Runs a lightweight on-device object detector (MobileNet-SSD, COCO classes,
same model class used in the wildlife camera project) and translates
detected real-world object labels into their fantasy equivalents using a
configurable mapping table.

This stays coarse on purpose. The goal at capture time is tagging, not
scene understanding. The heavy lore-writing work happens later on the
desktop with a real LLM.
"""

import logging
import cv2
import numpy as np
import yaml
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("vb.objects")


def load_tflite_interpreter(model_path: str):
    """Loads a TFLite interpreter from whichever runtime is installed:
    tflite_runtime (the Pi deployment target), ai-edge-litert (its
    successor package), or full TensorFlow (desktop testing)."""
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=model_path)


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

# MobileNet-SSD COCO class IDs are 0-based against the model zip's
# labelmap.txt with its leading "???" line removed (so class 0 = person).
# This is the 91-class COCO paper label space, which has holes ("???")
# where classes were dropped from the dataset; detections landing on a
# hole are discarded. Verified empirically against
# coco_ssd_mobilenet_v1_1.0_quant_2018_06_29 output.
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "???", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "???", "backpack", "umbrella",
    "???", "???", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "???",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "???",
    "dining table", "???", "???", "toilet", "???", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "???", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def load_fantasy_map(path: str = None) -> dict:
    if path is None:
        return DEFAULT_FANTASY_MAP
    try:
        with open(path, "r") as f:
            custom_map = yaml.safe_load(f) or {}
        merged = dict(DEFAULT_FANTASY_MAP)
        if not isinstance(custom_map, dict):
            raise ValueError(f"Fantasy map must be a mapping: {path}")
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
        self._warned_not_implemented = False
        try:
            self.model = load_tflite_interpreter(model_path)
            self.model.allocate_tensors()
        except Exception as e:
            logger.warning("Model not loaded (%s). Tagging disabled, records "
                           "will have empty object_tags until a model is in place.", e)

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

        try:
            input_detail = self.model.get_input_details()[0]
            output_details = self.model.get_output_details()
            _, height, width, _ = input_detail["shape"]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(width), int(height)))
            tensor = np.expand_dims(image, axis=0)
            dtype = input_detail["dtype"]
            if dtype == np.float32:
                tensor = tensor.astype(np.float32) / 255.0
            else:
                tensor = tensor.astype(dtype)
            self.model.set_tensor(input_detail["index"], tensor)
            self.model.invoke()

            tensors = [self.model.get_tensor(d["index"]) for d in output_details]
            boxes = next((t for t in tensors if t.ndim >= 3 and t.shape[-1] == 4), None)
            vectors = [t.reshape(-1) for t in tensors if t is not boxes]
            if boxes is None or len(vectors) < 2:
                raise ValueError("Unsupported MobileNet-SSD output tensors")
            boxes = boxes.reshape(-1, 4)
            # Standard SSD exports provide scores and class IDs as the two
            # same-length vectors. The optional detection count is ignored.
            score_candidates = [v for v in vectors
                                if len(v) == len(boxes)
                                and np.all(np.isfinite(v))
                                and float(np.min(v)) >= -1e-6
                                and float(np.max(v)) <= 1.000001]
            if not score_candidates:
                raise ValueError("No score tensor found in MobileNet-SSD outputs")
            def score_likelihood(vector):
                fractional = np.any(np.abs(vector - np.round(vector)) > 1e-5)
                return (bool(fractional), len(vector))
            scores = max(score_candidates, key=score_likelihood)
            candidates = [v for v in vectors if v is not scores and len(v) == len(scores)]
            classes = candidates[0] if candidates else np.zeros(len(scores))
            results = []
            for box, score, class_id in zip(boxes, scores, classes, strict=False):
                score = float(score)
                if score < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = max(0, min(frame.shape[1], int(x1 * frame.shape[1])))
                y1 = max(0, min(frame.shape[0], int(y1 * frame.shape[0])))
                x2 = max(x1, min(frame.shape[1], int(x2 * frame.shape[1])))
                y2 = max(y1, min(frame.shape[0], int(y2 * frame.shape[0])))
                label_index = int(class_id)
                label = (COCO_LABELS[label_index]
                         if 0 <= label_index < len(COCO_LABELS)
                         else f"class_{label_index}")
                if label == "???":
                    continue  # hole in the COCO label space, not a real class
                results.append(TaggedObject(
                    real_label=label,
                    fantasy_label=self._to_fantasy_label(label),
                    confidence=score,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                ))
            return results
        except Exception as exc:
            logger.warning("Inference failed; omitting tags: %s", exc)
            return []
