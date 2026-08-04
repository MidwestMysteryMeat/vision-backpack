"""
gait_estimator.py

Produces a short, human-readable gait descriptor ("brisk and rigid",
"slow, shuffling gait") from a burst of frames, and nothing else.

CRITICAL PRIVACY PROPERTY: this module never returns and never persists
raw keypoints, pose vectors, or anything that could be compared against a
future capture to re-identify the same person. The burst frames and all
intermediate pose data exist only for the duration of describe_burst() and
are discarded when it returns. Only the resulting text label leaves this
module. Do not modify this module to return or cache keypoint data: that
would turn an ephemeral descriptor into a biometric signature, which is
explicitly out of scope for this project (see SPEC_SHEET.md section 7).
"""

import logging
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("vb.gait")


# MoveNet Lightning keypoint indices (COCO-style, 17 points)
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6


@dataclass
class _FrameKeypoints:
    """Internal only. Never returned from this module, never persisted."""
    keypoints: np.ndarray  # shape (17, 3): x, y, confidence
    timestamp: float


# Rough starting points for mapping metrics to labels; override any of
# these via config (gait.thresholds) once tuned against real captures.
DEFAULT_THRESHOLDS = {
    "brisk_speed": 0.15,       # hip-midpoint displacement per frame, above = brisk
    "slow_speed": 0.05,        # below = slow
    "rigid_variance": 0.02,    # ankle-separation variance, below = rigid
    "shuffle_variance": 0.08,  # above = shuffling
    "hunched_angle_deg": 10,   # torso lean from vertical, above = hunched
}


class GaitEstimator:
    def __init__(self, model_path: str, confidence_threshold: float = 0.3,
                 thresholds: dict = None):
        self.confidence_threshold = confidence_threshold
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.model = None
        self._warned_not_implemented = False
        try:
            try:
                from object_mapper import load_tflite_interpreter
            except ImportError:  # imported as part of the field package
                from field.object_mapper import load_tflite_interpreter
            self.model = load_tflite_interpreter(model_path)
            self.model.allocate_tensors()
        except Exception as e:
            logger.warning("Pose model not loaded (%s). Gait descriptors "
                           "will be omitted until a model is in place.", e)

    def _run_pose_model(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Returns raw keypoints for a single frame, or None. Internal use
        only within this module's burst analysis; never exposed outside it."""
        if self.model is None:
            return None

        try:
            input_detail = self.model.get_input_details()[0]
            output_detail = self.model.get_output_details()[0]
            _, height, width, _ = input_detail["shape"]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(width), int(height)))
            tensor = np.expand_dims(image, axis=0)
            if input_detail["dtype"] == np.float32:
                tensor = tensor.astype(np.float32) / 255.0
            else:
                tensor = tensor.astype(input_detail["dtype"])
            self.model.set_tensor(input_detail["index"], tensor)
            self.model.invoke()
            output = self.model.get_tensor(output_detail["index"])
            keypoints = np.asarray(output).reshape(-1, 17, 3)
            if keypoints.shape[0] != 1:
                raise ValueError(f"Unexpected MoveNet output shape: {output.shape}")
            return keypoints[0].astype(np.float32, copy=True)
        except Exception as exc:
            logger.warning("Inference failed; omitting descriptor: %s", exc)
            return None

    def _describe(self, avg_speed: float, stride_variance: float,
                  avg_torso_angle: float) -> str:
        """Maps coarse numeric metrics to a text label using the
        configured thresholds (gait.thresholds in config.yaml)."""
        t = self.thresholds
        speed_word = "brisk" if avg_speed > t["brisk_speed"] else (
            "slow" if avg_speed < t["slow_speed"] else "steady")
        regularity_word = "rigid" if stride_variance < t["rigid_variance"] else (
            "shuffling" if stride_variance > t["shuffle_variance"] else "even")
        posture_word = "upright" if avg_torso_angle < t["hunched_angle_deg"] else "hunched"

        return f"{speed_word}, {regularity_word} stride, {posture_word} posture"

    def describe_burst(self, frames: List[np.ndarray]) -> Optional[str]:
        """
        Takes a burst of frames (already captured, already in memory;
        this function does not itself touch the camera), runs pose
        estimation across them, derives a text descriptor, and returns
        ONLY that string. All keypoint data used along the way is local
        to this function call and is not retained after it returns.
        """
        if self.model is None or len(frames) < 3:
            return None

        frame_kps: List[_FrameKeypoints] = []
        for frame in frames:
            kp = self._run_pose_model(frame)
            if kp is not None:
                frame_kps.append(_FrameKeypoints(keypoints=kp, timestamp=time.monotonic()))

        if len(frame_kps) < 3:
            return None  # not enough usable frames to derive a stable descriptor

        # Apparent walking speed: displacement of hip midpoint across the burst
        hip_positions = []
        for fk in frame_kps:
            lh, rh = fk.keypoints[KP_LEFT_HIP], fk.keypoints[KP_RIGHT_HIP]
            if lh[2] > self.confidence_threshold and rh[2] > self.confidence_threshold:
                hip_positions.append(((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2))

        avg_speed = 0.0
        if len(hip_positions) >= 2:
            displacements = [
                np.linalg.norm(np.array(hip_positions[i]) - np.array(hip_positions[i - 1]))
                for i in range(1, len(hip_positions))
            ]
            avg_speed = float(np.mean(displacements))

        # Stride regularity: variance in ankle separation across the burst
        ankle_seps = []
        for fk in frame_kps:
            la, ra = fk.keypoints[KP_LEFT_ANKLE], fk.keypoints[KP_RIGHT_ANKLE]
            if la[2] > self.confidence_threshold and ra[2] > self.confidence_threshold:
                ankle_seps.append(float(np.linalg.norm(la[:2] - ra[:2])))
        stride_variance = float(np.var(ankle_seps)) if len(ankle_seps) >= 3 else 0.05

        # Posture: torso angle from shoulder midpoint to hip midpoint vs vertical
        torso_angles = []
        for fk in frame_kps:
            ls, rs = fk.keypoints[KP_LEFT_SHOULDER], fk.keypoints[KP_RIGHT_SHOULDER]
            lh, rh = fk.keypoints[KP_LEFT_HIP], fk.keypoints[KP_RIGHT_HIP]
            if min(ls[2], rs[2], lh[2], rh[2]) > self.confidence_threshold:
                shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
                hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
                vec = shoulder_mid - hip_mid
                angle = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))
                torso_angles.append(angle)
        avg_torso_angle = float(np.mean(torso_angles)) if torso_angles else 0.0

        descriptor = self._describe(avg_speed, stride_variance, avg_torso_angle)

        # frame_kps and all local arrays go out of scope and are garbage
        # collected here. Nothing from this function is retained beyond
        # the returned string.
        return descriptor
