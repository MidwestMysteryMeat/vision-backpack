"""
anonymizer.py

On-device, at-capture-time face anonymization. This runs BEFORE any frame
is written to disk. The raw face region should never exist as a
persisted file, even transiently.

Overlay is procedurally generated (not a reversible blur/pixelation) so
there's no way to reconstruct the original face from the stored output.
"""

import cv2
import numpy as np
import hashlib


class FaceAnonymizer:
    def __init__(self, cascade_path: str, min_face_size_px: int = 30):
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise ValueError(f"Face anonymizer model could not be loaded: {cascade_path}")
        self.min_face_size_px = min_face_size_px

    def detect_faces(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size_px, self.min_face_size_px)
        )
        return [tuple(f) for f in faces]  # list of (x, y, w, h)

    def _generate_fantasy_overlay(self, w: int, h: int, seed_bytes: bytes) -> np.ndarray:
        """
        Procedurally generates a fantasy-styled mask pattern sized to the
        face region. Seeded off a hash of the region (not the pixel data
        itself) purely for visual variety run-to-run. This is NOT a
        reversible encoding of the face, just a way to avoid every masked
        face looking identical.
        """
        seed = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        base_color = rng.integers(40, 200, size=3)
        overlay[:] = base_color

        # Simple procedural "ornate mask" pattern: concentric shapes,
        # varied per-seed so masks don't look copy-pasted.
        center = (w // 2, h // 2)
        for i in range(3):
            radius = int(min(w, h) * (0.4 - i * 0.1))
            color = tuple(int(c) for c in rng.integers(0, 255, size=3))
            cv2.circle(overlay, center, max(radius, 1), color, thickness=2)

        return overlay

    def anonymize(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a new frame with all detected faces replaced by procedural
        overlays. The input frame is never returned or persisted unmodified.
        """
        output = frame.copy()
        faces = self.detect_faces(frame)

        for (x, y, w, h) in faces:
            # Seed off region coordinates + frame shape, not pixel content,
            # so this is trivially non-reversible.
            seed_bytes = f"{x}{y}{w}{h}{frame.shape}".encode("utf-8")
            overlay = self._generate_fantasy_overlay(w, h, seed_bytes)
            output[y:y + h, x:x + w] = overlay

        return output
