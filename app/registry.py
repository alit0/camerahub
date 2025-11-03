"""Face registration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import face_recognition
import numpy as np

from .storage import Storage


@dataclass
class KnownFace:
    label: str
    encoding: np.ndarray


class FaceRegistry:
    """Manages registration and retrieval of known faces."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._cache: Dict[str, List[np.ndarray]] = {}
        self.reload()

    @property
    def labels(self) -> Sequence[str]:
        return tuple(self._cache.keys())

    def reload(self) -> None:
        """Reload face encodings from persistent storage."""
        self._cache.clear()
        for label, encoding in self._storage.get_face_encodings():
            self._cache.setdefault(label, []).append(encoding)

    def register_from_image(self, label: str, image: np.ndarray) -> None:
        """Register faces detected in the provided image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            raise ValueError("No face detected in the provided image")
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        for encoding in encodings:
            self._storage.add_face_encoding(label, encoding)
            self._cache.setdefault(label, []).append(encoding)

    def find_best_match(self, encoding: np.ndarray, tolerance: float) -> Tuple[str, float] | None:
        """Return the best matching label for the encoding within tolerance."""
        matches: List[Tuple[str, float]] = []
        for label, encodings in self._cache.items():
            distances = face_recognition.face_distance(encodings, encoding)
            if distances.size == 0:
                continue
            best_distance = float(np.min(distances))
            matches.append((label, best_distance))
        if not matches:
            return None
        label, distance = min(matches, key=lambda item: item[1])
        if distance <= tolerance:
            return label, distance
        return None
