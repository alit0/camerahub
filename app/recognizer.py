"""Real-time recognition pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import cv2
import face_recognition
import numpy as np

from .config import AppConfig
from .detection import Detection, PersonDetector
from .registry import FaceRegistry
from .storage import Storage


@dataclass
class Recognition:
    label: str
    confidence: float
    is_known: bool
    box: Sequence[int]  # x, y, w, h


class RecognitionPipeline:
    """Coordinates object detection and face recognition."""

    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        registry: FaceRegistry,
        detector: Optional[PersonDetector] = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._registry = registry
        self._detector = detector

    def process_frame(self, frame: np.ndarray) -> List[Recognition]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        recognitions: List[Recognition] = []

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            match = self._registry.find_best_match(encoding, self._config.recognition_tolerance)
            if match:
                label, distance = match
                confidence = max(0.0, 1.0 - distance)
                is_known = True
            else:
                label = "Desconocido"
                confidence = 0.0
                is_known = False
            box = (left, top, right - left, bottom - top)
            recognitions.append(Recognition(label=label, confidence=confidence, is_known=is_known, box=box))
            self._storage.log_event(label, is_known)

        if self._detector is not None:
            for detection in self._detector.detect(frame):
                # Avoid duplicates where a face already identified a person.
                if not self._overlaps_with_recognitions(detection, recognitions):
                    recognitions.append(
                        Recognition(
                            label=detection.label,
                            confidence=detection.confidence,
                            is_known=False,
                            box=detection.box,
                        )
                    )
        return recognitions

    @staticmethod
    def _overlaps_with_recognitions(detection: Detection, recognitions: Iterable[Recognition]) -> bool:
        dx, dy, dw, dh = detection.box
        for recognition in recognitions:
            rx, ry, rw, rh = recognition.box
            if RecognitionPipeline._boxes_overlap((dx, dy, dx + dw, dy + dh), (rx, ry, rx + rw, ry + rh)):
                return True
        return False

    @staticmethod
    def _boxes_overlap(a, b) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
