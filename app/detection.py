"""Object detection wrapper using OpenCV DNN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2

from .config import ModelConfig


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]  # x, y, w, h


class PersonDetector:
    """Detects persons in frames using OpenCV DNN models."""

    def __init__(self, config: ModelConfig) -> None:
        if not config.config_path.exists() or not config.weights_path.exists():
            raise FileNotFoundError(
                "Model configuration or weights file not found. "
                "Please download a MobileNet-SSD or YOLO model."
            )
        self._config = config
        self._net = cv2.dnn_DetectionModel(
            str(config.weights_path), str(config.config_path)
        )
        self._net.setInputSize(320, 320)
        self._net.setInputScale(1.0 / 127.5)
        self._net.setInputMean((127.5, 127.5, 127.5))
        self._net.setInputSwapRB(True)
        self._class_names = self._load_class_names(config.class_names_path)
        if "person" not in self._class_names:
            raise ValueError("The provided model does not include a 'person' class.")

    def _load_class_names(self, path: Path | None) -> List[str]:
        if path is None:
            # Default to COCO classes used by many SSD/YOLO models.
            return [
                "background",
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def detect(self, frame) -> Iterable[Detection]:
        detections = self._net.detect(frame, confThreshold=self._config.confidence_threshold)
        if detections[0] is None or len(detections[0]) == 0:
            return []
        class_ids, confidences, boxes = detections
        results: List[Detection] = []
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            label = self._class_names[class_id] if class_id < len(self._class_names) else str(class_id)
            if label != "person":
                continue
            results.append(Detection(label=label, confidence=float(confidence), box=tuple(map(int, box))))
        return results
