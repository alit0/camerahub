"""Camera abstraction for CameraHub."""
from __future__ import annotations

from typing import Iterator, Tuple

import cv2

from .config import CameraConfig


class Camera:
    """Wrapper around ``cv2.VideoCapture`` supporting local or IP cameras."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture = cv2.VideoCapture(config.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara: {config.source}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self) -> Tuple[bool, any]:
        return self._capture.read()

    def frames(self) -> Iterator[Tuple[bool, any]]:
        while True:
            yield self.read()

    def release(self) -> None:
        if self._capture.isOpened():
            self._capture.release()
