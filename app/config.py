"""Configuration utilities for the CameraHub application."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the person detection model."""

    config_path: Path
    weights_path: Path
    class_names_path: Optional[Path] = None
    confidence_threshold: float = 0.5


@dataclass
class CameraConfig:
    """Configuration for the camera input."""

    source: str | int = 0
    width: int = 640
    height: int = 480


@dataclass
class AppConfig:
    """Main application configuration."""

    database_path: Path = Path("data/camerahub.db")
    model: Optional[ModelConfig] = None
    camera: CameraConfig = field(default_factory=CameraConfig)
    recognition_tolerance: float = 0.5
    registration_samples: int = 5

    def ensure_data_directory(self) -> None:
        """Ensure that directories for persistent data exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()
