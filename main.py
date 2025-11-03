"""Entry point for the CameraHub application."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from tkinter import Tk

from app.config import AppConfig, CameraConfig, ModelConfig
from app.gui import Application

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CameraHub surveillance assistant")
    parser.add_argument("--camera", help="Fuente de video (índice o URL)", default="0")
    parser.add_argument("--width", type=int, default=640, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=480, help="Alto de captura")
    parser.add_argument("--model-config", type=Path, help="Ruta al archivo de configuración del modelo DNN")
    parser.add_argument("--model-weights", type=Path, help="Ruta al archivo de pesos del modelo DNN")
    parser.add_argument("--class-names", type=Path, help="Archivo opcional con nombres de clases")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerancia para el reconocimiento facial (menor es más estricto)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/camerahub.db"),
        help="Ruta al archivo de base de datos",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    camera_source: str | int
    if args.camera.isdigit():
        camera_source = int(args.camera)
    else:
        camera_source = args.camera

    model_config: Optional[ModelConfig] = None
    if args.model_config and args.model_weights:
        model_config = ModelConfig(
            config_path=args.model_config,
            weights_path=args.model_weights,
            class_names_path=args.class_names,
        )

    return AppConfig(
        database_path=args.database,
        model=model_config,
        camera=CameraConfig(source=camera_source, width=args.width, height=args.height),
        recognition_tolerance=args.tolerance,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    root = Tk()
    app = Application(root, config)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
