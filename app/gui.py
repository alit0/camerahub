"""Tkinter based interface for the CameraHub application."""
from __future__ import annotations

import logging
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    TOP,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)
from tkinter import Tk

import cv2
from PIL import Image, ImageTk

from .camera import Camera
from .config import AppConfig, DEFAULT_CONFIG, ModelConfig
from .detection import PersonDetector
from .recognizer import RecognitionPipeline
from .registry import FaceRegistry
from .storage import Storage

LOGGER = logging.getLogger(__name__)


class Application:
    """Main GUI application."""

    def __init__(self, root: Tk, config: AppConfig = DEFAULT_CONFIG) -> None:
        self.root = root
        self.config = config
        self.config.ensure_data_directory()
        self.storage = Storage(config.database_path)
        self.registry = FaceRegistry(self.storage)
        self.detector = self._create_detector(config.model) if config.model else None
        self.pipeline = RecognitionPipeline(config, self.storage, self.registry, self.detector)

        self.camera: Camera | None = None
        self._frame = None
        self._photo = None
        self._job = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.title("CameraHub")
        self.root.geometry("960x720")

        controls = ttk.Frame(self.root)
        controls.pack(side=TOP, fill=BOTH)

        self.start_button = ttk.Button(controls, text="Iniciar", command=self.toggle_stream)
        self.start_button.pack(side=LEFT, padx=4, pady=4)

        register_button = ttk.Button(controls, text="Registrar rostro", command=self.open_registration_dialog)
        register_button.pack(side=LEFT, padx=4, pady=4)

        refresh_button = ttk.Button(controls, text="Recargar base", command=self.reload_registry)
        refresh_button.pack(side=LEFT, padx=4, pady=4)

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill=BOTH, expand=True)

        self.log_box = ttk.Treeview(self.root, columns=("timestamp", "label", "estado"), show="headings", height=5)
        self.log_box.heading("timestamp", text="Timestamp")
        self.log_box.heading("label", text="Etiqueta")
        self.log_box.heading("estado", text="Estado")
        self.log_box.pack(side=TOP, fill=BOTH, padx=4, pady=4)
        self.refresh_log()

    def _create_detector(self, model: ModelConfig | None) -> PersonDetector | None:
        if model is None:
            return None
        try:
            return PersonDetector(model)
        except FileNotFoundError as exc:
            messagebox.showwarning("Modelo no disponible", str(exc))
            LOGGER.warning("Model files missing: %s", exc)
            return None

    def toggle_stream(self) -> None:
        if self.camera is None:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self) -> None:
        try:
            self.camera = Camera(self.config.camera)
        except RuntimeError as exc:
            messagebox.showerror("Error de cámara", str(exc))
            LOGGER.error("Unable to open camera: %s", exc)
            self.camera = None
            return
        self.start_button.config(text="Detener")
        self._update_frame()

    def stop_stream(self) -> None:
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
        if self.camera:
            self.camera.release()
            self.camera = None
        self.start_button.config(text="Iniciar")

    def _update_frame(self) -> None:
        if self.camera is None:
            return
        success, frame = self.camera.read()
        if not success:
            LOGGER.warning("No se pudo obtener frame de la cámara")
            self._job = self.root.after(100, self._update_frame)
            return

        self._frame = frame.copy()
        overlay = frame.copy()
        recognitions = self.pipeline.process_frame(frame)
        for recognition in recognitions:
            x, y, w, h = recognition.box
            color = (0, 255, 0) if recognition.is_known else (0, 0, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label = f"{recognition.label}"
            cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self._update_photo(overlay)
        self.refresh_log()
        self._job = self.root.after(50, self._update_frame)

    def _update_photo(self, frame) -> None:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self._photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._photo)
        self.video_label.image = self._photo

    def refresh_log(self) -> None:
        for row in self.log_box.get_children():
            self.log_box.delete(row)
        for event in self.storage.get_events():
            estado = "Conocido" if event.is_known else "Desconocido"
            self.log_box.insert("", END, values=(event.timestamp.strftime("%Y-%m-%d %H:%M:%S"), event.label, estado))

    def reload_registry(self) -> None:
        self.registry.reload()
        messagebox.showinfo("Registro actualizado", "Se recargaron las identidades registradas.")

    def open_registration_dialog(self) -> None:
        dialog = RegistrationDialog(self.root, self)
        dialog.show()

    def capture_frame(self):
        if self._frame is None:
            raise RuntimeError("No hay frame disponible. Inicie el video antes de registrar.")
        return self._frame.copy()

    def shutdown(self) -> None:
        self.stop_stream()


class RegistrationDialog:
    """Handles face registration via a pop-up dialog."""

    def __init__(self, parent: Tk, app: Application) -> None:
        self.app = app
        self.window = Toplevel(parent)
        self.window.title("Registrar nuevo rostro")
        self.window.geometry("400x200")

        ttk.Label(self.window, text="Nombre / etiqueta:").pack(pady=5)
        self.name_entry = ttk.Entry(self.window)
        self.name_entry.pack(fill=BOTH, padx=10)

        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=BOTH, expand=True, pady=10)

        capture_button = ttk.Button(button_frame, text="Capturar de cámara", command=self.capture)
        capture_button.pack(side=LEFT, padx=5)

        upload_button = ttk.Button(button_frame, text="Subir imagen", command=self.upload_image)
        upload_button.pack(side=LEFT, padx=5)

        close_button = ttk.Button(button_frame, text="Cerrar", command=self.window.destroy)
        close_button.pack(side=RIGHT, padx=5)

    def show(self) -> None:
        self.window.transient(self.app.root)
        self.window.grab_set()
        self.app.root.wait_window(self.window)

    def _register_image(self, image_path: Path | None, frame=None) -> None:
        label = self.name_entry.get().strip()
        if not label:
            messagebox.showwarning("Nombre requerido", "Ingrese un nombre para registrar el rostro.")
            return
        try:
            if frame is not None:
                image = frame
            elif image_path is not None:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("No se pudo leer la imagen seleccionada.")
            else:
                raise ValueError("No hay imagen para registrar.")
            self.app.registry.register_from_image(label, image)
            messagebox.showinfo("Registro exitoso", f"Se registró {label} correctamente.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error registering face: %s", exc)
            messagebox.showerror("Error", str(exc))

    def capture(self) -> None:
        try:
            frame = self.app.capture_frame()
        except RuntimeError as exc:
            messagebox.showerror("Error", str(exc))
            return
        self._register_image(None, frame)

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagen", "*.png;*.jpg;*.jpeg")],
        )
        if not file_path:
            return
        self._register_image(Path(file_path))
