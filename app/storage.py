"""Persistent storage for CameraHub events and face encodings."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import numpy as np


@dataclass
class DetectionEvent:
    """Represents a single recognition event."""

    timestamp: datetime
    label: str
    is_known: bool


class Storage:
    """SQLite backed storage for face encodings and events."""

    def __init__(self, database_path: Path) -> None:
        self._db_path = database_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._create_schema(conn)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                encoding BLOB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                label TEXT NOT NULL,
                is_known INTEGER NOT NULL
            )
            """
        )

    def add_face_encoding(self, label: str, encoding: np.ndarray) -> None:
        data = encoding.astype(np.float32).tobytes()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO face_encodings(label, encoding) VALUES(?, ?)",
                (label, data),
            )

    def get_face_encodings(self) -> List[Tuple[str, np.ndarray]]:
        with self._connect() as conn:
            cursor = conn.execute("SELECT label, encoding FROM face_encodings")
            records: List[Tuple[str, np.ndarray]] = []
            for label, blob in cursor.fetchall():
                encoding = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
                records.append((label, encoding))
            return records

    def log_event(self, label: str, is_known: bool) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events(timestamp, label, is_known) VALUES(?, ?, ?)",
                (timestamp, label, int(is_known)),
            )

    def get_events(self, limit: int = 100) -> Iterable[DetectionEvent]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT timestamp, label, is_known FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            for timestamp_str, label, is_known in cursor.fetchall():
                yield DetectionEvent(
                    timestamp=datetime.fromisoformat(timestamp_str),
                    label=label,
                    is_known=bool(is_known),
                )
