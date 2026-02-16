"""SQLite metadata storage for registered faces."""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH: str = os.path.join("storage", "faces.db")
_lock = threading.Lock()


def init_db(path: str = DB_PATH) -> None:
    """Create the database and faces table if they don't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    logger.info("SQLite database initialized at %s.", path)


def insert_face(face_uuid: str, name: str) -> int:
    """Insert a new face record and return its integer row ID.

    Args:
        face_uuid: Unique UUID string for external reference.
        name: Human-readable name for the face.

    Returns:
        The auto-incremented integer ID (used as FAISS index ID).
    """
    with _lock:
        with _connect() as conn:
            cursor = conn.execute(
                "INSERT INTO faces (uuid, name, created_at) VALUES (?, ?, ?)",
                (face_uuid, name, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            return cursor.lastrowid


def get_face_by_id(face_id: int) -> dict | None:
    """Look up a face record by its integer row ID.

    Returns:
        A dict with keys {id, uuid, name, created_at}, or None if not found.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        return dict(row) if row else None


def get_face_count() -> int:
    """Return the total number of registered faces."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]


def _connect(path: str = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)
