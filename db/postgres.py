"""PostgreSQL metadata storage for registered faces."""

import logging

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2 import sql

logger = logging.getLogger(__name__)

DB_CONFIG: dict = {
    "host": "localhost",
    "port": 5433,
    "user": "javasoft",
    "password": "javas0ft",
    "dbname": "face_database",
}

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_db() -> None:
    """Ensure the database and faces table exist, then create a connection pool."""
    global _pool
    _ensure_database()
    _pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)

    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id SERIAL PRIMARY KEY,
                    uuid TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
    finally:
        _pool.putconn(conn)

    logger.info(
        "PostgreSQL database initialized at %s:%s/%s.",
        DB_CONFIG["host"],
        DB_CONFIG["port"],
        DB_CONFIG["dbname"],
    )


def insert_face(face_uuid: str, name: str) -> int:
    """Insert a new face record and return its serial ID.

    Args:
        face_uuid: Unique UUID string for external reference.
        name: Human-readable name for the face.

    Returns:
        The auto-incremented integer ID (used as FAISS index ID).
    """
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO faces (uuid, name) VALUES (%s, %s) RETURNING id",
                (face_uuid, name),
            )
            face_id: int = cur.fetchone()[0]
        conn.commit()
        return face_id
    finally:
        _pool.putconn(conn)


def get_face_by_id(face_id: int) -> dict | None:
    """Look up a face record by its integer row ID.

    Returns:
        A dict with keys {id, uuid, name, created_at}, or None if not found.
    """
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM faces WHERE id = %s", (face_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        _pool.putconn(conn)


def get_face_count() -> int:
    """Return the total number of registered faces."""
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM faces")
            return cur.fetchone()[0]
    finally:
        _pool.putconn(conn)


def _ensure_database() -> None:
    """Create the target database if it doesn't exist yet."""
    admin_cfg = {**DB_CONFIG, "dbname": "postgres"}
    conn = psycopg2.connect(**admin_cfg)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (DB_CONFIG["dbname"],),
            )
            if not cur.fetchone():
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(DB_CONFIG["dbname"])
                    )
                )
                logger.info("Created database '%s'.", DB_CONFIG["dbname"])
    finally:
        conn.close()
