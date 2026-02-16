"""FAISS vector index for face embeddings.

Uses IndexFlatIP (inner product) on L2-normalized vectors,
which is equivalent to cosine similarity search.
Wrapped in IndexIDMap to map FAISS results back to SQLite row IDs.
"""

import logging
import os
import threading

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH: str = os.path.join("storage", "faiss.index")
DEFAULT_DIMENSION: int = 512

_index: faiss.IndexIDMap | None = None
_lock = threading.Lock()


def load_index(path: str = INDEX_PATH, dimension: int = DEFAULT_DIMENSION) -> None:
    """Load existing FAISS index from disk or create a new one."""
    global _index
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        logger.info("Loading FAISS index from %s...", path)
        _index = faiss.read_index(path)
        logger.info(
            "FAISS index loaded: %d vectors, dimension=%d.",
            _index.ntotal,
            _index.d,
        )
    else:
        logger.info("Creating new FAISS index (dimension=%d)...", dimension)
        base = faiss.IndexFlatIP(dimension)
        _index = faiss.IndexIDMap(base)


def add_embedding(embedding: np.ndarray, face_id: int) -> None:
    """Add a normalized embedding to the index and persist to disk.

    Args:
        embedding: Raw embedding vector (will be L2-normalized internally).
        face_id: Integer ID matching the SQLite row ID.
    """
    with _lock:
        vec = embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(vec)
        _index.add_with_ids(vec, np.array([face_id], dtype=np.int64))
        _save()


def search_embedding(
    embedding: np.ndarray, k: int = 1
) -> list[tuple[int, float]]:
    """Search for the closest embeddings in the index.

    Args:
        embedding: Query embedding vector (will be L2-normalized internally).
        k: Number of nearest neighbors to return.

    Returns:
        List of (face_id, similarity_score) tuples, sorted by descending similarity.
        Returns empty list if index is empty.
    """
    with _lock:
        if _index is None or _index.ntotal == 0:
            return []
        vec = embedding.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(vec)
        distances, indices = _index.search(vec, min(k, _index.ntotal))
        results: list[tuple[int, float]] = []
        for i in range(distances.shape[1]):
            idx = int(indices[0][i])
            if idx != -1:
                results.append((idx, float(distances[0][i])))
        return results


def total_faces() -> int:
    """Return the number of indexed face embeddings."""
    if _index is None:
        return 0
    return _index.ntotal


def _save() -> None:
    """Persist current index to disk."""
    faiss.write_index(_index, INDEX_PATH)
