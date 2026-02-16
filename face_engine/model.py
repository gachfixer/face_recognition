"""
Face analysis model singleton.

Loads InsightFace buffalo_l (ArcFace) once at startup.
Thread-safe lazy initialization with explicit load_model() call.
"""

import logging
import threading

from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

_model: FaceAnalysis | None = None
_lock = threading.Lock()


def load_model(
    det_size: tuple[int, int] = (640, 640),
    det_thresh: float = 0.1,
) -> None:
    """Load the InsightFace model. Must be called once at startup."""
    global _model
    with _lock:
        if _model is not None:
            return
        logger.info("Loading InsightFace model (buffalo_l)...")
        _model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _model.prepare(ctx_id=-1, det_size=det_size)
        _model.det_model.det_thresh = det_thresh
        logger.info(
            "InsightFace model loaded successfully (det_size=%s, det_thresh=%s).",
            det_size,
            det_thresh,
        )


def get_model() -> FaceAnalysis:
    """Return the loaded model instance. Raises if not initialized."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model
