"""Face embedding extraction."""

import numpy as np

from .model import get_model
from .utils import decode_image


def get_embedding(image_bytes: bytes) -> np.ndarray:
    """Detect a single face and return its 512-d ArcFace embedding.

    Args:
        image_bytes: Raw image file bytes (JPEG, PNG, etc.).

    Returns:
        A float32 numpy array of shape (512,).

    Raises:
        ValueError: If zero or multiple faces are detected, or image is corrupt.
    """
    img = decode_image(image_bytes)
    model = get_model()
    faces = model.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    if len(faces) > 1:
        raise ValueError(
            f"Multiple faces detected ({len(faces)}). "
            "Please upload an image with exactly one face."
        )

    return faces[0].embedding.astype(np.float32)
