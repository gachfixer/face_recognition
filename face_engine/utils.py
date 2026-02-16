"""Image decoding utilities."""

import cv2
import numpy as np


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR numpy array (OpenCV format).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. The file may be corrupt or unsupported.")
    return img
