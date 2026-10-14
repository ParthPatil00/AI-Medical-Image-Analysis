"""Shared image preprocessing helpers for the chest X-ray demo."""
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_SIZE = (180, 180)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def load_image_for_prediction(image_path: str | Path) -> np.ndarray:
    """Load one image as RGB float32 in [0, 1], with shape (1, H, W, 3)."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
