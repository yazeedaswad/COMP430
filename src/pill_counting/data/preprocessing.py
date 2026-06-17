from __future__ import annotations

import numpy as np


def preprocess_rgb(image: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode == "none":
        return image
    raise ValueError(f"Unknown preprocessing mode: {mode}")
