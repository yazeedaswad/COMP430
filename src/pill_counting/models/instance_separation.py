from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InstanceSeparationResult:
    count: int
    labels: np.ndarray
    foreground: np.ndarray
    markers: np.ndarray | None = None


def clean_foreground(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    foreground = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return foreground
