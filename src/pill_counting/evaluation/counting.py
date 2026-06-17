from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CountingMetrics:
    mae: float
    mse: float
    exact_count_accuracy: float


def compute_counting_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> CountingMetrics:
    true_counts = np.asarray(list(y_true), dtype=float)
    pred_counts = np.asarray(list(y_pred), dtype=float)

    if true_counts.shape != pred_counts.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    if true_counts.size == 0:
        raise ValueError("At least one count is required.")

    errors = pred_counts - true_counts
    return CountingMetrics(
        mae=float(np.mean(np.abs(errors))),
        mse=float(np.mean(errors**2)),
        exact_count_accuracy=float(np.mean(true_counts == pred_counts)),
    )

