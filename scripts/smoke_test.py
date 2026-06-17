from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.evaluation.counting import compute_counting_metrics


def main() -> None:
    metrics = compute_counting_metrics(y_true=[3, 5, 2], y_pred=[3, 4, 2])
    print("Project smoke test passed.")
    print(f"MAE: {metrics.mae:.3f}")
    print(f"MSE: {metrics.mse:.3f}")
    print(f"Exact count accuracy: {metrics.exact_count_accuracy:.3f}")


if __name__ == "__main__":
    main()

