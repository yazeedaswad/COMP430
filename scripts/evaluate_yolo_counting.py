from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.evaluation.counting import compute_counting_metrics


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO counting error on a split.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", default=str(ROOT / "data" / "processed" / "countingpills_detect" / "val" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "data" / "processed" / "countingpills_detect" / "val" / "labels"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=str(ROOT / "reports" / "tables" / "yolo_counting_val.csv"))
    return parser.parse_args()


def count_ground_truth(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

    from ultralytics import YOLO

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    rows: list[dict[str, int | str]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]
        pred_count = 0 if result.boxes is None else len(result.boxes)
        true_count = count_ground_truth(label_dir / f"{image_path.stem}.txt")
        y_true.append(true_count)
        y_pred.append(pred_count)
        rows.append(
            {
                "image": image_path.name,
                "true_count": true_count,
                "pred_count": pred_count,
                "error": pred_count - true_count,
            }
        )

    metrics = compute_counting_metrics(y_true, y_pred)

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "true_count", "pred_count", "error"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"images={len(image_paths)}")
    print(f"MAE={metrics.mae:.4f}")
    print(f"MSE={metrics.mse:.4f}")
    print(f"exact_count_accuracy={metrics.exact_count_accuracy:.4f}")
    print(f"results={out_path}")


if __name__ == "__main__":
    main()

