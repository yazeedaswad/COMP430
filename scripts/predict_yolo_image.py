from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO pill detection on one image or folder.")
    parser.add_argument("source", help="Image path, folder path, or glob pattern.")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "experiments" / "baseline_yolo" / "countingpills_yolo11n_detect_full" / "weights" / "best.pt"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="test_image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(ROOT / "experiments" / "baseline_yolo" / "predictions"),
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=False,
    )

    for result in results:
        count = 0 if result.boxes is None else len(result.boxes)
        print(f"{Path(result.path).name}: predicted_count={count}")

    print(
        "Saved predictions to "
        f"{ROOT / 'experiments' / 'baseline_yolo' / 'predictions' / args.name}"
    )


if __name__ == "__main__":
    main()

