from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.yolo_center import YoloCenterDataset, yolo_center_collate
from pill_counting.evaluation.counting import compute_counting_metrics
from pill_counting.models.center_split import center_marker_watershed
from pill_counting.models.tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the proposed center-marker method on YOLO-format data.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "raw" / "CountingPills.v36i.yolov8"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--center-threshold", type=float, default=0.35)
    parser.add_argument("--peak-min-distance", type=int, default=24)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.0)
    parser.add_argument("--min-solidity", type=float, default=0.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--preprocessing", choices=["none"], default="none")
    parser.add_argument("--out", default=str(ROOT / "reports" / "tables" / "proposed_method_counting_val.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = YoloCenterDataset(
        args.data_root,
        args.split,
        image_size=args.image_size,
        limit=args.limit,
        preprocessing=args.preprocessing,
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=yolo_center_collate)

    model = TinyUNet(out_channels=2, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    rows: list[dict[str, int | str | float]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, _, metadata in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).detach().cpu().numpy()
            for pred, meta in zip(probs, metadata):
                fg = cv2.resize(pred[0], (int(meta["original_width"]), int(meta["original_height"])), interpolation=cv2.INTER_LINEAR)
                centers = cv2.resize(pred[1], (int(meta["original_width"]), int(meta["original_height"])), interpolation=cv2.INTER_LINEAR)
                result = center_marker_watershed(
                    fg,
                    centers,
                    foreground_threshold=args.foreground_threshold,
                    center_threshold=args.center_threshold,
                    peak_min_distance=args.peak_min_distance,
                    min_area=args.min_area,
                    max_area_ratio=args.max_area_ratio,
                    min_solidity=args.min_solidity,
                    max_aspect_ratio=args.max_aspect_ratio,
                )
                true_count = int(meta["true_count"])
                pred_count = int(result.count)
                y_true.append(true_count)
                y_pred.append(pred_count)
                rows.append(
                    {
                        "file_name": str(meta["file_name"]),
                        "true_count": true_count,
                        "pred_count": pred_count,
                        "error": pred_count - true_count,
                    }
                )

    metrics = compute_counting_metrics(y_true, y_pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "true_count", "pred_count", "error"])
        writer.writeheader()
        writer.writerows(rows)

    under = sum(1 for true, pred in zip(y_true, y_pred) if pred < true)
    over = sum(1 for true, pred in zip(y_true, y_pred) if pred > true)
    print(f"images={len(rows)}")
    print(f"foreground_threshold={args.foreground_threshold}")
    print(f"center_threshold={args.center_threshold}")
    print(f"peak_min_distance={args.peak_min_distance}")
    print(f"preprocessing={args.preprocessing}")
    print(f"MAE={metrics.mae:.4f}")
    print(f"MSE={metrics.mse:.4f}")
    print(f"exact_count_accuracy={metrics.exact_count_accuracy:.4f}")
    print(f"under_split_rate={under / len(rows) if rows else 0:.4f}")
    print(f"over_split_rate={over / len(rows) if rows else 0:.4f}")
    print(f"results={out_path}")


if __name__ == "__main__":
    main()
