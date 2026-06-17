from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.countingpills_instance import CountingPillsInstanceDataset
from pill_counting.data.mediseg import MedisegInstanceDataset, detection_collate
from pill_counting.evaluation.counting import compute_counting_metrics
from pill_counting.models.mask_rcnn import create_mask_rcnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN counting error.")
    parser.add_argument("--dataset", choices=["mediseg", "countingpills"], default="mediseg")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--out", default=str(ROOT / "reports" / "tables" / "mask_rcnn_counting_val.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.dataset == "countingpills":
        data_root = args.data_root or str(ROOT / "data" / "raw" / "CountingPills.v36i.yolov8")
        dataset = CountingPillsInstanceDataset(data_root, args.split, limit=args.limit)
    else:
        data_root = args.data_root or str(ROOT / "data" / "raw" / "MEDISEG" / "3pills")
        dataset = MedisegInstanceDataset(data_root, args.split, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=detection_collate)

    model = create_mask_rcnn(num_classes=dataset.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    rows: list[dict[str, int | float]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                true_count = int(target["boxes"].shape[0])
                scores = output.get("scores", torch.empty(0, device=device))
                pred_count = int((scores >= args.score_threshold).sum().item())
                image_id = int(target["image_id"].item())
                y_true.append(true_count)
                y_pred.append(pred_count)
                row = {
                    "image_id": image_id,
                    "true_count": true_count,
                    "pred_count": pred_count,
                    "error": pred_count - true_count,
                }
                if args.dataset == "countingpills":
                    row["image"] = dataset.image_paths[len(rows)].name
                rows.append(row)

    metrics = compute_counting_metrics(y_true, y_pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["image", "image_id", "true_count", "pred_count", "error"] if args.dataset == "countingpills" else ["image_id", "true_count", "pred_count", "error"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"images={len(rows)}")
    print(f"score_threshold={args.score_threshold}")
    print(f"MAE={metrics.mae:.4f}")
    print(f"MSE={metrics.mse:.4f}")
    print(f"exact_count_accuracy={metrics.exact_count_accuracy:.4f}")
    print(f"results={out_path}")


if __name__ == "__main__":
    main()
