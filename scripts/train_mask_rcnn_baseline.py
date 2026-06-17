from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.countingpills_instance import CountingPillsInstanceDataset
from pill_counting.data.mediseg import MedisegInstanceDataset, detection_collate
from pill_counting.models.mask_rcnn import create_mask_rcnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN baseline.")
    parser.add_argument("--dataset", choices=["mediseg", "countingpills"], default="mediseg")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="mediseg3_maskrcnn")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to load before continuing training.")
    parser.add_argument("--log-every", type=int, default=25, help="Print batch losses every N batches.")
    return parser.parse_args()


def move_targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def loss_value(loss_dict: dict[str, torch.Tensor], key: str) -> float:
    value = loss_dict.get(key)
    if value is None:
        return 0.0
    return float(value.detach().cpu())


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = ROOT / "experiments" / "baseline_mask_rcnn" / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "countingpills":
        data_root = args.data_root or str(ROOT / "data" / "raw" / "CountingPills.v36i.yolov8")
        train_dataset = CountingPillsInstanceDataset(data_root, "train", limit=args.limit)
    else:
        data_root = args.data_root or str(ROOT / "data" / "raw" / "MEDISEG" / "3pills")
        train_dataset = MedisegInstanceDataset(data_root, "train", limit=args.limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
    )

    model = create_mask_rcnn(num_classes=train_dataset.num_classes, pretrained=not args.no_pretrained and args.resume is None)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Loaded checkpoint from {args.resume}", flush=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    history_path = run_dir / "train_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "time_sec",
                "loss_total",
                "loss_classifier",
                "loss_box_reg",
                "loss_mask",
                "loss_objectness",
                "loss_rpn_box_reg",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            totals: dict[str, float] = {}
            batch_count = 0

            for batch_idx, (images, targets) in enumerate(train_loader, 1):
                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)

                loss_dict = model(images, targets)
                loss = sum(loss_value for loss_value in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_count += 1
                for key, value in loss_dict.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
                totals["loss_total"] = totals.get("loss_total", 0.0) + float(loss.detach().cpu())

                if batch_idx == 1 or batch_idx == len(train_loader) or batch_idx % args.log_every == 0:
                    print(
                        f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(train_loader)} "
                        f"loss_total={float(loss.detach().cpu()):.4f} "
                        f"loss_classifier={loss_value(loss_dict, 'loss_classifier'):.4f} "
                        f"loss_box_reg={loss_value(loss_dict, 'loss_box_reg'):.4f} "
                        f"loss_mask={loss_value(loss_dict, 'loss_mask'):.4f}",
                        flush=True,
                    )

            row = {
                "epoch": epoch,
                "time_sec": round(time.time() - start, 2),
                "loss_total": totals.get("loss_total", 0.0) / batch_count,
                "loss_classifier": totals.get("loss_classifier", 0.0) / batch_count,
                "loss_box_reg": totals.get("loss_box_reg", 0.0) / batch_count,
                "loss_mask": totals.get("loss_mask", 0.0) / batch_count,
                "loss_objectness": totals.get("loss_objectness", 0.0) / batch_count,
                "loss_rpn_box_reg": totals.get("loss_rpn_box_reg", 0.0) / batch_count,
            }
            writer.writerow(row)
            file.flush()
            torch.save(model.state_dict(), weights_dir / "last.pt")
            torch.save(model.state_dict(), weights_dir / f"epoch_{epoch:03d}.pt")
            print(
                f"epoch={epoch} done loss_total={row['loss_total']:.4f} "
                f"time_sec={row['time_sec']}",
                flush=True,
            )

    print(f"Saved weights to {weights_dir}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
