from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.yolo_center import YoloCenterDataset, yolo_center_collate
from pill_counting.models.tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train foreground + center heatmap model for the proposed method.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "raw" / "CountingPills.v36i.yolov8"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--name", default="countingpills_tiny_unet_centers")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--center-loss-weight", type=float, default=4.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="Stop if the best validation score does not improve for this many epochs. 0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to continue training from.")
    parser.add_argument("--log-every", type=int, default=25, help="Print training progress every N batches. 0 disables batch progress logging.")
    parser.add_argument("--preprocessing", choices=["none"], default="none")
    return parser.parse_args()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return float(((2 * intersection + 1e-6) / (union + 1e-6)).mean().detach().cpu())


def train_or_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    center_loss_weight: float,
    epoch: int,
    total_epochs: int,
    log_every: int = 0,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    total_loss = 0.0
    total_fg_dice = 0.0
    total_center_mse = 0.0
    batches = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch_idx, (images, targets, _) in enumerate(loader, start=1):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            fg_logits = outputs[:, 0:1]
            center_logits = outputs[:, 1:2]
            fg_target = targets[:, 0:1]
            center_target = targets[:, 1:2]
            center_prob = torch.sigmoid(center_logits)

            loss = bce(fg_logits, fg_target) + center_loss_weight * mse(center_prob, center_target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_fg_dice += dice_score(fg_logits, fg_target)
            total_center_mse += float(mse(center_prob, center_target).detach().cpu())
            batches += 1
            if is_train and log_every and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader)):
                avg_loss = total_loss / batches
                avg_dice = total_fg_dice / batches
                print(
                    f"epoch={epoch}/{total_epochs} batch={batch_idx}/{len(loader)} "
                    f"avg_train_loss={avg_loss:.4f} avg_train_fg_dice={avg_dice:.4f}",
                    flush=True,
                )

    return total_loss / batches, total_fg_dice / batches, total_center_mse / batches


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = ROOT / "experiments" / "proposed_pipeline" / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = YoloCenterDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        limit=args.limit,
        preprocessing=args.preprocessing,
    )
    val_dataset = YoloCenterDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        limit=args.limit,
        preprocessing=args.preprocessing,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=yolo_center_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=yolo_center_collate)

    model = TinyUNet(out_channels=2, base_channels=args.base_channels).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    best_score = -1.0
    stale_epochs = 0
    history_path = run_dir / "train_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "time_sec",
                "lr",
                "train_loss",
                "train_fg_dice",
                "train_center_mse",
                "val_loss",
                "val_fg_dice",
                "val_center_mse",
                "score",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_dice, train_center_mse = train_or_eval_epoch(
                model, train_loader, device, optimizer, args.center_loss_weight, epoch, args.epochs, args.log_every
            )
            val_loss, val_dice, val_center_mse = train_or_eval_epoch(
                model, val_loader, device, None, args.center_loss_weight, epoch, args.epochs, 0
            )
            row = {
                "epoch": epoch,
                "time_sec": round(time.time() - start, 2),
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_fg_dice": train_dice,
                "train_center_mse": train_center_mse,
                "val_loss": val_loss,
                "val_fg_dice": val_dice,
                "val_center_mse": val_center_mse,
                "score": val_dice - val_center_mse,
            }
            writer.writerow(row)
            file.flush()
            score = val_dice - val_center_mse
            torch.save(model.state_dict(), weights_dir / "last.pt")
            if score > best_score + args.min_delta:
                best_score = score
                stale_epochs = 0
                torch.save(model.state_dict(), weights_dir / "best.pt")
            else:
                stale_epochs += 1
            scheduler.step(score)
            print(
                f"epoch={epoch}/{args.epochs} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"train_loss={train_loss:.4f} train_fg_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_fg_dice={val_dice:.4f} val_center_mse={val_center_mse:.5f} "
                f"best_score={best_score:.5f}",
                flush=True,
            )
            if args.patience and stale_epochs >= args.patience:
                print(f"Early stopping: no validation score improvement for {stale_epochs} epochs.", flush=True)
                break

    print(f"Saved weights to {weights_dir}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
