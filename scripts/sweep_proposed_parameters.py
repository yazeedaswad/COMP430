from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact validation sweep for proposed-method post-processing.")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "experiments" / "proposed_pipeline" / "countingpills_proposed_long_e60_bc16" / "weights" / "best.pt"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=str(ROOT / "reports" / "tables" / "proposed_method_parameter_sensitivity.csv"))
    return parser.parse_args()


def parse_metric_output(output: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def main() -> None:
    args = parse_args()
    configs = [
        ("baseline", 0.50, 0.35, 24),
        ("lower foreground threshold", 0.45, 0.35, 24),
        ("higher foreground threshold", 0.55, 0.35, 24),
        ("lower center threshold", 0.50, 0.30, 24),
        ("higher center threshold", 0.50, 0.40, 24),
        ("shorter peak distance", 0.50, 0.35, 16),
        ("longer peak distance", 0.50, 0.35, 32),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_path.parent / "proposed_sensitivity_rows"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | int]] = []
    for name, foreground_threshold, center_threshold, peak_min_distance in configs:
        row_csv = scratch_dir / f"{name.replace(' ', '_')}.csv"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_proposed_method.py"),
            "--weights",
            args.weights,
            "--split",
            args.split,
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--foreground-threshold",
            str(foreground_threshold),
            "--center-threshold",
            str(center_threshold),
            "--peak-min-distance",
            str(peak_min_distance),
            "--out",
            str(row_csv),
        ]
        print(
            "running "
            f"name={name} foreground_threshold={foreground_threshold} "
            f"center_threshold={center_threshold} peak_min_distance={peak_min_distance}",
            flush=True,
        )
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        print(completed.stdout, flush=True)
        metrics = parse_metric_output(completed.stdout)
        rows.append(
            {
                "setting": name,
                "split": args.split,
                "foreground_threshold": foreground_threshold,
                "center_threshold": center_threshold,
                "peak_min_distance": peak_min_distance,
                "images": int(metrics["images"]),
                "mae": float(metrics["MAE"]),
                "mse": float(metrics["MSE"]),
                "exact_count_accuracy": float(metrics["exact_count_accuracy"]),
                "under_split_rate": float(metrics["under_split_rate"]),
                "over_split_rate": float(metrics["over_split_rate"]),
            }
        )

    fieldnames = [
        "setting",
        "split",
        "foreground_threshold",
        "center_threshold",
        "peak_min_distance",
        "images",
        "mae",
        "mse",
        "exact_count_accuracy",
        "under_split_rate",
        "over_split_rate",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"sensitivity_table={out_path}")


if __name__ == "__main__":
    main()
