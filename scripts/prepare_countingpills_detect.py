from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def convert_label_line(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None

    class_id = parts[0]
    coords = [float(value) for value in parts[1:]]

    if len(coords) == 4:
        x_center, y_center, width, height = coords
        return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"

    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_split(raw_root: Path, out_root: Path, split: str) -> tuple[int, int, int]:
    raw_images = raw_root / split / "images"
    raw_labels = raw_root / split / "labels"
    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    instance_count = 0
    skipped_lines = 0

    for image_path in raw_images.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_count += 1
        link_or_copy(image_path, out_images / image_path.name)

        raw_label = raw_labels / f"{image_path.stem}.txt"
        out_label = out_labels / f"{image_path.stem}.txt"
        converted: list[str] = []

        if raw_label.exists():
            for line in raw_label.read_text(encoding="utf-8").splitlines():
                converted_line = convert_label_line(line)
                if converted_line is None:
                    skipped_lines += 1
                else:
                    converted.append(converted_line)

        instance_count += len(converted)
        out_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

    return image_count, instance_count, skipped_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(ROOT / "data" / "raw" / "CountingPills.v36i.yolov8"))
    parser.add_argument("--out-root", default=str(ROOT / "data" / "processed" / "countingpills_detect"))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    for split in ["train", "val", "test"]:
        image_count, instance_count, skipped_lines = prepare_split(raw_root, out_root, split)
        print(
            f"{split}: images={image_count}, converted_instances={instance_count}, "
            f"skipped_lines={skipped_lines}"
        )


if __name__ == "__main__":
    main()

