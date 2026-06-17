from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def draw_yolo_segmentation(image_path: Path, label_path: Path, out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = [float(value) for value in parts[1:]]
            points = [
                (coords[i] * width, coords[i + 1] * height)
                for i in range(0, len(coords), 2)
            ]
            draw.polygon(points, fill=(255, 80, 80, 60), outline=(255, 0, 0, 220))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def draw_coco_segmentation(image_path: Path, annotations: list[dict], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    for ann in annotations:
        segmentation = ann.get("segmentation", [])
        polygons = segmentation if isinstance(segmentation, list) else []
        for polygon in polygons:
            if len(polygon) < 6:
                continue
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=(80, 160, 255, 60), outline=(0, 80, 255, 220))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def visualize_countingpills(limit: int, seed: int) -> None:
    random.seed(seed)
    root = ROOT / "data" / "raw" / "CountingPills.v36i.yolov8"
    output_dir = ROOT / "data" / "samples" / "countingpills"
    candidates: list[tuple[Path, Path]] = []

    for split in ["train", "val", "test"]:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        for image_path in images_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            candidates.append((image_path, label_path))

    for idx, (image_path, label_path) in enumerate(random.sample(candidates, min(limit, len(candidates))), 1):
        draw_yolo_segmentation(image_path, label_path, output_dir / f"sample_{idx:02d}_{image_path.name}")


def visualize_mediseg(subset: str, limit: int, seed: int) -> None:
    random.seed(seed)
    root = ROOT / "data" / "raw" / "MEDISEG" / subset
    output_dir = ROOT / "data" / "samples" / f"mediseg_{subset}"
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for ann in data.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    images = [image for image in data.get("images", []) if annotations_by_image[int(image["id"])]]
    selected = random.sample(images, min(limit, len(images)))

    for idx, image_info in enumerate(selected, 1):
        image_path = root / "images" / image_info["file_name"]
        draw_coco_segmentation(
            image_path,
            annotations_by_image[int(image_info["id"])],
            output_dir / f"sample_{idx:02d}_{image_info['file_name']}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["countingpills", "mediseg_3pills", "mediseg_32pills", "all"], default="all")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.dataset in {"countingpills", "all"}:
        visualize_countingpills(args.limit, args.seed)
    if args.dataset in {"mediseg_3pills", "all"}:
        visualize_mediseg("3pills", args.limit, args.seed)
    if args.dataset in {"mediseg_32pills", "all"}:
        visualize_mediseg("32pills", args.limit, args.seed)

    print("Sample visualizations written to data/samples/.")


if __name__ == "__main__":
    main()

