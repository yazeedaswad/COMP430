from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_files(path: Path, extensions: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def summarize_yolo_dataset(name: str, root: Path) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for split in ["train", "val", "test"]:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        image_count = count_files(images_dir, (".jpg", ".jpeg", ".png"))
        label_count = count_files(labels_dir, (".txt",))
        instances = 0
        empty_labels = 0

        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                instances += len(lines)
                if not lines:
                    empty_labels += 1

        avg_instances = instances / image_count if image_count else 0
        print(
            f"{split}: images={image_count}, labels={label_count}, "
            f"instances={instances}, avg_instances_per_image={avg_instances:.2f}, "
            f"empty_label_files={empty_labels}"
        )


def summarize_coco_dataset(name: str, root: Path) -> None:
    ann_path = root / "annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    by_image: dict[int, int] = defaultdict(int)
    by_category: Counter[int] = Counter()

    for ann in annotations:
        by_image[int(ann["image_id"])] += 1
        by_category[int(ann["category_id"])] += 1

    annotated_images = sum(1 for image in images if by_image[int(image["id"])] > 0)
    avg_instances = len(annotations) / len(images) if images else 0
    category_names = {int(cat["id"]): cat["name"] for cat in categories}

    print(f"\n{name}")
    print("-" * len(name))
    print(f"images={len(images)}")
    print(f"annotated_images={annotated_images}")
    print(f"annotations={len(annotations)}")
    print(f"categories={len(categories)}")
    print(f"avg_instances_per_image={avg_instances:.2f}")
    print("top_categories:")
    for category_id, count in by_category.most_common(8):
        print(f"  {category_names.get(category_id, category_id)}: {count}")


def main() -> None:
    summarize_yolo_dataset("CountingPills", ROOT / "data" / "raw" / "CountingPills.v36i.yolov8")
    summarize_coco_dataset("MEDISEG 3pills", ROOT / "data" / "raw" / "MEDISEG" / "3pills")
    summarize_coco_dataset("MEDISEG 32pills", ROOT / "data" / "raw" / "MEDISEG" / "32pills")


if __name__ == "__main__":
    main()

