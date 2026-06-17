from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MedisegImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int


def load_coco_annotations(annotation_path: Path) -> tuple[list[MedisegImageRecord], dict[int, list[dict[str, Any]]], dict[int, int]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = [
        MedisegImageRecord(
            image_id=int(image["id"]),
            file_name=image["file_name"],
            width=int(image["width"]),
            height=int(image["height"]),
        )
        for image in data["images"]
    ]

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    category_ids = sorted({int(ann["category_id"]) for ann in data["annotations"]})
    category_to_label = {category_id: idx + 1 for idx, category_id in enumerate(category_ids)}

    for ann in data["annotations"]:
        annotations_by_image[int(ann["image_id"])].append(ann)

    return records, annotations_by_image, category_to_label


def split_records(
    records: list[MedisegImageRecord],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    split: str,
    seed: int = 7,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    limit: int | None = None,
) -> list[MedisegImageRecord]:
    annotated_records = [record for record in records if annotations_by_image.get(record.image_id)]
    rng = random.Random(seed)
    shuffled = annotated_records[:]
    rng.shuffle(shuffled)

    if limit is not None:
        shuffled = shuffled[:limit]

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if split == "train":
        return shuffled[:train_end]
    if split == "val":
        return shuffled[train_end:val_end]
    if split == "test":
        return shuffled[val_end:]
    raise ValueError(f"Unknown split: {split}")


def polygons_to_mask(polygons: list[Any], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask


class MedisegInstanceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        seed: int = 7,
        limit: int | None = None,
        binary_labels: bool = True,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.annotation_path = self.root / "annotations.json"
        self.records, self.annotations_by_image, self.category_to_label = load_coco_annotations(self.annotation_path)
        self.records = split_records(
            self.records,
            self.annotations_by_image,
            split=split,
            seed=seed,
            limit=limit,
        )
        self.binary_labels = binary_labels

    def __len__(self) -> int:
        return len(self.records)

    @property
    def num_classes(self) -> int:
        if self.binary_labels:
            return 2
        return len(self.category_to_label) + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        image_path = self.images_dir / record.file_name
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        annotations = self.annotations_by_image[record.image_id]
        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in annotations:
            mask = polygons_to_mask(ann.get("segmentation", []), record.height, record.width)
            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)
            x_min = float(xs.min())
            x_max = float(xs.max())
            y_min = float(ys.min())
            y_max = float(ys.max())

            if x_max <= x_min or y_max <= y_min:
                continue

            masks.append(mask)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1 if self.binary_labels else self.category_to_label[int(ann["category_id"])])
            areas.append(float(mask.sum()))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.as_tensor([record.image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)

