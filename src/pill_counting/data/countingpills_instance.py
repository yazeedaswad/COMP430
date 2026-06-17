from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pill_counting.data.yolo_foreground import IMAGE_EXTENSIONS


def _polygon_to_mask(coords: list[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(
        [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)],
        dtype=np.float32,
    )
    points = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def _box_to_mask(coords: list[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x_center, y_center, box_width, box_height = coords
    x_min = max(0, int(round((x_center - box_width / 2) * width)))
    x_max = min(width - 1, int(round((x_center + box_width / 2) * width)))
    y_min = max(0, int(round((y_center - box_height / 2) * height)))
    y_max = min(height - 1, int(round((y_center + box_height / 2) * height)))
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
    return mask


def load_yolo_instance_masks(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    if not label_path.exists():
        return []

    masks: list[np.ndarray] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        coords = [float(value) for value in parts[1:]]
        if len(coords) == 4:
            mask = _box_to_mask(coords, height, width)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            mask = _polygon_to_mask(coords, height, width)
        else:
            continue
        if int(mask.sum()) > 0:
            masks.append(mask)
    return masks


class CountingPillsInstanceDataset(Dataset):
    def __init__(self, root: str | Path, split: str, limit: int | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.image_paths = sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if limit is not None:
            self.image_paths = self.image_paths[:limit]

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def num_classes(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        masks_np = load_yolo_instance_masks(label_path, height=height, width=width)

        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        areas: list[float] = []
        for mask in masks_np:
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min = float(xs.min())
            x_max = float(xs.max())
            y_min = float(ys.min())
            y_max = float(ys.max())
            if x_max <= x_min or y_max <= y_min:
                continue
            masks.append(mask)
            boxes.append([x_min, y_min, x_max, y_max])
            areas.append(float(mask.sum()))

        if masks:
            mask_tensor = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            area_tensor = torch.as_tensor(areas, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            area_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": box_tensor,
            "labels": torch.ones((box_tensor.shape[0],), dtype=torch.int64),
            "masks": mask_tensor,
            "image_id": torch.as_tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": torch.zeros((box_tensor.shape[0],), dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
