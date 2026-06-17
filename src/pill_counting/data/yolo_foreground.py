from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def yolo_label_to_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        coords = [float(value) for value in parts[1:]]
        if len(coords) == 4:
            x_center, y_center, box_width, box_height = coords
            x_min = int(round((x_center - box_width / 2) * width))
            x_max = int(round((x_center + box_width / 2) * width))
            y_min = int(round((y_center - box_height / 2) * height))
            y_max = int(round((y_center + box_height / 2) * height))
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1, thickness=-1)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = np.asarray(
                [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)],
                dtype=np.float32,
            )
            points = np.round(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)

    return mask


class YoloForegroundDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.image_size = image_size
        self.image_paths = sorted(
            path for path in self.images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit is not None:
            self.image_paths = self.image_paths[:limit]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        foreground = yolo_label_to_mask(label_path, height=height, width=width)
        true_count = 0
        if label_path.exists():
            true_count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(foreground).unsqueeze(0).float()

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        metadata = {
            "image_id": index,
            "file_name": image_path.name,
            "true_count": true_count,
            "original_height": height,
            "original_width": width,
        }
        return image_tensor, mask_tensor, metadata


def yolo_foreground_collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int | str]]]:
    images, masks, metadata = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(masks)), list(metadata)

