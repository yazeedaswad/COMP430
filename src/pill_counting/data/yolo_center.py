from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from pill_counting.data.preprocessing import preprocess_rgb
from pill_counting.data.yolo_foreground import IMAGE_EXTENSIONS, yolo_label_to_mask


def draw_gaussian(heatmap: np.ndarray, center_x: float, center_y: float, sigma: float) -> None:
    height, width = heatmap.shape
    radius = max(2, int(round(sigma * 3)))
    x_min = max(0, int(round(center_x)) - radius)
    x_max = min(width - 1, int(round(center_x)) + radius)
    y_min = max(0, int(round(center_y)) - radius)
    y_max = min(height - 1, int(round(center_y)) + radius)

    if x_max <= x_min or y_max <= y_min:
        return

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2))
    heatmap[y_min : y_max + 1, x_min : x_max + 1] = np.maximum(
        heatmap[y_min : y_max + 1, x_min : x_max + 1],
        gaussian.astype(np.float32),
    )


def yolo_label_to_center_heatmap(label_path: Path, height: int, width: int) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not label_path.exists():
        return heatmap

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        coords = [float(value) for value in parts[1:]]
        if len(coords) == 4:
            x_center = coords[0] * width
            y_center = coords[1] * height
            box_width = coords[2] * width
            box_height = coords[3] * height
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = np.asarray(coords[0::2], dtype=np.float32) * width
            ys = np.asarray(coords[1::2], dtype=np.float32) * height
            x_center = float(xs.mean())
            y_center = float(ys.mean())
            box_width = float(xs.max() - xs.min())
            box_height = float(ys.max() - ys.min())
        else:
            continue

        sigma = max(2.0, min(box_width, box_height) * 0.18)
        draw_gaussian(heatmap, x_center, y_center, sigma)

    return heatmap


class YoloCenterDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        limit: int | None = None,
        preprocessing: str = "none",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.image_size = image_size
        self.preprocessing = preprocessing
        self.image_paths = sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if limit is not None:
            self.image_paths = self.image_paths[:limit]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        foreground = yolo_label_to_mask(label_path, height=height, width=width).astype(np.float32)
        centers = yolo_label_to_center_heatmap(label_path, height=height, width=width)
        true_count = 0
        if label_path.exists():
            true_count = sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())

        image_array = preprocess_rgb(np.asarray(image, dtype=np.uint8), mode=self.preprocessing)
        image_tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).permute(2, 0, 1)
        target = torch.from_numpy(np.stack([foreground, centers], axis=0)).float()

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        target = F.interpolate(
            target.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        metadata = {
            "image_id": index,
            "file_name": image_path.name,
            "true_count": true_count,
            "original_height": height,
            "original_width": width,
        }
        return image_tensor, target, metadata


def yolo_center_collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int | str]]]:
    images, targets, metadata = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(targets)), list(metadata)
