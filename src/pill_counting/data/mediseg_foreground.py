from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pill_counting.data.mediseg import MedisegInstanceDataset


class MedisegForegroundDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        seed: int = 7,
        limit: int | None = None,
        image_size: int = 256,
    ) -> None:
        self.instance_dataset = MedisegInstanceDataset(root, split, seed=seed, limit=limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.instance_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        image, target = self.instance_dataset[index]
        masks = target["masks"].float()
        foreground = (masks.sum(dim=0, keepdim=True) > 0).float()
        original_height = int(image.shape[1])
        original_width = int(image.shape[2])

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        foreground = F.interpolate(
            foreground.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        metadata = {
            "image_id": int(target["image_id"].item()),
            "true_count": int(target["masks"].shape[0]),
            "original_height": original_height,
            "original_width": original_width,
        }
        return image, foreground, metadata


def foreground_collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, int]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    images, masks, metadata = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(masks)), list(metadata)

