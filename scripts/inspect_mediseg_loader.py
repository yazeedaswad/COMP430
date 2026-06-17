from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.mediseg import MedisegInstanceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data" / "raw" / "MEDISEG" / "3pills"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    dataset = MedisegInstanceDataset(args.root, args.split, limit=args.limit)
    image, target = dataset[0]
    print(f"dataset_size={len(dataset)}")
    print(f"num_classes={dataset.num_classes}")
    print(f"image_shape={tuple(image.shape)}")
    print(f"boxes_shape={tuple(target['boxes'].shape)}")
    print(f"masks_shape={tuple(target['masks'].shape)}")
    print(f"labels={target['labels'].tolist()}")
    print(f"image_id={target['image_id'].item()}")


if __name__ == "__main__":
    main()

