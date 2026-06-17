from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.models.mask_rcnn import create_mask_rcnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mask R-CNN on one image.")
    parser.add_argument("source")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "baseline_mask_rcnn" / "predictions" / "prediction.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    image = Image.open(args.source).convert("RGB")
    image_tensor = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    image_tensor = (image_tensor / 255.0).permute(2, 0, 1).to(device)

    model = create_mask_rcnn(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model([image_tensor])[0]

    scores = output.get("scores", torch.empty(0, device=device))
    keep = scores >= args.score_threshold
    boxes = output["boxes"][keep].detach().cpu().numpy()

    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    for box in boxes:
        draw.rectangle(tuple(box.tolist()), outline="red", width=3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drawn.save(out_path)
    print(f"predicted_count={len(boxes)}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
