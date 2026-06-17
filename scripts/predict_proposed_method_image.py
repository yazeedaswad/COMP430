from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pill_counting.data.preprocessing import preprocess_rgb
from pill_counting.models.center_split import center_marker_watershed
from pill_counting.models.tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed center-marker method on a single image.")
    parser.add_argument("source")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "experiments" / "proposed_pipeline" / "countingpills_tiny_unet_centers" / "weights" / "best.pt"),
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--center-threshold", type=float, default=0.35)
    parser.add_argument("--peak-min-distance", type=int, default=24)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area-ratio", type=float, default=0.0)
    parser.add_argument("--min-solidity", type=float, default=0.0)
    parser.add_argument("--max-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--preprocessing", choices=["none"], default="none")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="center_test")
    return parser.parse_args()


def overlay(image: Image.Image, labels: np.ndarray, count: int) -> Image.Image:
    base = image.convert("RGB").convert("RGBA")
    rng = np.random.default_rng(9)
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in sorted(set(int(value) for value in np.unique(labels)) - {0}):
        color[labels == label_id] = rng.integers(40, 240, size=3, dtype=np.uint8)
    overlay_img = Image.fromarray(color).convert("RGBA")
    overlay_img.putalpha(Image.fromarray(np.where(labels > 0, 95, 0).astype(np.uint8)))
    composed = Image.alpha_composite(base, overlay_img)
    draw = ImageDraw.Draw(composed)
    for label_id in sorted(set(int(value) for value in np.unique(labels)) - {0}):
        contour_image = ((labels == label_id).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [(int(point[0][0]), int(point[0][1])) for point in contour]
            if len(pts) > 1:
                draw.line(pts + [pts[0]], fill=(255, 0, 0, 255), width=2)
    draw.rectangle((0, 0, 190, 30), fill=(255, 255, 255, 230))
    draw.text((8, 8), f"Proposed method count={count}", fill=(0, 0, 0, 255))
    return composed.convert("RGB")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    image_path = Path(args.source)
    image = Image.open(image_path).convert("RGB")
    image_array = preprocess_rgb(np.asarray(image, dtype=np.uint8), mode=args.preprocessing)
    tensor = torch.from_numpy(image_array.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = F.interpolate(tensor.unsqueeze(0), size=(args.image_size, args.image_size), mode="bilinear", align_corners=False).to(device)

    model = TinyUNet(out_channels=2, base_channels=args.base_channels).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor)).squeeze(0).detach().cpu().numpy()

    fg = cv2.resize(pred[0], image.size, interpolation=cv2.INTER_LINEAR)
    centers = cv2.resize(pred[1], image.size, interpolation=cv2.INTER_LINEAR)
    result = center_marker_watershed(
        fg,
        centers,
        foreground_threshold=args.foreground_threshold,
        center_threshold=args.center_threshold,
        peak_min_distance=args.peak_min_distance,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        min_solidity=args.min_solidity,
        max_aspect_ratio=args.max_aspect_ratio,
    )

    out_dir = ROOT / "experiments" / "proposed_pipeline" / "predictions" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{image_path.stem}_proposed_method.jpg"
    center_path = out_dir / f"{image_path.stem}_center_heatmap.png"
    mask_path = out_dir / f"{image_path.stem}_foreground_prob.png"
    overlay(image, result.labels, result.count).save(overlay_path)
    Image.fromarray((np.clip(centers, 0, 1) * 255).astype(np.uint8)).save(center_path)
    Image.fromarray((np.clip(fg, 0, 1) * 255).astype(np.uint8)).save(mask_path)

    print(f"predicted_count={result.count}")
    print(f"foreground_threshold={args.foreground_threshold}")
    print(f"center_threshold={args.center_threshold}")
    print(f"peak_min_distance={args.peak_min_distance}")
    print(f"preprocessing={args.preprocessing}")
    print(f"overlay={overlay_path}")
    print(f"center_heatmap={center_path}")
    print(f"foreground_prob={mask_path}")


if __name__ == "__main__":
    main()
