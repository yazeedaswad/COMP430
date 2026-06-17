# Automated Pill Counting with Detection Baselines and a Proposed Image-Processing Method

This repository contains the COMP 430 image processing term project comparing two established pill-counting baselines with a proposed image-processing-based counting method.

## Goal

The project compares:

- **YOLO detector:** counts predicted pill bounding boxes.
- **Mask R-CNN:** counts predicted pill instance masks.
- **Proposed method:** predicts foreground and center maps, then uses thresholding, morphology, local-maxima extraction, non-maximum suppression, and marker-controlled watershed segmentation to count separated pill regions.

The proposed method is not claimed to outperform the baselines overall. Its purpose is to provide an interpretable image-processing pipeline for pill counting and touching-pill separation.

## Datasets

- `data/raw/CountingPills.v36i.yolov8`
  - Used for YOLO, Mask R-CNN, and the final proposed method in the main fair comparison.
- `data/raw/MEDISEG`
  - Used during initial Mask R-CNN baseline development and retained as auxiliary context.

Processed YOLO detection labels are stored in:

```text
data/processed/countingpills_detect
```

## Main Results

| Method | Dataset | Images | MAE | Exact count accuracy |
|---|---|---:|---:|---:|
| YOLO detector | CountingPills test | 175 | 0.0171 | 0.9829 |
| Mask R-CNN | CountingPills test | 175 | 0.2229 | 0.8171 |
| Proposed method | CountingPills test | 175 | 0.3143 | 0.7429 |

The proposed method test results are saved in:

```text
reports/tables/proposed_method_counting_test.csv
```


## Setup

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Core Commands

Train YOLO:

```powershell
python scripts\train_yolo_baseline.py --data configs\countingpills_detect_yolo.yaml --epochs 50 --batch 16 --device 0
```

Evaluate YOLO:

```powershell
python scripts\evaluate_yolo_counting.py --weights experiments\baseline_yolo\countingpills_yolo11n_detect_full\weights\best.pt --images data\processed\countingpills_detect\test\images --labels data\processed\countingpills_detect\test\labels --out reports\tables\yolo_counting_test.csv
```

Train Mask R-CNN:

```powershell
python scripts\train_mask_rcnn_baseline.py --dataset countingpills --epochs 1 --batch 2 --lr 0.005 --name countingpills_maskrcnn_e1 --device cuda
```

Evaluate Mask R-CNN on CountingPills:

```powershell
python scripts\evaluate_mask_rcnn_counting.py --dataset countingpills --split test --weights experiments\baseline_mask_rcnn\countingpills_maskrcnn_e1\weights\last.pt --score-threshold 0.5 --out reports\tables\mask_rcnn_countingpills_test.csv --device cuda
```

Train the proposed method:

```powershell
python scripts\train_center_segmenter.py --data-root data\raw\CountingPills.v36i.yolov8 --name countingpills_proposed_long_e60_bc16 --epochs 60 --batch 16 --lr 0.001 --weight-decay 0.0001 --center-loss-weight 4.0 --base-channels 16 --patience 12 --log-every 25 --device cuda
```

Evaluate the proposed method on test images:

```powershell
python scripts\evaluate_proposed_method.py --split test --weights experiments\proposed_pipeline\countingpills_tiny_unet_centers\weights\best.pt --foreground-threshold 0.5 --center-threshold 0.35 --peak-min-distance 32 --preprocessing none --out reports\tables\proposed_method_counting_test.csv --device cuda --batch 16
```

Run the proposed method on one image:

```powershell
python scripts\predict_proposed_method_image.py "path\to\image.jpg" --weights experiments\proposed_pipeline\countingpills_tiny_unet_centers\weights\best.pt --foreground-threshold 0.5 --center-threshold 0.35 --peak-min-distance 32 --name proposed_method_demo
```

## Repository Structure

```text
configs/              Dataset and experiment configuration files
data/                 Raw and processed datasets
experiments/          Training runs, weights, and prediction outputs
reports/              Final report, figures, and result tables
scripts/              Training, evaluation, prediction, and report scripts
src/pill_counting/    Project package
tests/                Smoke tests
```

## Submission Notes

Before submitting:

- Upload the project to a public GitHub repository.
- Open the report in Word and confirm it is between 6 and 10 pages.
