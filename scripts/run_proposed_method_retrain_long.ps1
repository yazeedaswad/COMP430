$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\train_center_segmenter.py `
  --data-root data\raw\CountingPills.v36i.yolov8 `
  --name countingpills_proposed_long_e60_bc16 `
  --epochs 60 `
  --batch 16 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --center-loss-weight 4.0 `
  --base-channels 16 `
  --patience 12 `
  --min-delta 0.0001 `
  --log-every 25 `
  --device cuda

python scripts\evaluate_proposed_method.py `
  --split val `
  --weights experiments\proposed_pipeline\countingpills_proposed_long_e60_bc16\weights\best.pt `
  --foreground-threshold 0.5 `
  --center-threshold 0.35 `
  --peak-min-distance 32 `
  --base-channels 16 `
  --preprocessing none `
  --out reports\tables\proposed_method_long_e60_bc16_val.csv `
  --device cuda

python scripts\evaluate_proposed_method.py `
  --split test `
  --weights experiments\proposed_pipeline\countingpills_proposed_long_e60_bc16\weights\best.pt `
  --foreground-threshold 0.5 `
  --center-threshold 0.35 `
  --peak-min-distance 32 `
  --base-channels 16 `
  --preprocessing none `
  --out reports\tables\proposed_method_long_e60_bc16_test.csv `
  --device cuda
