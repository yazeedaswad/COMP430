param(
    [int]$Epochs = 20,
    [int]$Batch = 16,
    [int]$ImageSize = 640,
    [int]$Workers = 0,
    [double]$Fraction = 1.0,
    [string]$Device = "0",
    [string]$Name = "countingpills_yolo11n_detect_full",
    [switch]$SaveLog
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"
$env:PYTHONIOENCODING = "utf-8"

chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$LogDir = Join-Path $ProjectRoot "experiments\baseline_yolo\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $LogDir "$Name-$Timestamp.log"

Write-Host "Starting YOLO baseline training"
Write-Host "Project root: $ProjectRoot"
Write-Host "Run name: $Name"
Write-Host "Epochs: $Epochs"
Write-Host "Batch: $Batch"
Write-Host "Image size: $ImageSize"
Write-Host "Device: $Device"
Write-Host "Fraction: $Fraction"
if ($SaveLog) {
    Write-Host "Log: $LogPath"
    Write-Host "Note: PowerShell logging may print progress updates on separate lines."
}
else {
    Write-Host "Log: disabled for cleaner progress bar rendering"
}
Write-Host ""

$TrainArgs = @(
    "-u", "$ProjectRoot\scripts\train_yolo_baseline.py",
    "--task", "detect",
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--imgsz", "$ImageSize",
    "--workers", "$Workers",
    "--fraction", "$Fraction",
    "--device", "$Device",
    "--name", "$Name"
)

if ($SaveLog) {
    python @TrainArgs 2>&1 | Tee-Object -FilePath $LogPath
}
else {
    python @TrainArgs
}
