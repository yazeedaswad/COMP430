param(
    [int]$Epochs = 10,
    [int]$Batch = 2,
    [double]$LearningRate = 0.005,
    [int]$Workers = 0,
    [string]$Device = "cuda",
    [string]$DataRoot = "data\raw\MEDISEG\3pills",
    [string]$Name = "mediseg3_maskrcnn_full",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:TORCH_HOME = Join-Path $ProjectRoot ".torch"
$env:PYTHONIOENCODING = "utf-8"

chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

Write-Host "Starting Mask R-CNN baseline training"
Write-Host "Project root: $ProjectRoot"
Write-Host "Data root: $DataRoot"
Write-Host "Run name: $Name"
Write-Host "Epochs: $Epochs"
Write-Host "Batch: $Batch"
Write-Host "Learning rate: $LearningRate"
Write-Host "Device: $Device"
if ($Limit -gt 0) {
    Write-Host "Limit: $Limit"
}
else {
    Write-Host "Limit: full dataset"
}
Write-Host ""

$TrainArgs = @(
    "-u", "$ProjectRoot\scripts\train_mask_rcnn_baseline.py",
    "--data-root", "$DataRoot",
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--lr", "$LearningRate",
    "--workers", "$Workers",
    "--device", "$Device",
    "--name", "$Name"
)

if ($Limit -gt 0) {
    $TrainArgs += @("--limit", "$Limit")
}

python @TrainArgs

