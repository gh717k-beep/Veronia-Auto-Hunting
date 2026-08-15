$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host '.venv Python was not found.'
    Write-Host "Expected path: $python"
    Read-Host 'Press Enter to exit'
    exit 1
}

$runsRoot = Join-Path $root 'runs'
Write-Host '======================================================'
Write-Host ' Roboflow YOLOv8 Auto Training Helper'
Write-Host '======================================================'
Write-Host ''
Write-Host 'Current saved model list:'
if (Test-Path $runsRoot) {
    $items = Get-ChildItem $runsRoot -Directory | Where-Object { $_.Name -ne 'detect' } | Select-Object -ExpandProperty Name
    if ($items.Count -gt 0) {
        foreach ($name in $items) { Write-Host " - $name" }
    } else {
        Write-Host ' - none'
    }
} else {
    Write-Host ' - none'
}

Write-Host ''
$modelName = Read-Host 'Model name (example: active-indian-1)'
if ([string]::IsNullOrWhiteSpace($modelName)) {
    Write-Host 'Model name is required.'
    Read-Host 'Press Enter to exit'
    exit 1
}

$apiKey = Read-Host 'Roboflow API KEY'
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    Write-Host 'API KEY is required.'
    Read-Host 'Press Enter to exit'
    exit 1
}

$project = Read-Host 'PROJECT'
if ([string]::IsNullOrWhiteSpace($project)) {
    Write-Host 'PROJECT is required.'
    Read-Host 'Press Enter to exit'
    exit 1
}

Write-Host ''
Write-Host 'Starting training...'

& $python 'train_custom_model.py' $modelName $apiKey $project

if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'Training failed.'
    Read-Host 'Press Enter to exit'
    exit $LASTEXITCODE
}

Write-Host ''
Write-Host 'Training finished.'
Read-Host 'Press Enter to exit'
