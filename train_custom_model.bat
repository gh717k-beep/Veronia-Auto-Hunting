@echo off
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m pip install -q roboflow ultralytics
    py -3 train_custom_model.py
) else (
    python -m pip install -q roboflow ultralytics
    python train_custom_model.py
)

pause
