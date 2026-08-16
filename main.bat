@echo off
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m pip install -q ultralytics pillow opencv-python numpy pyautogui pydirectinput pynput
    py -3 main.py
) else (
    python -m pip install -q ultralytics pillow opencv-python numpy pyautogui pydirectinput pynput
    python main.py
)

pause
