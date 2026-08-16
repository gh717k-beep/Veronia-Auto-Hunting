@echo off
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m pip install -q pynput pillow
    py -3 capture_mobs.py
) else (
    python -m pip install -q pynput pillow
    python capture_mobs.py
)

pause
