@echo off
chcp 65001
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo .venv Python not found.
    echo Expected: %PYTHON%
    pause
    exit /b 1
)

echo Starting capture_mobs.py...
cmd /k ""%PYTHON%" "%~dp0capture_mobs.py""
