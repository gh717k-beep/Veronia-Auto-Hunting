@echo off
chcp 65001
cd /d "%~dp0"

echo [1/3] ultralytics 설치 중...
"C:\Users\gh717\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install -q ultralytics

echo [2/3] 데이터셋 분할 중...
"C:\Users\gh717\AppData\Local\Python\pythoncore-3.14-64\python.exe" train_yolo.py

echo [3/3] YOLO 학습 시작...
"C:\Users\gh717\AppData\Local\Python\pythoncore-3.14-64\python.exe" run_training.py

pause
