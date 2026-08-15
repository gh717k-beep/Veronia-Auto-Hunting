from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ROBOFLOW_DATASET_DIR = ROOT / "roboflow_dataset"
ROBOFLOW_API_KEY = "YV337YEdjsbxNRxZhQgS"
WORKSPACE_NAME = "kimgunho717-gmail-com"
PROJECT_NAME = "veronia-auto-hunting"
VERSION_NUMBER = 1


def get_data_yaml() -> Path:
    existing = sorted(ROOT.glob("**/data.yaml"))
    if existing:
        return existing[0]

    ROBOFLOW_DATASET_DIR.mkdir(exist_ok=True)
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    version = project.version(VERSION_NUMBER)
    version.download("yolov8", location=str(ROBOFLOW_DATASET_DIR))

    downloaded = sorted(ROBOFLOW_DATASET_DIR.glob("**/data.yaml"))
    if not downloaded:
        raise FileNotFoundError("Roboflow 데이터셋이 다운로드되지 않았습니다.")

    return downloaded[0]


DATA_YAML = get_data_yaml()
print(f"데이터셋 경로: {DATA_YAML}")

# YOLOv8n 모델 로드
model = YOLO("yolov8n.pt")

# 모델 학습
print("\n[YOLO 학습 시작]")
model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=416,
    batch=8,
    device="cpu",
    patience=10,
    save=True,
    verbose=True,
)

print("\n[학습 완료]")
print(f"모델 저장 위치: {ROOT / 'runs'}")
