import re
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"


def sanitize_model_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", cleaned)
    cleaned = cleaned.strip()
    return cleaned or "untitled_model"


def list_existing_models() -> list[str]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted(
        [p.name for p in MODELS_DIR.iterdir() if p.is_dir()],
        key=lambda s: s.lower(),
    )

    if not models:
        print("[기존 학습 모델 목록] 아직 저장된 모델이 없습니다.")
        return []

    print("[기존 학습 모델 목록]")
    for index, model in enumerate(models, start=1):
        print(f"  {index}. {model}")
    return models


def parse_rf_value(raw: str, label: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"{label} 값을 입력하지 않았습니다.")
    return value


def find_data_yaml(dataset_path: str | Path | object) -> Path:
    if hasattr(dataset_path, "location"):
        dataset_path = dataset_path.location

    base_dir = Path(dataset_path).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"다운로드된 데이터셋 경로를 찾지 못했습니다: {base_dir}")

    data_yaml = next(base_dir.rglob("data.yaml"), None)
    if data_yaml is None:
        raise FileNotFoundError(f"{base_dir} 안에서 data.yaml 파일을 찾지 못했습니다.")
    return data_yaml


def run_training(model_name: str, api_key: str, project_expr: str) -> None:
    api_key_value = parse_rf_value(api_key, "API KEY")
    project_value = parse_rf_value(project_expr, "PROJECT")

    if "Roboflow(api_key=" not in api_key_value and "api_key=" not in api_key_value:
        api_key_value = f'Roboflow(api_key="{api_key_value}")'

    try:
        rf = eval(api_key_value, {"Roboflow": Roboflow})
    except Exception as exc:
        raise ValueError(f"API KEY 형식이 올바르지 않습니다. 오류: {exc}") from exc

    if not project_value.startswith("rf."):
        raise ValueError("PROJECT 값은 rf.workspace(...).project(...) 형식이어야 합니다.")

    try:
        project = eval(project_value, {"rf": rf})
    except Exception as exc:
        raise ValueError(f"PROJECT 형식이 올바르지 않습니다. 오류: {exc}") from exc

    version = project.version(1)
    print("[Roboflow 데이터셋 다운로드 중]")
    dataset = version.download("yolov8")
    print(f"[다운로드 완료] {getattr(dataset, 'location', dataset)}")

    data_yaml = find_data_yaml(dataset)
    model_name = sanitize_model_name(model_name)
    save_dir = MODELS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[학습 시작]")
    print(f"모델 이름: {model_name}")
    print(f"저장 경로: {save_dir}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(MODELS_DIR),
        name=model_name,
        verbose=False,
    )

    print(f"[학습 완료] 모델 저장 위치: {save_dir}")


def prompt_value(label: str, allow_blank: bool = False) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        if allow_blank:
            return ""
        print(f"{label} 값을 입력해야 합니다.")


def main() -> None:
    print("========================================")
    print("YOLOv8 모델 학습 설정")
    print("========================================")
    list_existing_models()
    print("========================================")

    model_name = prompt_value("학습 모델 이름")
    api_key = prompt_value("API KEY")
    project_expr = prompt_value("PROJECT")

    print("\n[입력값 확인]")
    print(f"- 모델: {model_name}")
    print(f"- API KEY: {api_key}")
    print(f"- PROJECT: {project_expr}")
    print("========================================")

    run_training(model_name, api_key, project_expr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[오류] {exc}")
        input("엔터를 눌러 종료하세요...")
