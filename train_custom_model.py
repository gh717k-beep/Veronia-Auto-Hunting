import argparse
import re
import shutil
import sys
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def sanitize_model_name(name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]+', '_', name.strip())
    safe = safe.strip().strip('.')
    return safe or "custom_model"


def parse_api_key(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("API KEY 가 비어 있습니다.")

    if value.lower().startswith("roboflow("):
        match = re.search(r"Roboflow\s*\(\s*api_key\s*=\s*['\"]?([^'\")\]]+)['\"]?\s*\)", value, re.IGNORECASE)
        if match:
            return match.group(1)
        raise ValueError("API KEY 형식이 올바르지 않습니다. 예: Roboflow(api_key='...')")

    if "api_key=" in value.lower():
        match = re.search(r"api_key\s*=\s*['\"]?([^'\"\)]+)['\"]?", value, re.IGNORECASE)
        if match:
            return match.group(1)
        raise ValueError("api_key= 형식이 올바르지 않습니다.")

    return value


def parse_project(raw: str):
    value = raw.strip()
    if not value:
        raise ValueError("PROJECT 가 비어 있습니다.")

    normalized = value.replace("'", '"')

    if "rf.workspace" in normalized.lower():
        match = re.search(
            r'rf\.workspace\s*\(\s*"([^\"]+)"\s*\)\s*\.project\s*\(\s*"([^\"]+)"\s*\)',
            normalized,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        match = re.search(
            r'rf\.workspace\s*\(\s*([^\)\s]+)\s*\)\s*\.project\s*\(\s*([^\)\s]+)\s*\)',
            normalized,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        raise ValueError('PROJECT 는 "workspace/project" 또는 rf.workspace("workspace").project("project") 형식이어야 합니다.')

    if "/" in value:
        workspace, project = value.split("/", 1)
        return workspace.strip(), project.strip()

    raise ValueError('PROJECT 는 "workspace/project" 또는 rf.workspace("workspace").project("project") 형식이어야 합니다.')


def ensure_dataset(api_key: str, workspace_name: str, project_name: str):
    dataset_dir = ROOT / "roboflow_dataset"
    dataset_dir.mkdir(exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    versions = project.versions()
    if not versions:
        raise ValueError(f"Roboflow 프로젝트 '{workspace_name}/{project_name}' 에 버전이 없습니다.")

    latest_version = max(versions, key=lambda v: getattr(v, "number", getattr(v, "version", 1)))
    version_num = getattr(latest_version, "number", getattr(latest_version, "version", 1))
    print(f"[INFO] 의 최신 버전: {version_num}")

    version = project.version(version_num)
    version.download("yolov8", location=str(dataset_dir), overwrite=True)

    matches = sorted(dataset_dir.glob("**/data.yaml"))
    if not matches:
        raise FileNotFoundError("Roboflow 다운로드 후 data.yaml 을 찾지 못했습니다.")

    return matches[0], version_num


def main():
    parser = argparse.ArgumentParser(description="Roboflow YOLOv8 커스텀 학습 실행기")
    parser.add_argument("model_name", help="학습 모델 이름")
    parser.add_argument("api_key", help="Roboflow API KEY 또는 Roboflow(api_key='...')")
    parser.add_argument("project", help='PROJECT 형식: "workspace/project" 또는 rf.workspace("...").project("...")')
    parser.add_argument("--dry-run", action="store_true", help="실제 학습 대신 인자만 확인하고 끝냅니다.")
    args = parser.parse_args()

    try:
        api_key = parse_api_key(args.api_key)
        workspace_name, project_name = parse_project(args.project)
        model_name = args.model_name.strip()
        if not model_name:
            raise ValueError("모델 이름이 비어 있습니다.")

        print(f"[INFO] 모델 이름: {model_name}")
        print(f"[INFO] workspace: {workspace_name}")
        print(f"[INFO] project: {project_name}")

        if args.dry_run:
            print("[INFO] dry-run 모드: 실제 학습은 수행하지 않습니다.")
            return

        safe_name = sanitize_model_name(model_name)
        print(f"[INFO] 저장용 이름: {safe_name}")

        data_yaml, version_num = ensure_dataset(api_key, workspace_name, project_name)
        print(f"[INFO] 데이터셋 경로: {data_yaml}")

        runs_dir = ROOT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO("yolov8n.pt")
        print(f"\n[INFO] YOLO 학습 시작: {safe_name}")
        model.train(
            data=str(data_yaml),
            project=str(runs_dir),
            name=safe_name,
            epochs=50,
            imgsz=416,
            batch=8,
            device="cpu",
            patience=10,
            save=True,
            verbose=True,
            exist_ok=True,
        )

        candidate_models = [
            runs_dir / safe_name / "weights" / "best.pt",
            runs_dir / "detect" / safe_name / "weights" / "best.pt",
        ]
        final_model = next((p for p in candidate_models if p.exists()), candidate_models[0])
        if not final_model.exists():
            raise FileNotFoundError(
                f"학습 결과 모델을 찾지 못했습니다. 후보 경로: {', '.join(str(p) for p in candidate_models)}"
            )

        print("\n[INFO] 학습 완료")
        print(f"[INFO] 모델 저장 위치: {final_model}")
        print(f"[INFO] 사용 예시: model = YOLO(r\"{final_model}\")")

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
