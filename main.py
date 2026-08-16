import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO

try:
    import win32api
    import win32con
except ImportError:
    win32api = None
    win32con = None

ROOT = Path(__file__).resolve().parent
TRAINED_MODELS_DIR = ROOT / "trained_models"
RUNS_DIR = ROOT / "runs"
SCREEN_CENTER_X = 960
SCREEN_CENTER_Y = 540
EMA_ALPHA = 0.28
DECAY = 0.85
MAX_MISSED_FRAMES = 15


def list_model_paths() -> list[Path]:
    model_paths: list[Path] = []
    seen: set[str] = set()

    for base_dir in [TRAINED_MODELS_DIR, RUNS_DIR]:
        if not base_dir.exists():
            continue
        for weight_path in base_dir.rglob("best.pt"):
            if not weight_path.is_file():
                continue
            resolved = str(weight_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            model_paths.append(weight_path.resolve())

    return sorted(model_paths, key=lambda p: str(p).lower())


def display_model_list(model_paths: list[Path]) -> None:
    print("========================================")
    print("학습 모델 목록")
    print("========================================")
    for index, model_path in enumerate(model_paths, start=1):
        label = model_path.parent.parent.name if model_path.parent.name == "weights" else model_path.stem
        print(f"{index}. {label}")
    print("========================================")


def select_model_and_threshold() -> tuple[Path, float]:
    model_paths = list_model_paths()
    if not model_paths:
        print("[알림] 학습된 모델이 없습니다. 먼저 모델을 학습해 주세요.")
        raise SystemExit(1)

    display_model_list(model_paths)

    while True:
        raw_choice = input("모델 번호를 선택하세요: ").strip()
        try:
            choice_index = int(raw_choice) - 1
            if 0 <= choice_index < len(model_paths):
                selected_model = model_paths[choice_index]
                break
        except ValueError:
            pass
        print("올바른 번호를 입력해 주세요.")

    while True:
        raw_threshold = input("유사도 역치 값을 입력하세요 (기본값 0.60): ").strip() or "0.60"
        try:
            threshold = float(raw_threshold)
            if 0.0 <= threshold <= 1.0:
                break
        except ValueError:
            pass
        print("역치 값은 0.0 ~ 1.0 사이의 숫자로 입력해 주세요.")

    return selected_model, threshold


def move_mouse(delta_x: int, delta_y: int) -> None:
    if delta_x == 0 and delta_y == 0:
        return

    if win32api and win32con:
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(delta_x), int(delta_y), 0)
    else:
        ctypes.windll.user32.mouse_event(0x0001, int(delta_x), int(delta_y), 0, 0)


def capture_screen() -> np.ndarray:
    image = np.array(ImageGrab.grab())
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def select_main_target(detections: list[dict], threshold: float):
    candidates = [d for d in detections if d["confidence"] >= threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["area"])


def iterate_targets(results) -> list[dict]:
    detections: list[dict] = []

    for result in results:
        if result is None or result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0].cpu().item())
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            detections.append({
                "area": area,
                "confidence": conf,
                "cx": cx,
                "cy": cy,
            })

    return detections


def run_detection_loop(model_path: Path, threshold: float) -> None:
    print("========================================")
    print("State 1: 전투 및 고정밀 관성 추적 (Combat & Velocity Tracking Mode)")
    print("========================================")
    print(f"모델: {model_path}")
    print(f"유사도 역치: {threshold:.2f}")
    print("- YOLOv8 + ByteTrack 기준으로 가장 큰 바운딩 박스를 메인 타겟으로 선택합니다.")
    print("- Ctrl + C 로 종료합니다.")

    model = YOLO(str(model_path))

    ema_dx = 0.0
    ema_dy = 0.0
    last_vx = 0.0
    last_vy = 0.0
    missed_frames = 0

    try:
        while True:
            frame = capture_screen()
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=threshold,
                iou=0.45,
                imgsz=640,
                verbose=False,
            )

            detections = iterate_targets(results)
            target = select_main_target(detections, threshold)

            if target is not None:
                missed_frames = 0
                dx = target["cx"] - SCREEN_CENTER_X
                dy = target["cy"] - SCREEN_CENTER_Y

                ema_dx = EMA_ALPHA * dx + (1.0 - EMA_ALPHA) * ema_dx
                ema_dy = EMA_ALPHA * dy + (1.0 - EMA_ALPHA) * ema_dy

                move_mouse(int(ema_dx * 0.62), int(ema_dy * 0.62))

                last_vx = dx
                last_vy = dy
            else:
                missed_frames += 1
                if missed_frames <= MAX_MISSED_FRAMES:
                    predicted_dx = int(last_vx * 0.9)
                    predicted_dy = int(last_vy * 0.9)
                    move_mouse(predicted_dx, predicted_dy)
                    last_vx *= DECAY
                    last_vy *= DECAY
                else:
                    last_vx = 0.0
                    last_vy = 0.0

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[종료] 사용자가 중단했습니다.")


def main() -> None:
    model_path, threshold = select_model_and_threshold()
    run_detection_loop(model_path, threshold)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[오류] {exc}")
        input("엔터를 눌러 종료하세요...")
