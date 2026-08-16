import ctypes
import re
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import torch
from pynput import keyboard
from ultralytics import YOLO

try:
    import easyocr
except ImportError:
    easyocr = None

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
ACTIVE_DETECTION = False
EXIT_REQUESTED = False
TARGET_ACTION_SIZE = 37000
CLICK_INTERVAL_SECONDS = 0.12
FRAME_INTERVAL_SECONDS = 0.08
AUTO_HEAL_INTERVAL_SECONDS = 35.0
AUTO_HEAL_RIGHT_CLICK_GAP_SECONDS = 0.3
LAST_CLICK_TIME = 0.0
LAST_DETECTION_TIME = 0.0
AUTO_HEAL_MACRO_LAST_RUN = 0.0
DETECTION_LOG_COUNTER = 0
W_KEY_PRESSED = False
USER_W_PRESSED = False
S_KEY_PRESSED = False
KEYBOARD_CONTROLLER = keyboard.Controller()
SCREEN_CAPTURE = mss.mss()
OCR_READER = None
DIVE_MODE_ACTIVE = False
DIVE_MODE_STARTED_AT = 0.0
LAST_TARGET_TIME = 0.0
LAST_PITCH_OCR_TIME = 0.0
LAST_PITCH_VALUE = 0


def detect_device():
    """CUDA 호환성을 검사하고 사용 가능한 장치를 반환"""
    if not torch.cuda.is_available():
        print("[장치] CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        return 'cpu'

    try:
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        print("[장치] CUDA 호환성 확인 완료. GPU 모드로 실행합니다.")
        return 0
    except RuntimeError as e:
        print(f"[장치] CUDA 호환성 오류: {e}")
        print("[장치] CPU 모드로 폴백합니다.")
        return 'cpu'


YOLO_DEVICE = detect_device()


def on_w_key_down(key):
    global USER_W_PRESSED
    try:
        if key.char == 'w':
            USER_W_PRESSED = True
    except (AttributeError, TypeError):
        pass


def on_w_key_up(key):
    global USER_W_PRESSED
    try:
        if key.char == 'w':
            USER_W_PRESSED = False
    except (AttributeError, TypeError):
        pass


def set_w_key_state(should_press: bool) -> None:
    global W_KEY_PRESSED, USER_W_PRESSED
    effective_should_press = should_press or USER_W_PRESSED
    
    if effective_should_press and not W_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.press('w')
            W_KEY_PRESSED = True
        except Exception:
            pass
    elif not effective_should_press and W_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.release('w')
            W_KEY_PRESSED = False
        except Exception:
            pass


def press_w_key() -> None:
    set_w_key_state(True)


def release_w_key() -> None:
    set_w_key_state(False)


def force_release_w_key() -> None:
    """사용자 입력 무시하고 W를 강제 해제 (큰 타겟에서 클릭 모드로 전환)"""
    global W_KEY_PRESSED
    if W_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.release('w')
            W_KEY_PRESSED = False
        except Exception:
            pass


def click_left_mouse() -> None:
    global LAST_CLICK_TIME
    now = time.monotonic()
    if now - LAST_CLICK_TIME < CLICK_INTERVAL_SECONDS:
        return
    LAST_CLICK_TIME = now

    if win32api and win32con:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0)
    else:
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)


def click_right_mouse() -> None:
    if win32api and win32con:
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0)
    else:
        ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)


def run_auto_heal_macro() -> None:
    global AUTO_HEAL_MACRO_LAST_RUN
    AUTO_HEAL_MACRO_LAST_RUN = time.monotonic()

    print("[자동 힐] 1번 → 우클릭 → 5초 대기 → 2번 → 우클릭 x3 → 3번")

    KEYBOARD_CONTROLLER.tap('1')
    click_right_mouse()
    time.sleep(5.0)

    KEYBOARD_CONTROLLER.tap('2')
    for _ in range(3):
        click_right_mouse()
        time.sleep(AUTO_HEAL_RIGHT_CLICK_GAP_SECONDS)

    KEYBOARD_CONTROLLER.tap('3')


def set_s_key_state(should_press: bool) -> None:
    global S_KEY_PRESSED
    if should_press and not S_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.press('s')
            S_KEY_PRESSED = True
        except Exception:
            pass
    elif not should_press and S_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.release('s')
            S_KEY_PRESSED = False
        except Exception:
            pass


def toggle_detection():
    global ACTIVE_DETECTION, AUTO_HEAL_MACRO_LAST_RUN
    ACTIVE_DETECTION = not ACTIVE_DETECTION
    if ACTIVE_DETECTION:
        AUTO_HEAL_MACRO_LAST_RUN = time.monotonic()
    status = "활성화" if ACTIVE_DETECTION else "비활성화"
    print(f"[F8] 탐지 {status}")


def request_exit():
    global EXIT_REQUESTED
    EXIT_REQUESTED = True
    print("[F9] 종료 요청")


def on_key_press(key):
    try:
        if key == keyboard.Key.f8:
            toggle_detection()
        elif key == keyboard.Key.f9:
            request_exit()
    except AttributeError:
        pass


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
    monitor = SCREEN_CAPTURE.monitors[1]
    shot = np.ascontiguousarray(np.array(SCREEN_CAPTURE.grab(monitor), dtype=np.uint8))
    image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image)


def get_ocr_reader():
    global OCR_READER
    if OCR_READER is not None:
        return OCR_READER
    if easyocr is None:
        print("[OCR] easyocr가 설치되지 않아 Pitch OCR을 비활성화합니다.")
        OCR_READER = False
        return False
    try:
        OCR_READER = easyocr.Reader(['en'], gpu=YOLO_DEVICE != 'cpu')
        print("[OCR] Pitch OCR 초기화 완료")
        return OCR_READER
    except Exception as exc:
        print(f"[OCR] OCR 초기화 실패: {exc}")
        OCR_READER = False
        return False


def read_pitch_value(frame: np.ndarray):
    return LAST_PITCH_VALUE


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
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0].detach().cpu().item())
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
    global ACTIVE_DETECTION, EXIT_REQUESTED, LAST_DETECTION_TIME, DETECTION_LOG_COUNTER
    global DIVE_MODE_ACTIVE, DIVE_MODE_STARTED_AT, LAST_TARGET_TIME, LAST_PITCH_OCR_TIME, LAST_PITCH_VALUE

    print("========================================")
    print("State 1: 전투 및 고정밀 관성 추적 (Combat & Velocity Tracking Mode)")
    print("========================================")
    print(f"모델: {model_path}")
    print(f"유사도 역치: {threshold:.2f}")
    print("- YOLOv8 + ByteTrack 기준으로 가장 큰 바운딩 박스를 메인 타겟으로 선택합니다.")
    print("- F8: 탐지 활성화/비활성화")
    print("- F9: 종료")

    model = YOLO(str(model_path))
    if YOLO_DEVICE != 'cpu':
        model.to(f'cuda:{YOLO_DEVICE}')
        model.half()
    model.eval()

    hotkeys = keyboard.GlobalHotKeys({
        '<f8>': toggle_detection,
        '<f9>': request_exit,
    })
    hotkeys.start()

    listener = keyboard.Listener(on_press=on_w_key_down, on_release=on_w_key_up)
    listener.start()

    ema_dx = 0.0
    ema_dy = 0.0
    last_vx = 0.0
    last_vy = 0.0
    missed_frames = 0
    frame_counter = 0
    LAST_TARGET_TIME = time.monotonic()
    dive_phase_start = time.monotonic()
    dive_phase_up = True

    try:
        while not EXIT_REQUESTED:
            if not ACTIVE_DETECTION:
                DIVE_MODE_ACTIVE = False
                force_release_w_key()
                time.sleep(0.02)
                continue

            now = time.monotonic()
            if now - AUTO_HEAL_MACRO_LAST_RUN >= AUTO_HEAL_INTERVAL_SECONDS:
                macro_thread = threading.Thread(target=run_auto_heal_macro, daemon=True)
                macro_thread.start()

            if now - LAST_DETECTION_TIME < FRAME_INTERVAL_SECONDS:
                time.sleep(0.01)
                continue
            LAST_DETECTION_TIME = now

            with torch.inference_mode():
                frame = capture_screen()
                results = model.predict(
                    frame,
                    conf=threshold,
                    iou=0.45,
                    imgsz=640,
                    device=0 if YOLO_DEVICE != 'cpu' else 'cpu',
                    verbose=False,
                    max_det=50,
                )

                detections = iterate_targets(results)
                target = select_main_target(detections, threshold)

            if target is not None:
                LAST_TARGET_TIME = time.monotonic()
                DIVE_MODE_ACTIVE = False
                missed_frames = 0
                dx = target["cx"] - SCREEN_CENTER_X
                dy = target["cy"] - SCREEN_CENTER_Y

                ema_dx = EMA_ALPHA * dx + (1.0 - EMA_ALPHA) * ema_dx
                ema_dy = EMA_ALPHA * dy + (1.0 - EMA_ALPHA) * ema_dy

                mouse_move_x = int(ema_dx * 0.62)
                mouse_move_y = int(ema_dy * 0.62)
                is_target_offset = abs(dx) > 2.0 or abs(dy) > 2.0

                move_mouse(mouse_move_x, mouse_move_y)

                DETECTION_LOG_COUNTER += 1

                if is_target_offset and target["area"] <= TARGET_ACTION_SIZE:
                    set_s_key_state(False)
                    press_w_key()
                else:
                    if target["area"] > TARGET_ACTION_SIZE:
                        # 객체가 너무 커서 왼쪽 클릭 연타만으로는 맞지 않을 때,
                        # 크기가 역치에 수렴할 때까지 S를 꾹 누른 상태를 유지한다.
                        set_s_key_state(True)
                        force_release_w_key()
                        click_left_mouse()
                    else:
                        set_s_key_state(False)
                        release_w_key()

                last_vx = dx
                last_vy = dy
            else:
                set_s_key_state(False)
                missed_frames += 1
                if (time.monotonic() - LAST_TARGET_TIME) >= 1.0:
                    if not DIVE_MODE_ACTIVE:
                        DIVE_MODE_ACTIVE = True
                        DIVE_MODE_STARTED_AT = time.monotonic()
                        dive_phase_start = time.monotonic()
                        dive_phase_up = True

                    force_release_w_key()
                    now_dive = time.monotonic()
                    elapsed = now_dive - dive_phase_start
                    phase_duration = 6.0

                    if elapsed >= phase_duration:
                        dive_phase_start = now_dive
                        dive_phase_up = not dive_phase_up

                    if dive_phase_up:
                        move_mouse(10, -28)
                    else:
                        move_mouse(10, 28)
                    continue

                force_release_w_key()
                if missed_frames <= MAX_MISSED_FRAMES:
                    predicted_dx = int(last_vx * 0.9)
                    predicted_dy = int(last_vy * 0.9)
                    move_mouse(predicted_dx, predicted_dy)
                    last_vx *= DECAY
                    last_vy *= DECAY
                else:
                    last_vx = 0.0
                    last_vy = 0.0

            del frame, results, detections, target
            frame_counter += 1
            if frame_counter % 300 == 0 and YOLO_DEVICE != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[종료] 사용자가 중단했습니다.")
    finally:
        set_s_key_state(False)
        force_release_w_key()
        try:
            hotkeys.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass
        try:
            SCREEN_CAPTURE.close()
        except Exception:
            pass
        print("\n[종료] 프로그램을 종료합니다.")


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
