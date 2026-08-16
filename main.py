import ctypes
import gc
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
AUTO_REPAIR_INTERVAL_SECONDS = 160.0
LAST_CLICK_TIME = 0.0
LAST_DETECTION_TIME = 0.0
AUTO_HEAL_MACRO_LAST_RUN = 0.0
AUTO_REPAIR_MACRO_LAST_RUN = 0.0
DETECTION_LOG_COUNTER = 0

# 키보드 입력 및 매크로 상태 관리
W_KEY_PRESSED = False
USER_W_PRESSED = False
S_KEY_PRESSED = False
SHIFT_KEY_PRESSED = False
IS_HEALING_MACRO_RUNNING = False   # 힐 매크로 플래그
IS_REPAIRING_MACRO_RUNNING = False # 수리 매크로 플래그
KEYBOARD_CONTROLLER = keyboard.Controller()

SCREEN_CAPTURE = None
MONITOR_BOUNDS = None

OCR_READER = None
DIVE_MODE_ACTIVE = False
DIVE_MODE_STARTED_AT = 0.0
LAST_TARGET_TIME = 0.0
LAST_PITCH_OCR_TIME = 0.0
LAST_PITCH_VALUE = 0


def detect_device():
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


def get_ocr_reader():
    global OCR_READER
    if OCR_READER is None and easyocr is not None:
        use_gpu = True if YOLO_DEVICE != 'cpu' else False
        OCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
    return OCR_READER


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
    global W_KEY_PRESSED, USER_W_PRESSED, IS_REPAIRING_MACRO_RUNNING
    if IS_REPAIRING_MACRO_RUNNING:
        should_press = False

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
    global W_KEY_PRESSED
    if W_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.release('w')
            W_KEY_PRESSED = False
        except Exception:
            pass


def set_s_key_state(should_press: bool) -> None:
    global S_KEY_PRESSED, IS_REPAIRING_MACRO_RUNNING
    if IS_REPAIRING_MACRO_RUNNING:
        should_press = False

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


def set_shift_key_state(should_press: bool) -> None:
    global SHIFT_KEY_PRESSED, IS_HEALING_MACRO_RUNNING, IS_REPAIRING_MACRO_RUNNING
    if IS_HEALING_MACRO_RUNNING or IS_REPAIRING_MACRO_RUNNING:
        should_press = False

    if should_press and not SHIFT_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.press(keyboard.Key.shift)
            SHIFT_KEY_PRESSED = True
        except Exception:
            pass
    elif not should_press and SHIFT_KEY_PRESSED:
        try:
            KEYBOARD_CONTROLLER.release(keyboard.Key.shift)
            SHIFT_KEY_PRESSED = False
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


def stop_all_actions() -> None:
    """모든 조작 키 입력 해제 함수"""
    force_release_w_key()
    set_s_key_state(False)
    set_shift_key_state(False)


def run_auto_heal_macro() -> None:
    global AUTO_HEAL_MACRO_LAST_RUN, IS_HEALING_MACRO_RUNNING
    AUTO_HEAL_MACRO_LAST_RUN = time.monotonic()
    IS_HEALING_MACRO_RUNNING = True

    set_shift_key_state(False)
    print("[자동 힐] 1번 → 우클릭 → 5초 대기 → 2번 → 우클릭 x3 → 3번 (Shift 차단)")

    try:
        KEYBOARD_CONTROLLER.tap('1')
        click_right_mouse()
        time.sleep(5.0)

        KEYBOARD_CONTROLLER.tap('2')
        for _ in range(3):
            click_right_mouse()
            time.sleep(AUTO_HEAL_RIGHT_CLICK_GAP_SECONDS)

        KEYBOARD_CONTROLLER.tap('3')
    finally:
        IS_HEALING_MACRO_RUNNING = False


def run_auto_repair_macro() -> None:
    """수리 매크로: 하던 모든 행동을 정지하고 9번 -> 0.5s -> 우클릭 -> 1s -> 3번 순서로 수행"""
    global AUTO_REPAIR_MACRO_LAST_RUN, IS_REPAIRING_MACRO_RUNNING
    AUTO_REPAIR_MACRO_LAST_RUN = time.monotonic()
    IS_REPAIRING_MACRO_RUNNING = True

    # 현재 입력 중인 이동/조작 키 즉시 정지
    stop_all_actions()

    print("[자동 수리] 모든 동작 정지 ➔ 9번 ➔ (0.5초) ➔ 우클릭 ➔ (1초) ➔ 3번")

    try:
        KEYBOARD_CONTROLLER.tap('9')
        time.sleep(0.5)

        click_right_mouse()
        time.sleep(1.0)

        KEYBOARD_CONTROLLER.tap('3')
    finally:
        IS_REPAIRING_MACRO_RUNNING = False


def toggle_detection():
    global ACTIVE_DETECTION, AUTO_HEAL_MACRO_LAST_RUN, AUTO_REPAIR_MACRO_LAST_RUN
    ACTIVE_DETECTION = not ACTIVE_DETECTION
    if ACTIVE_DETECTION:
        AUTO_HEAL_MACRO_LAST_RUN = time.monotonic()
        AUTO_REPAIR_MACRO_LAST_RUN = time.monotonic()
    status = "활성화" if ACTIVE_DETECTION else "비활성화"
    print(f"[F8] 탐지 {status}")


def request_exit():
    global EXIT_REQUESTED
    EXIT_REQUESTED = True
    print("[F9] 종료 요청")


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
    global SCREEN_CAPTURE, MONITOR_BOUNDS
    if SCREEN_CAPTURE is None:
        SCREEN_CAPTURE = mss.mss()
        MONITOR_BOUNDS = SCREEN_CAPTURE.monitors[1]
    
    shot = np.array(SCREEN_CAPTURE.grab(MONITOR_BOUNDS), dtype=np.uint8)
    return shot[:, :, :3]


def read_pitch_from_screen(frame: np.ndarray) -> int | None:
    reader = get_ocr_reader()
    if reader is None:
        return None

    pitch_roi = frame[220:300, 0:300]
    if pitch_roi.size == 0:
        return None

    gray = cv2.cvtColor(pitch_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 170, 255, cv2.THRESH_BINARY)

    results = reader.readtext(thresh, detail=0, allowlist='Pitch:0123456789- (negativeY)')

    del pitch_roi, gray, resized, thresh

    if not results:
        print("[OCR 로그] 텍스트 인식 실패 (ROI 영역 내 글자를 감지하지 못함)")
        return None

    for text in results:
        print(f"[OCR 텍스트 감지] 원본 텍스트: '{text}'")
        match = re.search(r'Pitch:\s*(-?\d+)', text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                print(f"  └─► [Pitch 파싱 성공] 현재 인식된 Pitch 값: {val}")
                return val
            except ValueError:
                pass

    print("  └─► [파싱 실패] Pitch 패턴을 찾지 못했습니다.")
    return None


def select_main_target(detections: list[dict], threshold: float):
    candidates = [d for d in detections if d["confidence"] >= threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["area"])


def iterate_targets(results) -> list[dict]:
    detections: list[dict] = []

    for result in results:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue

        boxes_data = result.boxes.data.cpu().numpy()
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls_id = box[:6]
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            area = width * height
            cx = float(x1 + x2) / 2.0
            cy = float(y1 + y2) / 2.0

            detections.append({
                "area": area,
                "confidence": float(conf),
                "cx": cx,
                "cy": cy,
            })

    return detections


def run_detection_loop(model_path: Path, threshold: float) -> None:
    global ACTIVE_DETECTION, EXIT_REQUESTED, LAST_DETECTION_TIME, DETECTION_LOG_COUNTER
    global DIVE_MODE_ACTIVE, DIVE_MODE_STARTED_AT, LAST_TARGET_TIME, LAST_PITCH_OCR_TIME, LAST_PITCH_VALUE
    global IS_REPAIRING_MACRO_RUNNING

    print("========================================")
    print("State 1: 전투 및 고정밀 관성 추적 (Combat & Velocity Tracking Mode)")
    print("========================================")
    print(f"모델: {model_path}")
    print(f"유사도 역치: {threshold:.2f}")

    if YOLO_DEVICE != 'cpu':
        torch.backends.cudnn.benchmark = True

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

    try:
        while not EXIT_REQUESTED:
            if not ACTIVE_DETECTION:
                DIVE_MODE_ACTIVE = False
                stop_all_actions()
                gc.collect()
                if YOLO_DEVICE != 'cpu' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.02)
                continue

            now = time.monotonic()
            if now - AUTO_HEAL_MACRO_LAST_RUN >= AUTO_HEAL_INTERVAL_SECONDS:
                threading.Thread(target=run_auto_heal_macro, daemon=True).start()

            if now - AUTO_REPAIR_MACRO_LAST_RUN >= AUTO_REPAIR_INTERVAL_SECONDS:
                threading.Thread(target=run_auto_repair_macro, daemon=True).start()

            # 자동 수리 수행 중에는 일반 조작 및 마우스 추적 완전 대기
            if IS_REPAIRING_MACRO_RUNNING:
                time.sleep(0.05)
                continue

            elapsed_time = now - LAST_DETECTION_TIME
            if elapsed_time < FRAME_INTERVAL_SECONDS:
                time.sleep(max(0.001, FRAME_INTERVAL_SECONDS - elapsed_time))
                continue
            LAST_DETECTION_TIME = time.monotonic()

            frame = capture_screen()

            with torch.inference_mode():
                results = model.predict(
                    frame,
                    conf=threshold,
                    iou=0.45,
                    imgsz=640,
                    device=0 if YOLO_DEVICE != 'cpu' else 'cpu',
                    verbose=False,
                    max_det=20,
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

                if is_target_offset and target["area"] <= TARGET_ACTION_SIZE:
                    set_s_key_state(False)
                    set_shift_key_state(False)
                    press_w_key()
                else:
                    if target["area"] > TARGET_ACTION_SIZE:
                        set_s_key_state(True)
                        set_shift_key_state(True)
                        force_release_w_key()
                        click_left_mouse()
                    else:
                        set_s_key_state(False)
                        set_shift_key_state(False)
                        release_w_key()

                last_vx = dx
                last_vy = dy
            else:
                set_s_key_state(False)
                set_shift_key_state(False)
                missed_frames += 1

                if (now - LAST_TARGET_TIME) >= 1.0:
                    if not DIVE_MODE_ACTIVE:
                        DIVE_MODE_ACTIVE = True
                        DIVE_MODE_STARTED_AT = now
                        print("\n[상태 전환] 타겟 상실 -> 잠수 모드(Dive Mode) 진입")

                    force_release_w_key()

                    if now - LAST_PITCH_OCR_TIME >= 0.25:
                        LAST_PITCH_OCR_TIME = now
                        parsed_pitch = read_pitch_from_screen(frame)
                        if parsed_pitch is not None:
                            LAST_PITCH_VALUE = parsed_pitch

                    mouse_dx = 10
                    mouse_dy = 0

                    if LAST_PITCH_VALUE > 0:
                        mouse_dy = -15
                    elif LAST_PITCH_VALUE < 0:
                        mouse_dy = 15
                    else:
                        mouse_dy = 0

                    move_mouse(mouse_dx, mouse_dy)

                    del frame, results, detections, target
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

            del frame
            del results
            del detections
            del target

            frame_counter += 1
            if frame_counter % 150 == 0:
                gc.collect()
                frame_counter = 0

    except KeyboardInterrupt:
        print("\n[종료] 사용자가 중단했습니다.")
    finally:
        stop_all_actions()
        if SCREEN_CAPTURE is not None:
            SCREEN_CAPTURE.close()
        
        gc.collect()
        if YOLO_DEVICE != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
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