import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pydirectinput
from PIL import ImageGrab
from pynput import keyboard
from ultralytics import YOLO

try:
    import pytesseract
except Exception:
    pytesseract = None

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.001

# 성능 최적화
FRAME_SCALE = 0.8
MAX_PROCESS_FPS = 80
MIN_INTERVAL_SEC = 1.0 / MAX_PROCESS_FPS
MOVE_SMOOTHING = 1.2
CLICK_RATE_PER_SEC = 8.0
CLICK_INTERVAL_SEC = 1.0 / CLICK_RATE_PER_SEC
FAR_ATTACK_DISTANCE_BOX_H = 170
NEAR_ATTACK_DISTANCE_BOX_H = 260
AIM_DEADZONE = 50
MAX_MOVE_PER_FRAME = None  # 제한 없음, 방향으로 직접 이동
MAX_Y_OFFSET_FROM_START = 200  # 프로그램 시작 시 Y축 이동 가능 마지노선
SCAN_RIGHT_STEP = 18
SCAN_DOWN_STEP = 18
SCAN_UP_STEP = 18
W_KEY_ACTIVE = False

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
LEGACY_RUNS_DIR = ROOT / "runs" / "detect"


def resolve_model_path(model_name=None):
    if model_name:
        candidate_paths = [
            RUNS_DIR / model_name / "weights" / "best.pt",
            LEGACY_RUNS_DIR / model_name / "weights" / "best.pt",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"모델 '{model_name}' 을(를) 찾지 못했습니다. 후보 경로: {candidate_paths[0]} / {candidate_paths[1]}")

    candidates = []
    for base_dir in [RUNS_DIR, LEGACY_RUNS_DIR]:
        if base_dir.exists():
            candidates.extend(sorted(base_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))

    if candidates:
        return candidates[0]

    fallback = ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("사용 가능한 모델을 찾지 못했습니다. 먼저 학습을 진행하거나 모델 폴더를 확인하세요.")


def get_model_names():
    names = []
    for base_dir in [RUNS_DIR, LEGACY_RUNS_DIR]:
        if base_dir.exists():
            names.extend(sorted({p.parent.parent.name for p in base_dir.glob("**/weights/best.pt")}))
    return sorted(set(names))


def parse_cli_args():
    args = __import__('sys').argv[1:]
    selected_model = None
    confidence_threshold = 0.55

    if not args:
        return selected_model, confidence_threshold

    first = args[0]
    if "." in first and first.replace('.', '', 1).isdigit():
        confidence_threshold = float(first)
        if len(args) >= 2:
            selected_model = args[1]
        return selected_model, confidence_threshold

    selected_model = first
    if len(args) >= 2 and args[1].replace('.', '', 1).isdigit():
        confidence_threshold = float(args[1])

    if selected_model.isdigit():
        model_names = get_model_names()
        index = int(selected_model)
        if 1 <= index <= len(model_names):
            selected_model = model_names[index - 1]

    return selected_model, confidence_threshold


RUNNING = False
RUN_LOCK = threading.Lock()


def set_running(value):
    global RUNNING
    with RUN_LOCK:
        RUNNING = value


def get_running():
    with RUN_LOCK:
        return RUNNING


def on_press(key):
    try:
        if key == keyboard.Key.f8:
            set_running(True)
            print("[F8] 자동 사냥 시작")
        elif key == keyboard.Key.f9:
            set_running(False)
            print("[F9] 자동 사냥 정지")
    except AttributeError:
        pass


def keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def capture_screen():
    """화면을 캡처하고 축소합니다."""
    image = np.array(ImageGrab.grab())
    if FRAME_SCALE != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(
            image,
            (max(1, int(w * FRAME_SCALE)), max(1, int(h * FRAME_SCALE))),
            interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def detect_mobs(frame, model):
    """YOLO 모델로 몹을 감지하고, 거리 판단용 박스 높이도 같이 반환합니다."""
    h_scaled, w_scaled = frame.shape[:2]
    results = model(frame, conf=0.35, iou=0.45, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            conf = box.conf[0].cpu().item()

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 20 or box_h < 20:
                continue
            if box_w > w_scaled * 0.7 or box_h > h_scaled * 0.7:
                continue

            aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)
            if aspect_ratio > 3:
                continue

            x_center_orig = x_center / FRAME_SCALE
            y_center_orig = y_center / FRAME_SCALE
            detections.append((x_center_orig, y_center_orig, conf, box_h))

    return detections


def get_closest_mob(detections):
    if not detections:
        return None

    screen_width, screen_height = pyautogui.size()
    screen_center_x = screen_width / 2
    screen_center_y = screen_height / 2

    closest = min(
        detections,
        key=lambda d: (d[0] - screen_center_x) ** 2 + (d[1] - screen_center_y) ** 2
    )
    return closest


def move_mouse_relative_to_center(target_x, target_y, fast=False):
    """타깃 방향으로 이동. 감지 시 fast=True면 빠르게, 아니면 가볍게."""
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width / 2
    center_y = screen_height / 2

    dx = target_x - center_x
    dy = target_y - center_y

    if abs(dx) < AIM_DEADZONE and abs(dy) < AIM_DEADZONE:
        return 0, 0

    if fast:
        move_x = int(dx * 0.9)
        move_y = int(dy * 0.9)
    else:
        move_x = int(dx * MOVE_SMOOTHING * 0.35)
        move_y = int(dy * MOVE_SMOOTHING * 0.35)

    if abs(move_x) > 0 or abs(move_y) > 0:
        pydirectinput.moveRel(move_x, move_y, relative=True)

    return move_x, move_y


def capture_hp_overlay():
    """상단 HP 텍스트 영역을 캡처합니다."""
    width, height = pyautogui.size()
    left = int(width * 0.25)
    top = 0
    right = int(width * 0.75)
    bottom = min(int(height * 0.18), 180)
    if right <= left or bottom <= top:
        return None
    return np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))


def read_hp_text_from_screen():
    """상단 HP 텍스트를 OCR로 읽고 파싱합니다."""
    if pytesseract is None:
        return None

    try:
        image = capture_hp_overlay()
        if image is None:
            return None

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text.strip()
    except Exception:
        return None


def parse_hp_text(text):
    """예: Lv.45 활쏘는 인디언 HP 123 / 500"""
    if not text:
        return None

    match = re.search(r'Lv\.?\s*(\d+)\s*([^\n]*?)\s*HP\s*(\d+)\s*/\s*(\d+)', text, re.IGNORECASE)
    if match:
        level, name, current_hp, max_hp = match.groups()
        return {
            'level': int(level),
            'name': name.strip(),
            'current': int(current_hp),
            'max': int(max_hp),
        }

    match = re.search(r'HP\s*(\d+)\s*/\s*(\d+)', text, re.IGNORECASE)
    if match:
        current_hp, max_hp = match.groups()
        return {'level': None, 'name': None, 'current': int(current_hp), 'max': int(max_hp)}

    return None


def ensure_hp_ui_visible():
    """HP UI를 확인해보고 읽을 수 있으면 반환한다."""
    return parse_hp_text(read_hp_text_from_screen())


def set_movement_combo(active):
    """이동 키를 Ctrl + W + Space 조합으로 활성/비활성화한다."""
    global W_KEY_ACTIVE

    if active and not W_KEY_ACTIVE:
        pydirectinput.keyDown('ctrl')
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('space')
        W_KEY_ACTIVE = True
        pass
    elif not active and W_KEY_ACTIVE:
        pydirectinput.keyUp('space')
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('ctrl')
        W_KEY_ACTIVE = False
        pass


def press_w_key_hold(duration_sec=0.08):
    """짧게 이동 조합 키를 눌러 게임이 실제 입력을 인식하도록 보장한다."""
    pass
    set_movement_combo(True)
    time.sleep(duration_sec)
    set_movement_combo(False)


def send_attack_clicks():
    """감지 갱신마다 한 번씩 좌클릭한다."""
    pydirectinput.mouseDown(button='left')
    time.sleep(0.05)
    pydirectinput.mouseUp(button='left')


def attack_target_until_dead(model, confidence_threshold):
    """거리 기반 접근 로직: 멀면 W로 이동, 사거리 안이면 정지 후 좌클릭한다."""
    global W_KEY_ACTIVE

    last_target = None
    last_click_time = 0.0

    try:
        while get_running():
            hp_info = parse_hp_text(read_hp_text_from_screen())
            if hp_info is None:
                hp_info = ensure_hp_ui_visible()

            if hp_info is not None and hp_info.get('current', 1) == 0:
                return

            frame = capture_screen()
            detections = detect_mobs(frame, model)
            filtered = [d for d in detections if d[2] >= confidence_threshold]

            if filtered:
                target = get_closest_mob(filtered)
                last_target = target
                x_center, y_center, conf = target[:3]
                box_h = target[3] if len(target) > 3 else 0
                move_mouse_relative_to_center(x_center, y_center)

                screen_width, screen_height = pyautogui.size()
                center_x = screen_width / 2
                dx = x_center - center_x

                if box_h < 170:
                    if not W_KEY_ACTIVE:
                        set_movement_combo(True)
                        print(f"거리 멀음: box_h={box_h:.0f} -> Ctrl + W + Space 지속 전진")
                elif box_h > NEAR_ATTACK_DISTANCE_BOX_H:
                    if W_KEY_ACTIVE:
                        set_movement_combo(False)
                    if abs(dx) < AIM_DEADZONE:
                        now = time.monotonic()
                        if now - last_click_time >= CLICK_INTERVAL_SEC:
                            send_attack_clicks()
                            last_click_time = now
                else:
                    if W_KEY_ACTIVE:
                        set_movement_combo(False)
            else:
                if W_KEY_ACTIVE:
                    set_movement_combo(False)
                if last_target is not None:
                    x_center, y_center, conf = last_target[:3]
                    move_mouse_relative_to_center(x_center, y_center)

            time.sleep(0.01)
    finally:
        if W_KEY_ACTIVE:
            set_movement_combo(False)


def main():
    global W_KEY_ACTIVE
    print("YOLO 몹 감지 자동 사냥 시작")
    print("F8: 시작 | F9: 정지 | Ctrl + C: 종료")

    selected_model, confidence_threshold = parse_cli_args()

    if confidence_threshold < 0.0:
        confidence_threshold = 0.0
    if confidence_threshold > 1.0:
        confidence_threshold = 1.0

    model_path = resolve_model_path(selected_model) if selected_model else resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    print(f"모델 로드 중: {model_path}")
    print(f"신뢰도 임계치: {confidence_threshold:.2f}")
    model = YOLO(str(model_path))
    print("모델 로드 완료\n대기 중...")

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    last_process_time = 0
    last_click_time = 0.0
    last_detection_time = time.monotonic()
    start_mouse_x, start_mouse_y = pyautogui.position()
    max_y_lower_bound = start_mouse_y - MAX_Y_OFFSET_FROM_START
    scan_mode = "top_right"

    try:
        while True:
            if not get_running():
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            if now - last_process_time < MIN_INTERVAL_SEC:
                time.sleep(0.01)
                continue

            last_process_time = now
            frame = capture_screen()
            detections = detect_mobs(frame, model)

            if detections:
                filtered = [d for d in detections if d[2] >= confidence_threshold]
                if not filtered:
                    if W_KEY_ACTIVE:
                        set_movement_combo(False)
                    time.sleep(0.02)
                    continue

                last_detection_time = time.monotonic()
                mob = get_closest_mob(filtered)
                x_center, y_center, conf, box_h = mob
                move_mouse_relative_to_center(x_center, y_center, fast=True)
                print(f"감지 size={box_h:.0f}")

                # 거리 기반 W 키 제어
                if box_h < 170:
                    if not W_KEY_ACTIVE:
                        set_movement_combo(True)
                        print(f"거리 멀음: box_h={box_h:.0f} -> Ctrl + W + Space 지속 전진")
                else:
                    if W_KEY_ACTIVE:
                        set_movement_combo(False)
                        print(f"거리 적정: box_h={box_h:.0f} -> 이동 정지")

                now_click = time.monotonic()
                if now_click - last_click_time >= CLICK_INTERVAL_SEC:
                    send_attack_clicks()
                    last_click_time = now_click

                hp_info = parse_hp_text(read_hp_text_from_screen())
                if hp_info is None:
                    hp_info = ensure_hp_ui_visible()

                if hp_info is not None and hp_info.get('current', 1) != 0:
                    attack_target_until_dead(model, confidence_threshold)
                    continue

                if hp_info is not None and hp_info.get('current', 1) == 0:
                    set_running(False)
            else:
                if W_KEY_ACTIVE:
                    set_movement_combo(False)

                now_no_detection = time.monotonic()
                if now_no_detection - last_detection_time >= 10.0:
                    if scan_mode == "top_right":
                        scan_mode = "bottom_right"
                    else:
                        scan_mode = "top_right"
                    last_detection_time = now_no_detection
                    print("무감지 10초 경과: 우측 상단/하단 탐색 시작")

                if scan_mode == "top_right":
                    pydirectinput.moveRel(SCAN_RIGHT_STEP, -SCAN_UP_STEP, relative=True)
                else:
                    pydirectinput.moveRel(SCAN_RIGHT_STEP, SCAN_DOWN_STEP, relative=True)

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
