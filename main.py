import random
import re
import shutil
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
    if pytesseract is not None:
        candidate_paths = [
            shutil.which('tesseract'),
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for candidate in candidate_paths:
            if candidate and Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                break
        if not pytesseract.pytesseract.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    pytesseract = None

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.001

# 성능 최적화
FRAME_SCALE = 1.0
MAX_PROCESS_FPS = 1200
MIN_INTERVAL_SEC = 1.0 / MAX_PROCESS_FPS
MOVE_SMOOTHING = 4.2
CLICK_RATE_PER_SEC = 40.0
CLICK_INTERVAL_SEC = 1.0 / CLICK_RATE_PER_SEC
FAR_ATTACK_DISTANCE_BOX_H = 170
NEAR_ATTACK_DISTANCE_BOX_H = 260
AIM_DEADZONE = 10
MAX_MOVE_PER_FRAME = None  # 제한 없음, 방향으로 직접 이동
MAX_Y_OFFSET_FROM_START = 200  # 프로그램 시작 시 Y축 이동 가능 마지노선
SCAN_RIGHT_STEP = 18
SCAN_DOWN_STEP = 18
SCAN_UP_STEP = 18
W_KEY_ACTIVE = False
R_KEY_ACTIVE = False
SEARCH_BOOST_ACTIVE = False
SEARCH_BOOST_UNTIL = 0.0

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
    confidence_threshold = 0.7

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
HUD_CHECK_TRIGGER = False
OCR_WARNED = False
OCR_SCAN_ACTIVE = False


def ensure_ocr_ready():
    global OCR_WARNED
    if pytesseract is None:
        if not OCR_WARNED:
            print("[HUD OCR] pytesseract가 설치되지 않았습니다. pip install pytesseract를 실행해 주세요.")
            OCR_WARNED = True
        return False

    tess_path = pytesseract.pytesseract.tesseract_cmd or shutil.which('tesseract') or r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not Path(tess_path).exists():
        if not OCR_WARNED:
            print("[HUD OCR] Tesseract 엔진이 설치되지 않았습니다. Windows에서 Tesseract OCR를 설치해 주세요.")
            OCR_WARNED = True
        return False

    return True


def set_running(value):
    global RUNNING
    with RUN_LOCK:
        RUNNING = value


def get_running():
    with RUN_LOCK:
        return RUNNING


def on_press(key):
    global HUD_CHECK_TRIGGER
    try:
        if key == keyboard.Key.f8:
            set_running(True)
            print("[F8] 자동 사냥 시작")
            HUD_CHECK_TRIGGER = True
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
    results = model(frame, conf=0.25, iou=0.35, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            conf = box.conf[0].cpu().item()

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 8 or box_h < 8:
                continue
            if box_w > w_scaled * 0.9 or box_h > h_scaled * 0.9:
                continue

            aspect_ratio = max(box_w, box_h) / max(min(box_w, box_h), 1)
            if aspect_ratio > 5:
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
        move_x = int(dx * 2.2)
        move_y = int(dy * 2.2)
    else:
        move_x = int(dx * MOVE_SMOOTHING * 0.9)
        move_y = int(dy * MOVE_SMOOTHING * 0.9)

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


def capture_debug_hud_overlay():
    """좌측 상단 F3 디버그 HUD 텍스트를 더 넓게 캡처합니다."""
    width, height = pyautogui.size()
    left = 0
    top = 0
    right = min(420, width)
    bottom = min(420, height)
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


def read_debug_hud_text_from_screen():
    """좌측 상단 HUD 텍스트를 OCR로 읽고 파싱합니다."""
    if not ensure_ocr_ready():
        return None

    try:
        image = capture_debug_hud_overlay()
        if image is None:
            return None

        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(gray, config='--psm 6')
        return text.strip()
    except Exception:
        return None


def test_ocr_with_sample_text():
    """Tesseract가 실제로 동작하는지 샘플 텍스트로 검증한다."""
    if not ensure_ocr_ready():
        print("[HUD OCR] Tesseract 확인 실패")
        return False

    try:
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new('L', (400, 120), 255)
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), 'X: 123.4 Z: 456.7 Pitch: 10.2', fill=0)
        text = pytesseract.image_to_string(image, config='--psm 6')
        print(f"[HUD OCR TEST] {text.strip()}")
        return 'X:' in text or 'Z:' in text or 'Pitch' in text
    except Exception as exc:
        print(f"[HUD OCR TEST] 실패: {exc}")
        return False


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


def parse_debug_hud_text(text):
    """실제 게임 HUD 포맷에 맞춰 X/Y/Z, Pitch, Yaw, Facing 값을 각각 추출한다."""
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {}

    for line in lines:
        x_match = re.search(r'X\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if x_match:
            result['x'] = float(x_match.group(1))
            continue

        y_match = re.search(r'Y\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if y_match:
            result['y'] = float(y_match.group(1))
            continue

        z_match = re.search(r'Z\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if z_match:
            result['z'] = float(z_match.group(1))
            continue

        pitch_match = re.search(r'Pitch\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if pitch_match:
            result['pitch'] = float(pitch_match.group(1))
            continue

        yaw_match = re.search(r'Yaw\s*[:=]\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if yaw_match:
            result['yaw'] = float(yaw_match.group(1))
            continue

        xyz_match = re.search(r'XYZ\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*[/\\ ]+\s*([-+]?\d+(?:\.\d+)?)\s*[/\\ ]+\s*([-+]?\d+(?:\.\d+)?)', line, re.IGNORECASE)
        if xyz_match:
            result['xyz'] = tuple(float(v) for v in xyz_match.groups())
            continue

        facing_match = re.search(r'Facing\s*[:=]\s*(north|south|east|west|northeast|northwest|southeast|southwest)', line, re.IGNORECASE)
        if facing_match:
            result['facing'] = facing_match.group(1).lower()
            continue

    if not result:
        return None

    return result


def get_pitch_sign_from_hud_text(hud_text):
    if not hud_text:
        return None

    hud_info = parse_debug_hud_text(hud_text)
    if not hud_info:
        return None

    pitch = hud_info.get('pitch')
    if pitch is None:
        xyz = hud_info.get('xyz')
        if xyz is not None and len(xyz) >= 3:
            pitch = xyz[1]
    if pitch is None:
        return None

    return 'negative' if pitch < 0 else 'positive'


def apply_pitch_based_mouse_correction():
    """OCR 스캔 중일 때 pitch 상태를 기준으로 마우스를 위/아래로 보정한다."""
    if not OCR_SCAN_ACTIVE or not get_running():
        return

    hud_text = read_debug_hud_text_from_screen()
    pitch_sign = get_pitch_sign_from_hud_text(hud_text)
    if pitch_sign is None:
        pydirectinput.moveRel(150, 0, relative=True)
        return

    if pitch_sign == 'positive':
        for _ in range(6):
            if not OCR_SCAN_ACTIVE or not get_running():
                return
            next_text = read_debug_hud_text_from_screen()
            next_sign = get_pitch_sign_from_hud_text(next_text)
            if next_sign == 'negative':
                break
            pydirectinput.moveRel(150, -12, relative=True)
            time.sleep(0.04)
    elif pitch_sign == 'negative':
        for _ in range(6):
            if not OCR_SCAN_ACTIVE or not get_running():
                return
            next_text = read_debug_hud_text_from_screen()
            next_sign = get_pitch_sign_from_hud_text(next_text)
            if next_sign == 'positive':
                break
            pydirectinput.moveRel(150, 12, relative=True)
            time.sleep(0.04)


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


def set_r_search(active):
    """무감지 탐색 시 R 키를 누른 상태로 유지한다."""
    global R_KEY_ACTIVE

    if active and not R_KEY_ACTIVE:
        pydirectinput.keyDown('r')
        R_KEY_ACTIVE = True
        print("R 탐색 시작: 주변을 랜덤하게 둘러봅니다")
    elif not active and R_KEY_ACTIVE:
        pydirectinput.keyUp('r')
        R_KEY_ACTIVE = False
        print("R 탐색 종료")


def set_search_boost(active):
    """객체가 R 탐색 중 발견되면 Ctrl + Space + W를 3초 유지한다."""
    global SEARCH_BOOST_ACTIVE

    if active and not SEARCH_BOOST_ACTIVE:
        if W_KEY_ACTIVE:
            set_movement_combo(False)
        pydirectinput.keyDown('ctrl')
        pydirectinput.keyDown('space')
        pydirectinput.keyDown('w')
        SEARCH_BOOST_ACTIVE = True
        print("객체 발견 보정: Ctrl + Space + W 1.5초 유지")
    elif not active and SEARCH_BOOST_ACTIVE:
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('space')
        pydirectinput.keyUp('ctrl')
        SEARCH_BOOST_ACTIVE = False


def press_w_key_hold(duration_sec=0.08):
    """짧게 이동 조합 키를 눌러 게임이 실제 입력을 인식하도록 보장한다."""
    pass
    set_movement_combo(True)
    time.sleep(duration_sec)
    set_movement_combo(False)


def send_attack_clicks():
    """감지 갱신마다 빠르게 좌클릭한다."""
    pydirectinput.mouseDown(button='left')
    time.sleep(0.02)
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
                move_mouse_relative_to_center(x_center, y_center, fast=True)

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
                    move_mouse_relative_to_center(x_center, y_center, fast=True)

            time.sleep(0.002)
    finally:
        if W_KEY_ACTIVE:
            set_movement_combo(False)


def main():
    global W_KEY_ACTIVE, HUD_CHECK_TRIGGER, OCR_SCAN_ACTIVE, SEARCH_BOOST_UNTIL
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
    print("모델 로드 완료")
    print(f"HUD OCR 준비 상태: {ensure_ocr_ready()}")
    if pytesseract is not None and hasattr(pytesseract, 'pytesseract'):
        print(f"Tesseract 경로: {pytesseract.pytesseract.tesseract_cmd}")
    print("대기 중... F8을 눌러 시작하세요")

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    last_process_time = 0
    last_click_time = 0.0
    last_detection_time = time.monotonic()
    last_hud_log_time = 0.0
    hud_force_log_next = False
    start_mouse_x, start_mouse_y = pyautogui.position()
    max_y_lower_bound = start_mouse_y - MAX_Y_OFFSET_FROM_START
    scan_mode = "top_right"

    try:
        while True:
            if not get_running():
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            if now - last_process_time < 0.003:
                time.sleep(0.0015)
                continue

            if SEARCH_BOOST_ACTIVE and time.monotonic() >= SEARCH_BOOST_UNTIL:
                set_search_boost(False)

            if OCR_SCAN_ACTIVE and get_running():
                apply_pitch_based_mouse_correction()
                last_hud_log_time = time.monotonic()
                hud_force_log_next = False
                HUD_CHECK_TRIGGER = False

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

                OCR_SCAN_ACTIVE = False
                last_detection_time = time.monotonic()
                mob = get_closest_mob(filtered)
                x_center, y_center, conf, box_h = mob

                if R_KEY_ACTIVE:
                    move_mouse_relative_to_center(x_center, y_center, fast=True)
                    set_r_search(False)
                    SEARCH_BOOST_UNTIL = time.monotonic() + 1.5
                    set_search_boost(True)

                move_mouse_relative_to_center(x_center, y_center, fast=True)
                OCR_SCAN_ACTIVE = False

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
                if SEARCH_BOOST_ACTIVE and time.monotonic() >= SEARCH_BOOST_UNTIL:
                    set_search_boost(False)
                if R_KEY_ACTIVE:
                    set_r_search(False)

                now_no_detection = time.monotonic()
                if now_no_detection - last_detection_time >= 5.0:
                    if not OCR_SCAN_ACTIVE:
                        print("무감지 5초 경과: OCR 스캔 + 마우스 보정 시작")
                    OCR_SCAN_ACTIVE = True

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
