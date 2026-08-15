import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageGrab
from pynput import keyboard, mouse


BASE_DIR = os.path.join(os.getcwd(), "mob_screenshots")
os.makedirs(BASE_DIR, exist_ok=True)
CAPTURE_COOLDOWN_SEC = 0.25


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣_\- ]", "", name).strip()
    return cleaned or "unnamed_mob"


def ensure_mob_dir(mob_name: str) -> str:
    safe_name = sanitize_name(mob_name)
    mob_dir = os.path.join(BASE_DIR, safe_name)
    os.makedirs(mob_dir, exist_ok=True)
    return mob_dir


def capture_screen():
    img = np.array(ImageGrab.grab())
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_total_saved_count(mob_dir: str) -> int:
    if not os.path.isdir(mob_dir):
        return 0
    return sum(1 for file in os.listdir(mob_dir) if file.lower().endswith(".png"))


def save_capture(mob_dir: str, mob_name: str) -> str:
    frame = capture_screen()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"{sanitize_name(mob_name)}_{timestamp}.png"
    path = os.path.join(mob_dir, file_name)

    try:
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, format="PNG")
    except Exception as exc:
        raise IOError(f"이미지 저장 실패: {path} ({exc})") from exc

    total = get_total_saved_count(mob_dir)
    print(f"저장: {file_name} | 총 {total}장")
    return path


def main():
    print("몬스터 스크린샷 수집 모드")
    print("처음 실행 시 몬스터 이름을 입력하세요.")
    mob_name = input("몬스터 이름: ").strip()
    while not mob_name:
        print("몬스터 이름을 입력해야 합니다.")
        mob_name = input("몬스터 이름: ").strip()

    mob_dir = ensure_mob_dir(mob_name)
    print(f"저장 폴더: {mob_dir}")
    print("마우스 휠 클릭: 캡처, F9: 종료")

    stop = False
    last_capture_time = 0.0

    def on_click(x, y, button, pressed):
        nonlocal last_capture_time
        if not pressed:
            return
        if button != mouse.Button.middle:
            return

        now = time.perf_counter()
        if now - last_capture_time < CAPTURE_COOLDOWN_SEC:
            return
        last_capture_time = now

        try:
            save_capture(mob_dir, mob_name)
        except Exception as exc:
            print(f"저장 실패: {exc}")

    def on_press(key):
        nonlocal stop
        try:
            if key == keyboard.Key.f9:
                stop = True
                print("수집 종료")
                return False
        except AttributeError:
            pass

    with mouse.Listener(on_click=on_click) as mouse_listener, keyboard.Listener(on_press=on_press) as key_listener:
        while not stop:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
