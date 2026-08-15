import time
from pathlib import Path

import cv2
import numpy as np
import pydirectinput
from mss import mss
from ultralytics import YOLO

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.001

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "detect"


def resolve_model_path():
    candidates = sorted(RUNS_DIR.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    root_model = ROOT / "best.pt"
    if root_model.exists():
        return root_model
    raise FileNotFoundError(f"모델 파일을 찾지 못했습니다. {ROOT} 또는 {RUNS_DIR} 를 확인하세요.")


MODEL_PATH = resolve_model_path()
model = YOLO(str(MODEL_PATH))

# 게임 창/모니터 전체 캡처
sct = mss()
monitor = sct.monitors[1]
CAPTURE_AREA = {
    "top": monitor["top"],
    "left": monitor["left"],
    "width": monitor["width"],
    "height": monitor["height"],
    "mon": 1,
}

CENTER_X = CAPTURE_AREA["width"] // 2
CENTER_Y = CAPTURE_AREA["height"] // 2

SMOOTHING = 0.35
ATTACK_DIST = 40
CONFIDENCE = 0.5

print("[INFO] YOLO 인게임 자동사냥 시작")
print(f"[INFO] 모델: {MODEL_PATH}")
print("[INFO] 게임 창을 활성화하고 3초 뒤에 자동 조준이 시작됩니다.")
time.sleep(3)

while True:
    screenshot = sct.grab(CAPTURE_AREA)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

    target_found = False
    closest_dist = float("inf")
    target_dx = 0
    target_dy = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            dx = center_x - CENTER_X
            dy = center_y - CENTER_Y
            dist = np.hypot(dx, dy)

            if dist < closest_dist:
                closest_dist = dist
                target_dx = dx
                target_dy = dy
                target_found = True

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    if target_found:
        move_x = int(target_dx * SMOOTHING)
        move_y = int(target_dy * SMOOTHING)

        if abs(move_x) > 0 or abs(move_y) > 0:
            pydirectinput.moveRel(move_x, move_y, relative=True)

        if closest_dist < ATTACK_DIST:
            pydirectinput.click()
            time.sleep(0.05)

    cv2.imshow("YOLO Auto Hunt", cv2.resize(frame, (960, 540)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
