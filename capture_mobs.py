import time
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab
from pynput import keyboard, mouse

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = BASE_DIR / "mob_screenshots"

monster_name = "unknown_mob"
ready = False
stop_requested = False


def sanitize_name(name: str) -> str:
    cleaned = name.strip().replace("\\", "_").replace("/", "_")
    cleaned = cleaned.replace("|", "_").replace(":", "_").replace("*", "_")
    cleaned = cleaned.replace("?", "_").replace('"', "_").replace("<", "_")
    cleaned = cleaned.replace(">", "_")
    cleaned = cleaned.strip(" .")
    return cleaned or "unknown_mob"


def save_screenshot(target_name: str) -> None:
    folder = SCREENSHOT_DIR / sanitize_name(target_name)
    folder.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = folder / f"{now}.png"

    screenshot = ImageGrab.grab()
    screenshot.save(file_path)
    print(f"[캡처 완료] {file_path}")


def list_existing_monsters() -> list[str]:
    if not SCREENSHOT_DIR.exists():
        print("[기존 몬스터 목록] 아직 저장된 몬스터 폴더가 없습니다.")
        return []

    monster_folders = sorted(
        [p.name for p in SCREENSHOT_DIR.iterdir() if p.is_dir()],
        key=lambda name: name.lower(),
    )

    if not monster_folders:
        print("[기존 몬스터 목록] 아직 저장된 몬스터 폴더가 없습니다.")
        return []

    print("[기존 몬스터 목록]")
    for index, monster in enumerate(monster_folders, start=1):
        png_count = len(list((SCREENSHOT_DIR / monster).glob("*.png")))
        print(f"  {index}. {monster} ({png_count}장)")

    return monster_folders


def on_key_press(key):
    global ready, stop_requested

    try:
        if key == keyboard.Key.f8:
            ready = not ready
            print(f"[준비 상태] {'활성화' if ready else '비활성화'}")
            return

        if key == keyboard.Key.f9:
            stop_requested = True
            print("[종료] F9 눌림. 프로그램을 정지합니다.")
            return False
    except AttributeError:
        pass


def on_mouse_click(x, y, button, pressed):
    global ready

    if not ready:
        return

    if button == mouse.Button.middle and pressed:
        save_screenshot(monster_name)


def main() -> None:
    global monster_name

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("몹 스크린샷 캡처기")
    print("========================================")
    print("- 몬스터 이름 입력 후 엔터")
    print("- F8: 준비 상태 ON/OFF")
    print("- 마우스 휠 버튼: 현재 화면 캡처")
    print("- F9: 종료")
    print("========================================")

    list_existing_monsters()
    print("========================================")

    monster_name = input("몬스터 이름을 입력하세요: ").strip() or "unknown_mob"
    monster_name = sanitize_name(monster_name)
    print(f"[대상] {monster_name}")
    print(f"[저장 경로] {SCREENSHOT_DIR / monster_name}")
    print("F8 키를 눌러 준비 상태로 들어간 뒤, 마우스 휠 버튼을 눌러 캡처하세요.")

    keyboard_listener = keyboard.Listener(on_press=on_key_press)
    mouse_listener = mouse.Listener(on_click=on_mouse_click)

    keyboard_listener.start()
    mouse_listener.start()

    while not stop_requested:
        time.sleep(0.1)

    keyboard_listener.stop()
    mouse_listener.stop()
    keyboard_listener.join()
    mouse_listener.join()
    print("프로그램 종료")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[강제 종료] 사용자가 Ctrl+C를 눌렀습니다.")
    except Exception as exc:
        print(f"[오류] {exc}")
        input("엔터를 눌러 종료하세요...")
