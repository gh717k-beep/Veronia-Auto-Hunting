import os
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MOB_DIR = ROOT / "mob_screenshots"
DATASET_DIR = ROOT / "yolo_dataset"
IMG_TRAIN = DATASET_DIR / "images" / "train"
IMG_VAL = DATASET_DIR / "images" / "val"
LBL_TRAIN = DATASET_DIR / "labels" / "train"
LBL_VAL = DATASET_DIR / "labels" / "val"

for folder in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    folder.mkdir(parents=True, exist_ok=True)


def build_dataset():
    if not MOB_DIR.exists():
        raise FileNotFoundError(f"{MOB_DIR} 폴더가 없습니다. 먼저 몹 사진을 수집하세요.")

    image_files = []
    for mob_folder in sorted(MOB_DIR.iterdir()):
        if not mob_folder.is_dir():
            continue
        for image_file in mob_folder.glob("*.png"):
            image_files.append(image_file)

    if not image_files:
        raise FileNotFoundError("수집된 PNG 이미지가 없습니다. 스크린샷을 먼저 찍어주세요.")

    random.shuffle(image_files)
    val_count = max(1, min(20, len(image_files) // 5))
    train_files = image_files[val_count:]
    val_files = image_files[:val_count]

    for src in train_files:
        dst = IMG_TRAIN / src.name
        try:
            shutil.copy2(src, dst)
            label_path = src.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, LBL_TRAIN / src.with_suffix(".txt").name)
        except Exception as e:
            print(f"[ERROR] {src.name}: {e}")

    for src in val_files:
        dst = IMG_VAL / src.name
        try:
            shutil.copy2(src, dst)
            label_path = src.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, LBL_VAL / src.with_suffix(".txt").name)
        except Exception as e:
            print(f"[ERROR] {src.name}: {e}")

    print(f"총 이미지 수: {len(image_files)}")
    print(f"학습용: {len(train_files)}")
    print(f"검증용: {len(val_files)}")
    print(f"데이터셋 위치: {DATASET_DIR}")


if __name__ == "__main__":
    build_dataset()
