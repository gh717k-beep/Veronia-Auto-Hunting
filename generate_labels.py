import cv2
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MOB_DIR = ROOT / "mob_screenshots"


def find_mob_bbox(image):
    """
    이미지에서 몹 객체의 바운딩 박스를 찾습니다.
    가장 큰 컨투어를 기반으로 좌표를 계산합니다.
    
    Returns:
        (norm_x_center, norm_y_center, norm_width, norm_height) or None
    """
    if image is None or image.size == 0:
        return None
    
    h, w = image.shape[:2]
    
    # BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 어두운 색상 범위 (몹은 보통 어둡거나 중간 톤)
    lower_color = np.array([0, 0, 20])
    upper_color = np.array([180, 255, 220])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 컨투어를 못 찾으면 이미지 전체를 하나의 박스로 간주
        return (0.5, 0.5, 1.0, 1.0)
    
    # 가장 큰 컨투어 찾기
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # 노이즈 필터링: 너무 작은 객체는 무시
    if area < (w * h * 0.001):
        return (0.5, 0.5, 1.0, 1.0)
    
    x, y, cw, ch = cv2.boundingRect(largest_contour)
    
    # 정규화
    norm_x_center = (x + cw / 2) / w
    norm_y_center = (y + ch / 2) / h
    norm_width = cw / w
    norm_height = ch / h
    
    # 클램핑
    norm_x_center = max(0.0, min(1.0, norm_x_center))
    norm_y_center = max(0.0, min(1.0, norm_y_center))
    norm_width = max(0.01, min(1.0, norm_width))
    norm_height = max(0.01, min(1.0, norm_height))
    
    return (norm_x_center, norm_y_center, norm_width, norm_height)


def generate_labels():
    """
    mob_screenshots의 모든 이미지에 대해 라벨 파일을 생성합니다.
    """
    if not MOB_DIR.exists():
        raise FileNotFoundError(f"{MOB_DIR} 폴더가 없습니다.")
    
    image_files = sorted(MOB_DIR.glob("**/*.png"))
    
    if not image_files:
        raise FileNotFoundError("PNG 이미지가 없습니다.")
    
    label_count = 0
    
    for image_path in image_files:
        # OpenCV와 한글 경로 문제를 해결하기 위해 numpy로 바이너리 읽고 디코딩
        img_bytes = np.fromfile(str(image_path), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"[FAIL] 읽기 실패: {image_path.name}")
            continue
        
        bbox = find_mob_bbox(img)
        if bbox is None:
            print(f"[FAIL] 감지 실패: {image_path.name}")
            continue
        
        label_path = image_path.with_suffix(".txt")
        norm_x, norm_y, norm_w, norm_h = bbox
        
        # YOLO 포맷: <class_id> <x_center> <y_center> <width> <height>
        label_content = f"0 {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
        
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(label_content)
        
        label_count += 1
        print(f"[OK] {image_path.name} -> {label_path.name}")
    
    print(f"\n생성된 라벨 파일: {label_count}개")


if __name__ == "__main__":
    generate_labels()
