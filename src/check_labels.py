import os, random, glob, cv2, numpy as np
from pathlib import Path

# Configs
BASE_DIR     = Path(__file__).resolve().parent.parent
MIX_DIR      = BASE_DIR / "mix_data_ver2"
SOURE_DIR    = MIX_DIR / "images"      # Nguồn ảnh mix
LABELS_DIR   = MIX_DIR / "labels"      # nguồn labels
N_SAMPLES    = 20                       # số ảnh random cần check
DISPLAY_SIZE = (640, 640)
CLASS_COLORS = {
    "0": (255, 0, 0),                  # đỏ
    "1": (0, 255, 0),                  # xanh lá
    "2": (0, 0, 255),                  # xanh dương
    "3": (255, 255, 0),                # vàng
    "4": (255, 0, 255),                # hồng tím
    "5": (0, 255, 255),                # cyan
    "6": (128, 0, 255),                # tím nhạt
    "7": (255, 128, 0),                # cam
    "8": (128, 128, 255),              # xanh lam nhạt
    "9": (0, 128, 255),                # xanh biển
    "10": (0, 255, 128),               # xanh ngọc
    "11": (200, 200, 200),             # xám nhạt
    "12": (50, 50, 50),                # xám đậm
}

MODE = -1
"""
    -1: SHOW TẤT CẢ
    0: be_khong_sl,
    1: bt,
    2: decay, 
    3: hat_loai_2, 
    4: hat_loai_3, 
    5: lbw, 
    6: ow, 
    7: phe, 
    8: sk, 
    9: st, 
    10: tbts, 
    11: vo_cung, 
    12: ww
"""

def get_cls_polygon(line):
    # Chuẩn hoá input
    p = line.strip().split()    # Tách thành list theo khoảng trắng
    if len(p) < 3:
        return None, None
    try:
        cls_id = str(p[0])
    except:
        return None, None
    
    nums = list(map(float, p[1:]))                          # Bỏ cls_id ở đầu
    poly_01 = np.array(nums, dtype=float).reshape(-1, 2)    # Reshape thành N hàng, mỗi hàng x, y
    
    return cls_id, poly_01

def normalize_to_pixel(poly_01, w, h):
    poly = poly_01.copy()
    poly[:, 0] = np.clip(poly[:, 0], 0.0, 1.0) * w
    poly[:, 1] = np.clip(poly[:, 1], 0.0, 1.0) * h
    
    return poly.astype(int)

def overlay_polygon(img, polygon_pixel, color=(0, 255, 255), alpha=1, thickness=2):
    overlay = img.copy()
    # Vẽ viền polygon: reshape về format OpenCV, True: đảm bảo khép kín
    cv2.polylines(overlay, [polygon_pixel.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)
    
    mask = np.zeros(img.shape[:2], np.uint8)    # shape[:2]: lấy height, weight
    # Fill polygon bên trong
    cv2.fillPoly(mask, [polygon_pixel.reshape(-1,1,2)], 255)

    # Trộn masks + polygon: 1.0*overlay + alpha*mask + 0
    overlay = cv2.addWeighted(overlay, 1.0, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), alpha, 0)
    return overlay

def main():
    
    img_paths = []
    for ext in ("*.jpg", "*.png"):
        img_paths += list(Path(SOURE_DIR).glob(ext))

    if not img_paths:
        print(f"Không tìm thấy ảnh trong: {SOURE_DIR}")
        raise SystemExit # Thoát chương trình

    # Random ảnh để test
    samples = random.sample(img_paths, min(N_SAMPLES, len(img_paths)))

    for index, img_path in enumerate(samples, 1):    # Bắt đầu với index = 1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        stem = Path(img_path).stem
        txt_path = os.path.join(LABELS_DIR, stem + ".txt")
        image = img.copy()

        flags = []  # List chứa cờ lỗi 
        if not os.path.exists(txt_path):
            flags.append("NO_LABEL")
        else:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().strip().splitlines() if ln.strip()]
            if not lines:
                flags.append("EMPTY")
            else:
                for index_line, line in enumerate(lines):
                    cls_id, poly_01 = get_cls_polygon(line)
                    if poly_01 is None or poly_01.shape[0] < 3:
                        flags.append(f"ERROR_LINE_{index_line}")
                        continue
                    
                    if MODE != -1 and int(cls_id) != MODE:
                        continue    

                    polygon_pixel = normalize_to_pixel(poly_01, w, h)
                    color_poly = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    color_text = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    image = overlay_polygon(image, polygon_pixel, color=color_poly, alpha=0.15, thickness=2)
                    
                    # Duyệt từng hàng
                    x, y = polygon_pixel[0]
                    title = f"[{index}/{len(samples)}] {stem}"
                    cv2.putText(image, f"{cls_id}", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2, cv2.LINE_AA)
                    
        image = cv2.resize(image, DISPLAY_SIZE)
        cv2.imshow("QC window", image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("QC window")

        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()