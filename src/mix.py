import cv2
import numpy as np
import random
from pathlib import Path

N_MIX_IMG      = 5      # số lượng ảnh mix
IOU            = 0.15

# các thông số augmentation nếu muốn dùng
RESIZE         = False  
RANDOM_FLIP    = True
RANDOM_ROTATE  = True

BASE_DIR       = Path(__file__).resolve().parent.parent
BACKGROUND_DIR = BASE_DIR / "background"        # Nguồn ảnh background
OUTPUT_DIR     = BASE_DIR / "mix_data_ver4"
OBJECTS_DIR    = BASE_DIR / "pred_labels_ver4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "be_khong_sl": 0,
    "bt": 1,
    "decay": 2,
    "hat_loai_2": 3,
    "hat_loai_3": 4,
    "lbw": 5,
    "ow": 6,
    "phe": 7,
    "sk": 8,
    "st": 9,
    "tbts": 10,
    "vo_cung": 11,
    "ww": 12,
}

# Tải background 
background_path = BACKGROUND_DIR / "background.png"     # Chọn background
if not background_path.exists():
    raise FileNotFoundError(f"Không tìm thấy ảnh background")

def select_roi(background_path, display_size=(640,640)):

    roi_points = []

    bg_for_roi = cv2.imread(str(background_path))
    if bg_for_roi is None:
        raise FileNotFoundError(f"Không thể load background: {background_path}")

    orig_h, orig_w = bg_for_roi.shape[:2]
    bg_display = cv2.resize(bg_for_roi, display_size)

    scale_x = orig_w / display_size[0]
    scale_y = orig_h / display_size[1]

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            roi_points.append((orig_x, orig_y))
            print(f"Chọn điểm: {(orig_x, orig_y)}")

    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", mouse_callback)

    while True:
        temp = bg_display.copy()
        for p in roi_points:
            disp_x = int(p[0] / scale_x)
            disp_y = int(p[1] / scale_y)
            cv2.circle(temp, (disp_x, disp_y), 5, (0,0,255), -1)
        if len(roi_points) >= 4:
            pts_disp = np.array(
                [(int(px/scale_x), int(py/scale_y)) for (px,py) in roi_points],
                np.int32
            ).reshape((-1,1,2))
            cv2.polylines(temp, [pts_disp], True, (0,255,0), 2)
        cv2.imshow("ROI", temp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:   # Enter
            break
        elif key == 27: # ESC reset
            roi_points = []

    cv2.destroyAllWindows()

    roi_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if len(roi_points) >= 4:
        cv2.fillPoly(roi_mask, [np.array(roi_points, np.int32)], 255)

    return roi_mask, orig_h, orig_w

def paste_object(background, obj, position):
    x, y = position

    object_h, object_w = obj.shape[:2]
    background_h, background_w = background.shape[:2]

    # Return background nếu ngoài biên
    if x + object_w > background_w or y + object_h > background_h:
        return background

    # Kênh alpha
    alpha = obj[:,:,3] / 255.0

    alpha_3 = alpha[..., None]
    background[y:y+object_h, x:x+object_w, :3] = (
        (1 - alpha_3) * background[y:y+object_h, x:x+object_w, :3] + alpha_3 * obj[..., :3]
    )
    
    mask = (alpha > 0.5).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []

    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        cnt[:, 0] += x
        cnt[:, 1] += y
        polys.append(cnt)
    return background, polys

def random_transform(obj):
    h, w = obj.shape[:2]
    size = (w, h)

    if RANDOM_FLIP and random.random() < 30:
        # Random flip
        obj = cv2.flip(obj, 1)

    if RANDOM_ROTATE:
        # Random rotate
        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), angle, 1.0)
        obj = cv2.warpAffine(obj, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return obj

def mask_iou(maskA, maskB):
    # Tính IoU giữa 2 mask nhị phân 
    maskA = maskA > 0
    maskB = maskB > 0
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    if union == 0:
        return 0.0
    return intersection / union

def label_obj(txt_path, polygons, w, h):
    lines = []
    for poly, class_id in polygons:
        normalize = []
        for pixel_x, pixel_y in poly:
            normalize_x = float(pixel_x) / float(w)
            normalize_y = float(pixel_y) / float(h)
            normalize.append(f"{normalize_x:.6f} {normalize_y:.6f}")
        line = f"{class_id} " + " ".join(normalize)
        lines.append(line)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def mix_images(background_path, object_imgs, output_dir, roi_mask, n_mix=N_MIX_IMG):
    start_idx = len(list((output_dir / "images").glob("mixed_*.png")))
    for idx in range(start_idx, start_idx + n_mix):
        background = cv2.imread(str(background_path))
        h, w = background.shape[:2]
        polygons_all = []
        mask_list = []


        nums_obj = random.randint(50, 60)
        for _ in range(nums_obj):
            obj_path = random.choice(object_imgs)
            obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
            if obj is None or obj.shape[2] != 4:
                continue

            obj = random_transform(obj)

            for _ in range(100):
                x = random.randint(0, max(1, w - obj.shape[1]))
                y = random.randint(0, max(1, h - obj.shape[0]))
                cx, cy = x + obj.shape[1]//2, y + obj.shape[0]//2

                if roi_mask[cy, cx] == 255:     # Kiểm tra có trong ROI không

                    # tạo mask tạm của object
                    alpha = (obj[:,:,3] > 127).astype(np.uint8)
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    temp_mask[y:y+alpha.shape[0], x:x+alpha.shape[1]] = alpha * 255

                    overlap = False
                    for prev_mask in mask_list:
                        if mask_iou(temp_mask, prev_mask) > IOU:
                            overlap = True
                            break
                    if overlap:
                        continue

                    background, polys = paste_object(background, obj, (x,y))

                    if polys:
                        # Xác định thư mục class hiện tại
                        class_name = obj_path.parents[2].name
                        class_id = CLASS_MAP.get(class_name, 0)

                        for poly in polys:
                            poly = np.array(poly).reshape(-1, 2)
                            polygons_all.append((poly, class_id))
                        mask_list.append(temp_mask)
                    break

        out_img_path   = output_dir / "images" / f"mixed_{idx}.png"
        out_label_path = output_dir / "labels" / f"mixed_{idx}.txt"

        cv2.imwrite(str(out_img_path), background)
        label_obj(out_label_path, polygons_all, w, h)

        print(f"\n[Epoch {idx+1:04d}/{n_mix}] " + "="*30)
        print(f" === Saved: mixed_{idx}.png")
        print("   " + "-"*45)

    print("Mix xong, check ở folder:", output_dir)

if __name__ == "__main__":
    # B1: chọn ROI
    roi_mask, orig_h, orig_w = select_roi(background_path)
    
    # B2: mix ảnh
    object_imgs = []
    for cls_dir in OBJECTS_DIR.iterdir():
        object_dir = cls_dir/"object"
        if object_dir.exists():
            object_imgs += list(object_dir.rglob("*.png"))

    mix_images(background_path, object_imgs, OUTPUT_DIR, roi_mask, n_mix=N_MIX_IMG)