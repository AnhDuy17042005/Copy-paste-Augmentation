from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configs
BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL          = BASE_DIR / "model" / "hatdieu_seg_ver2.pt"
SAM            = BASE_DIR / "model" / "sam_vit_b_01ec64.pth"
SOURE_DIR      = BASE_DIR / "Data_hatdieu_test"              
OUT_DIR        = BASE_DIR / "pred_labels_ver2"
CONFIDENT      = 0.5

# Paraments smooth viền
BLUR_KERNEL    = 5           # Làm mờ Gauss
CLOSE_KERNEL   = 7           # Morphological Closing : Lấp khe hở, bo tròn   
OPEN_KERNEL    = 3           # Morphological Openning: Giảm nhiễu rời rạc
BLUR           = True           
CLOSE          = True          
OPEN           = True           
FILL_HOLES     = True      
WATERSHED      = True        # Tách Object dính nhau
MIN_AREA_RATIO = 0.001       # Tỷ lệ vật thể tối thiểu để lọc  

# Load model
model = YOLO(MODEL)
sam = sam_model_registry["vit_b"](checkpoint=SAM)
sam.to(device='cpu')
predictor = SamPredictor(sam)

# Tạo file txt
def label_obj(txt_path, instances, width, height):
    lines = [] # Dòng trong labels
    
    for i in instances:
        class_id = int(i["cls"])
        poly     = i["poly"]

        # Normalization
        normalize = []
        for x, y in poly:   
            normalize_x = np.clip((x / width), 0.0, 1.0)    # [0:1]
            normalize_y = np.clip((y / height), 0.0, 1.0)
            normalize += [f"{normalize_x:.6f}", f"{normalize_y:.6f}",]

        line = f"{class_id} " + " ".join(normalize)
        lines.append(line) 
    
    # Ghi lines vào txt
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def morphological(binary_mask):
    mask = binary_mask
    if BLUR:
        mask    = cv2.GaussianBlur(mask, (BLUR_KERNEL, BLUR_KERNEL), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if CLOSE:
        # Tạo cấu trúc kernel với dạng elipse 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if OPEN:
        # Tạo cấu trúc kernel với dạng elipse 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def fill_holes(binary_mask):
    if not FILL_HOLES:
        return binary_mask
    invert = cv2.bitwise_not(binary_mask)
    h, w = binary_mask.shape
    flood = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(invert, flood, (0,0), 255)
    return cv2.bitwise_not(invert)
    
def watershed(binary_mask):
    # Biến đổi distance transform
    distTrans = cv2.distanceTransform((binary_mask>0).astype(np.uint8), cv2.DIST_L2, 3)
    cv2.normalize(distTrans, distTrans, 0, 1.0, cv2.NORM_MINMAX)
    # Tạo seed hạt giống ở tâm object
    seeds = (distTrans > 0.3).astype(np.uint8)
    # Tạo label cho từng object
    # labels: số vùng của seed + background, markers: gán ID mỗi vùng seed
    _   , markers = cv2.connectedComponents(seeds)

    markers = markers.astype(np.int32)  # markers phải dạng int32
    bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(bgr, markers) # 2 mask dính nhau vùng biên = -1

    pieces = []
    for label in range(2, markers.max()+1):  # bỏ 0 = background, 1 = vùng chưa xác định
        piece = (markers == label).astype(np.uint8)*255
        if piece.sum() > 0:
            pieces.append(piece)

    # Trả về maske gốc nếu không tách được vật
    if pieces:
        return pieces
    else:
        return [binary_mask]

def remove_tiny(binary_mask, min_area_px):
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask>0).astype(np.uint8), connectivity=8
    )
    binary_mask = np.zeros_like(binary_mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            binary_mask[labels==i] = 255
    return binary_mask

# drop hạt điều
def extract_object(image_bgr, mask, out_path):
    h, w = mask.shape
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[mask>0] = 255
    b, g, r = cv2.split(image_bgr)
    bgra = cv2.merge([b, g, r, alpha])

    ys, xs = np.where(mask>0)
    if len(xs)==0 or len(ys)==0:
        return
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = bgra[y1:y2+1, x1:x2+1, :]
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)

# Tạo smooth viền
def smooth_binary_mask(binary_mask, w, h):

    # mượt + lấp lỗ trống + bỏ nhiễu nhỏ
    m = morphological(binary_mask)
    m = fill_holes(m)
    m = remove_tiny(m, int(MIN_AREA_RATIO*w*h))
    m = watershed(m)

    polys = []
    for mask in m:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:  
                c = c.reshape(-1, 2)
                polys.append(c)
    return polys

def main():
    # Duyệt qua các class trong data
    cls_folders = [p for p in Path(SOURE_DIR).iterdir() if p.is_dir()]
    if not cls_folders:
        print(f"Không tìm thấy folder trong {SOURE_DIR}")
        return   
   
    for cls_folder in cls_folders:
        cls_name = cls_folder.name
        print(f"Đang xử lý class: {cls_name}")

        # Out dir riêng cho từng class
        predict_dir = Path(OUT_DIR) / cls_name / "predict"
        labels_dir  = Path(OUT_DIR) / cls_name / "labels"
        object_dir  = Path(OUT_DIR) / cls_name / "object"
        for d in [predict_dir, labels_dir, object_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Duyệt qua ảnh trong source
        img_paths = []
        for ext in ("*.jpg", "*.png"):
            img_paths += list(Path(cls_folder).glob(ext))

        if not img_paths:
            print(f"Không tìm thấy ảnh trong: {cls_folder}")

        # Chạy predict
        results = model.predict(
            source=[str(p) for p in img_paths],
            conf=CONFIDENT,
            imgsz=640,
            stream=True,
            verbose=False
        )

        for r in results:
            # Thông tin mỗi ảnh
            img_path = Path(r.path)
            stem     = img_path.stem  # Lấy tên file không lấy đuôi ext

            img_bgr = cv2.imread(str(img_path))
            print(f"\n=== Đang xử lý ảnh: {stem}")
    
            # Vẽ visualization
            visualize = r.plot()    # Vẽ segment predict
            cv2.imwrite(str(predict_dir / f"{stem}.png"), visualize)

            # Giữ kích thước chuẩn của ảnh
            h, w = r.orig_shape

            # Trích instance
            instances = []

            if r.masks is not None and len(r.masks) > 0:
                num = len(r.masks.data)
                
                classes = r.boxes.cls.cpu().numpy() if r.boxes and r.boxes.cls is not None else np.zeros(num)
                # classes = array([0., 1.])      # class cho từng instance
                confs   = r.boxes.conf.cpu().numpy() if r.boxes and r.boxes.conf is not None else np.ones(num)
                # confs   = array([0.91, 0.87])  # confidence cho từng instance

                # Thư mục chứa mask instance png
                object_out_dir = object_dir / stem
                object_out_dir.mkdir(parents=True, exist_ok=True)
                predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

                for i in range(num):

                    # Lấy class id và confident
                    class_id = int(classes[i]) if i < len(classes) else 0
                    conf     = float(confs[i]) if i < len(confs)   else 1.0

                    m = r.masks.data[i].cpu().numpy()
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    m = (m > 0.5).astype(np.uint8) * 255

                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].cpu().numpy())

                    # SAM predict mask
                    masks, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=False
                    )

                    # Lấy mask SAM
                    mask_sam = masks[0].astype(np.uint8) * 255

                    # Cắt mask về kích thước gốc ảnh 
                    mask_sam = cv2.resize(mask_sam, (w, h), interpolation=cv2.INTER_NEAREST)

                    polys = smooth_binary_mask(mask_sam, w, h)

                    # Lưu mask và object PNG
                    for k, poly in enumerate(polys, 1):
                        if poly is None or len(poly) == 0:
                            continue
                        poly = poly.reshape(-1, 1, 2).astype(np.int32)
                        mm = np.zeros((h,w), np.uint8)
                        cv2.fillPoly(mm, [poly], 255)

                        # Object PNG nền trong suốt
                        drop = object_out_dir / f"{i+1}_{k}_cls{class_id}.png"
                        extract_object(img_bgr, mm, drop)
                        print(f"------ Cut object {i+1}_{k}_cls{class_id}.png")

                    for poly in polys:
                        instances.append({"cls": class_id, "conf": conf, "poly": poly})

            txt_path = labels_dir / f"{stem}.txt"
            label_obj(
                txt_path=txt_path,
                instances=instances,
                width=w, height=h
            )
            print(f"=== Hoàn tất ảnh: {stem}, tổng số object: {len(instances)}\n")
        print(f"--- Xong {cls_name}, kết quả ở: {OUT_DIR}/{cls_name}")

if __name__ == "__main__":
    main()