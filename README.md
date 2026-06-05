# Copy-Paste Augmentation Hạt Điều

Dự án sinh dữ liệu ảo, augmentation cho bài toán hạt điều bằng phương pháp copy-paste. Dự án nhầm tạo 1 pipeline làm tăng cường dữ liệu hạt điều ở môi trường công nghiệp. Giảm nỗ lực của con người trong việc phân loại và đánh nhãn trên 1 ảnh dữ liệu được kết hợp từ 13 loại

## Flow pipeline

<p align="center">
  <img src="visualize/Project-Flow.png" alt="Copy-paste augmentation pipeline" width="900">
</p>

<p align="center">
  <img src="visualize/Synthetic-Data.png" alt="Genarate Synthetic Data" width="900">
</p>

<p align="center">
  <em>Pipeline: Copy-paste tạo synthetic data các ảnh kết hợp 13 loại hạt điều từ ảnh các loại hạt điều riêng lẻ.</em>
</p>

## Chuẩn bị môi trường

Yêu cầu:

- Python 3.10+
- Git

Tạo môi trường ảo trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tạo môi trường ảo trên macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chạy bằng GPU/CUDA, nên cài `torch` và `torchvision` đúng build của máy trước khi cài toàn bộ dependencies.

## Cấu trúc thư mục

```text
Hatdieu/
|-- src/
|   |-- pred_labels.py
|   |-- mix.py
|   |-- check_labels.py
|   |-- augment.py
|   `-- count_obj_per_class.py
|-- background/
|-- data_hatdieu/
|-- model/
|-- yaml/
|-- pred_labels/        # sinh ra sau bước tách object
|-- mix_data/           # sinh ra sau bước copy-paste
|-- data_augment/       # sinh ra sau bước augment
|-- requirements.txt
`-- README.md
```

### Ý nghĩa các thư mục chính

- `src/`: chứa toàn bộ mã nguồn của pipeline sinh dữ liệu.
- `background/`: chứa ảnh nền dùng để dán object.
- `data_hatdieu/`: dữ liệu ảnh nguồn theo từng class dùng cho chạy đầy đủ.
- `model/`: chứa trọng số YOLO segmentation và checkpoint SAM.
- `yaml/`: chứa cấu hình dataset và train YOLO.
- `pred_labels/`: thư mục đầu ra của bước tách object, gồm `predict/`, `labels/` và `object/`.
- `mix_data/`: thư mục đầu ra của bước copy-paste, gồm `images/` và `labels/`.
- `data_augment/`: thư mục đầu ra của bước augment bổ sung.

## Cách chạy

Thứ tự chạy khuyến nghị:

1. Tách object và sinh label trung gian từ ảnh nguồn:

```powershell
python src/pred_labels.py
```

2. Tạo ảnh synthetic bằng copy-paste:

```powershell
python src/mix.py
```

Script sẽ mở cửa sổ OpenCV để chọn ROI trên ảnh nền. Nhấn `Enter` để xác nhận vùng chọn, nhấn `Esc` để xóa ROI và chọn lại.

3. Kiểm tra nhanh label của ảnh đã mix:

```powershell
python src/check_labels.py
```

4. Augment thêm dataset đã sinh:

```powershell
python src/augment.py
```

5. Thống kê số object theo class:

```powershell
python src/count_obj_per_class.py
```

## Cấu hình quan trọng

### 1. Đường dẫn dataset và output

- File: `src/config.py`
- Các biến quan trọng:
  - `DATA_HATDIEU_DIR` → thư mục ảnh gốc đầu vào.
  - `PRED_LABELS_DIR` → thư mục output của bước tách object.
  - `MIX_DATA_DIR` → thư mục output của bước copy-paste.
  - `DATA_AUGMENT_DIR` → thư mục output của bước augment.

### 2. Cấu hình tách object và sinh label

- File: `src/pred_labels.py`
- Các biến quan trọng:
  - `MODEL`, `SAM` → đường dẫn checkpoint YOLO/SAM.
  - `CONFIDENT` → ngưỡng confidence cho YOLO.
  - `DEVICE` → `auto`, `cpu`, `cuda:0`, ...
  - `BLUR`, `CLOSE`, `OPEN`, `FILL_HOLES`, `WATERSHED` → bật/tắt bước hậu xử lý mask.
  - `MIN_AREA_RATIO` → lọc object quá nhỏ.

### 3. Cấu hình sinh synthetic data

- File: `src/mix.py`
- Các biến quan trọng:
  - `N_MIX_IMG` → số ảnh synthetic tạo ra.
  - `IOU` → ngưỡng overlap giữa các object.
  - `NUM_OBJECT` → số lượng object trên mỗi ảnh mix.
  - `RESIZE`, `RANDOM_FLIP`, `RANDOM_ROTATE` → bật/tắt các biến đổi object trước khi dán.
  - `SHADOW_MODE`, `X_LIGHT`, `Y_LIGHT`, `BASE_*`, `CAST_*` → điều chỉnh bóng đổ của object.

### 4. Cấu hình kiểm tra label nhanh

- File: `src/check_labels.py`
- Các biến quan trọng:
  - `CHECK_IMAGES_DIR`, `CHECK_LABELS_DIR` → thư mục ảnh/label cần QC.
  - `N_SAMPLES` → số ảnh random dùng để QC.
  - `DISPLAY_SIZE` → kích thước cửa sổ hiển thị.
  - `MODE` → chọn class hiển thị, `-1` là hiển thị tất cả.

### 5. Cấu hình augment dữ liệu

- File: `src/augment.py`
- Các biến quan trọng:
  - `SCALE_DATASET` → số lượng bản augment sinh ra từ mỗi ảnh gốc.
  - `GEOMETRIC` → bật/tắt các phép biến đổi hình học.
  - `PHOTOMETRIC` → bật/tắt các phép biến đổi ánh sáng và nhiễu.
  - `PROBABILITY` → xác suất áp dụng từng phép augment.
  - `PARAMS` → tham số biên độ cho từng phép augment.
