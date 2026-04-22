# Copy-Paste Augmentation Hạt Điều

Dự án sinh dữ liệu augmentation cho bài toán hạt điều bằng phương pháp copy-paste. Pipeline chính gồm tách object từ ảnh nguồn, chuẩn hóa polygon label, dán object lên nền mới để tạo ảnh synthetic, kiểm tra lại annotation và augment thêm dữ liệu khi cần.

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
|-- Data_hatdieu/
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
- `Data_hatdieu/`: dữ liệu ảnh nguồn theo từng class.
- `model/`: chứa trọng số YOLO segmentation và checkpoint SAM.
- `yaml/`: chứa cấu hình dataset và train YOLO.
- `pred_labels/`: thư mục đầu ra của bước tách object, gồm `predict/`, `labels/` và `object/`.
- `mix_data/`: thư mục đầu ra của bước copy-paste, gồm `images/` và `labels/`.
- `data_augment/`: thư mục đầu ra của bước augment bổ sung.

### Các script chính

- `src/pred_labels.py`: tách từng hạt điều từ ảnh nguồn bằng YOLO segmentation kết hợp SAM; đầu ra là object PNG nền trong suốt, ảnh visualize dự đoán và polygon label theo định dạng YOLO segmentation. Đây là bước chuẩn bị dữ liệu object để dùng lại ở các bước sau.
- `src/mix.py`: sinh ảnh synthetic bằng cách dán các object đã tách lên nền mới; script có cơ chế chọn ROI, giới hạn chồng lấn, cân bằng class, thêm xoay/lật ngẫu nhiên và mô phỏng bóng đổ để ảnh đầu ra tự nhiên hơn. Đây là bước cốt lõi của pipeline copy-paste augmentation.
- `src/check_labels.py`: hiển thị ngẫu nhiên ảnh đã mix cùng polygon label để kiểm tra trực quan chất lượng annotation. Script này phục vụ QC nhanh sau khi sinh dữ liệu.
- `src/augment.py`: augment thêm tập dữ liệu đã tạo bằng các phép biến đổi hình học và quang học như flip, rotate, crop, scale, shear, blur, noise và thay đổi màu sắc. Mục tiêu là tăng độ đa dạng dữ liệu đầu vào cho mô hình.
- `src/count_obj_per_class.py`: đếm số object theo từng class từ thư mục label. Script này dùng để đánh giá độ cân bằng class của dataset sau khi sinh dữ liệu.

## Cách chạy

Thứ tự chạy khuyến nghị:

1. Tách object và sinh label trung gian từ ảnh nguồn:

```powershell
python src/pred_labels.py
```

Đầu ra mặc định: `pred_labels/`

2. Tạo ảnh synthetic bằng copy-paste:

```powershell
python src/mix.py
```

Script sẽ mở cửa sổ OpenCV để chọn ROI trên ảnh nền. Nhấn `Enter` để xác nhận vùng chọn, nhấn `Esc` để xóa ROI và chọn lại. Đầu ra mặc định: `mix_data/`

3. Kiểm tra nhanh label của ảnh đã mix:

```powershell
python src/check_labels.py
```

4. Augment thêm dataset đã sinh:

```powershell
python src/augment.py
```

Đầu ra mặc định: `data_augment/`

5. Thống kê số object theo class:

```powershell
python src/count_obj_per_class.py
```

## Cấu hình quan trọng

Các script hiện được cấu hình bằng hằng số ở đầu file. Khi muốn thay đổi cách chạy, hãy sửa trực tiếp các biến tương ứng trước khi chạy lại script.

- Muốn đổi dữ liệu nguồn cho bước tách object: sửa `SOURE_DIR` trong `src/pred_labels.py`.
- Muốn đổi model YOLO segmentation hoặc checkpoint SAM: sửa `MODEL` và `SAM` trong `src/pred_labels.py`.
- Muốn đổi nơi lưu kết quả bước tách object: sửa `OUT_DIR` trong `src/pred_labels.py`.

- Muốn đổi thư mục chứa object dùng để dán: sửa `OBJECTS_DIR` trong `src/mix.py`. Thông thường biến này nên trỏ tới thư mục đầu ra của `src/pred_labels.py`, mặc định là `pred_labels/`.
- Muốn đổi ảnh nền hoặc thư mục nền: sửa `BACKGROUND_DIR` hoặc `background_path` trong `src/mix.py`.
- Muốn đổi nơi lưu ảnh synthetic: sửa `OUTPUT_DIR` trong `src/mix.py`.
- Muốn tăng hoặc giảm số lượng ảnh sinh ra: sửa `N_MIX_IMG` trong `src/mix.py`.
- Muốn tăng hoặc giảm số object trên mỗi ảnh: sửa `NUM_OBJECT` trong `src/mix.py`.
- Muốn nới hoặc siết mức chồng lấn giữa các object: sửa `IOU` trong `src/mix.py`.
- Muốn bật hoặc tắt các biến đổi ngẫu nhiên khi dán object: sửa `RESIZE`, `RANDOM_FLIP`, `RANDOM_ROTATE` trong `src/mix.py`.

- Muốn kiểm tra thư mục ảnh và label khác: sửa `MIX_DIR`, `SOURE_DIR`, `LABELS_DIR` trong `src/check_labels.py`.
- Muốn đổi số lượng ảnh dùng để QC nhanh: sửa `N_SAMPLES` trong `src/check_labels.py`.
- Muốn chỉ hiển thị một class cụ thể khi kiểm tra: sửa `MODE` trong `src/check_labels.py`.

- Muốn augment từ bộ dữ liệu khác: sửa `INPUT_IMAGES` và `INPUT_LABELS` trong `src/augment.py`.
- Muốn đổi thư mục lưu dữ liệu sau augment: sửa `OUTPUT_IMAGES` và `OUTPUT_LABELS` trong `src/augment.py`.
- Muốn tăng số lượng mẫu augment sinh thêm từ mỗi ảnh: sửa `SCALE_DATASET` trong `src/augment.py`.
- Muốn thay đổi loại augment hoặc xác suất augment: sửa các nhóm biến `GEOMETRIC`, `PHOTOMETRIC`, `PROBABILITY`, `PARAMS` trong `src/augment.py`.

- Muốn đếm thống kê trên thư mục label khác: sửa `LABEL_DIR` trong `src/count_obj_per_class.py`.

## Ghi chú

- Theo cấu hình hiện tại, pipeline mặc định đang đồng bộ theo chuỗi `Data_hatdieu/ -> pred_labels/ -> mix_data/ -> data_augment/`.
- Các thư mục đầu ra sẽ được tạo tự động nếu chưa tồn tại.
