# Hatdieu

## Cấu trúc thư mục

- background: chứa background băng tải
- Data_hatdieu: chứa dataset hạt điều
- Data_hatdieu_test: dataset nhỏ để test thuật toán
- model: chứa các file model .pt
- pred_labels:
    / labels: chứa file .txt label từng hạt
    / object: các hạt đã tách ra
    / predict: visualize dự đoán từ mô hình
- mixe_data
    / images: chứa ảnh đã mix
    / labels: chứa label của từng hạt sau khi đã mix

- check_labels: visualize từ file txt từng polygon của hạt lên ảnh lại xem chuẩn chưa
- src: source code
- yaml: chứa file yaml

## Flow

- pred_labels.py         ----> tách từng hạt từ dataset
- mix.py                 ----> ghép các hạt đã được tách kèm theo label
- check_labels.py        ----> kiểm tra lại các ảnh đã mix
- count_obj_per_class.py ----> kiểm tra class balance
