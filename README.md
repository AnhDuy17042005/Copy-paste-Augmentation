# Hatdieu

## Cấu trúc thư mục

- **background/** — chứa ảnh background băng tải  
- **Data_hatdieu/** — chứa dataset hạt điều  
- **Data_hatdieu_test/** — dataset nhỏ để test thuật toán  

- **model/** — chứa các file model `.pt`  
  - `sam_vit_b_01ec64.pth` → [Link download](https://drive.google.com/file/d/1E9LgfKnAEFQ5f5bxvexcf4IlhpLdOlqy/view?usp=sharing)

- **pred_labels/**  
  - `labels/` — chứa file `.txt` label từng hạt  
  - `object/` — chứa các hạt đã tách ra  
  - `predict/` — visualize dự đoán từ mô hình  

- **mixe_data/**  
  - `images/` — chứa ảnh đã mix  
  - `labels/` — chứa label của từng hạt sau khi đã mix  

- **check_labels/** — visualize từ file `.txt` polygon từng hạt lên ảnh để kiểm tra lại  
- **src/** — chứa source code  
- **yaml/** — chứa file `.yaml`

---

## Flow

| src | Mô tả chức năng |
|------|------------------|
| `pred_labels.py` | Tách từng hạt từ dataset |
| `mix.py` | Ghép các hạt đã tách kèm theo label |
| `check_labels.py` | Kiểm tra lại các ảnh đã mix |
| `count_obj_per_class.py` | Kiểm tra cân bằng giữa các class |

---
