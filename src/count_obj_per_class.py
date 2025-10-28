import os
from collections import defaultdict
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
LABEL_DIR = BASE_DIR/"mix_data_ver2"/"labels"

def count_objects(label_dir):
    class_counts = defaultdict(int)
    total_objects = 0
    total_files = 0

    for filename in os.listdir(label_dir):
        if not filename.endswith(".txt"):
            continue
        total_files += 1
        file_path = os.path.join(label_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                class_counts[class_id] += 1
                total_objects += 1

    print("\n====== Thống kê số object mỗi class ======")
    for cid in sorted(class_counts.keys()):
        print(f"Class {cid:02d}: {class_counts[cid]:6d} hạt điều")
    print("="*40)
    print(f"Tổng cộng {total_objects} hạt điều\n")

if __name__ == "__main__":
    count_objects(LABEL_DIR)
