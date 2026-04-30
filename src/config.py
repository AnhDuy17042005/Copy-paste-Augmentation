from pathlib import Path


# ========================== PATH CONFIGS =================================

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Shared asset folders
MODEL_DIR = BASE_DIR / "model"
BACKGROUND_DIR = BASE_DIR / "background"

# Main dataset folders
DATA_HATDIEU_DIR = BASE_DIR / "data_hatdieu_demo"
PRED_LABELS_DIR = BASE_DIR / "pred_labels_demo"
MIX_DATA_DIR = BASE_DIR / "mix_data_demo"
DATA_AUGMENT_DIR = BASE_DIR / "data_augment_demo"

# Mix dataset subfolders
MIX_IMAGES_DIR = MIX_DATA_DIR / "images"
MIX_LABELS_DIR = MIX_DATA_DIR / "labels"

# Augmented dataset subfolders
AUGMENT_IMAGES_DIR = DATA_AUGMENT_DIR / "images"
AUGMENT_LABELS_DIR = DATA_AUGMENT_DIR / "labels"

# =========================================================================
