"""
config.py
Cấu hình toàn bộ project CNN + SVM Face Recognition
Hỗ trợ:
    - Local
    - Google Colab
    - Feature extraction cho SVM

"""
import os
# ══════════════════════════════════════════════════════════
# 1. ROOT PROJECT DIRECTORY
# ══════════════════════════════════════════════════════════

# LOCAL:
# BASE_DIR = thư mục project hiện tại

# COLAB:
# Uncomment nếu chạy trên Google Colab + Drive
# GDRIVE_BASE = "/content/drive/MyDrive/G5_CNN_SVM"
GDRIVE_BASE = None
if GDRIVE_BASE:
    BASE_DIR = GDRIVE_BASE
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[config] BASE_DIR: {BASE_DIR}")
# ══════════════════════════════════════════════════════════
# 2. DATASET PATH
# ══════════════════════════════════════════════════════════
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
# Dataset structure:
#
# dataset/
# ├── train/
# │   ├── person_1/
# │   ├── person_2/
# │   └── ...
# │
# └── val/
#     ├── person_1/
#     ├── person_2/
#     └── ...

# ══════════════════════════════════════════════════════════
# 3. MODEL / FEATURE PATH
# ══════════════════════════════════════════════════════════
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
FEATURE_DIR = os.path.join(ARTIFACT_DIR, "features")
# CNN model
MODEL_PATH = os.path.join(MODEL_DIR, "face_cnn.keras")
# Optional weights
WEIGHTS_PATH = os.path.join(MODEL_DIR, "face_cnn.weights.h5")
# Feature files for SVM
TRAIN_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "train_features.npz"
)
VAL_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "val_features.npz"
)
# SVM model
SVM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_model.pkl"
)
# Auto create folders
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
# ══════════════════════════════════════════════════════════
# 4. IMAGE CONFIG
# ══════════════════════════════════════════════════════════
IMG_WIDTH  = 224
IMG_HEIGHT = 224

IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

CHANNELS = 3
# ══════════════════════════════════════════════════════════
# 5. CNN BACKBONE
# ══════════════════════════════════════════════════════════
# Supported:
#   "vgg16"
#   "resnet50"
BACKBONE = "vgg16"
# ══════════════════════════════════════════════════════════
# 6. TRAINING CONFIG
# ══════════════════════════════════════════════════════════
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
# Fine-tuning
FINE_TUNE = False
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5
# Open last N layers
FINE_TUNE_AT = 15
# ══════════════════════════════════════════════════════════
# 7. DATA AUGMENTATION
# ══════════════════════════════════════════════════════════
USE_AUGMENTATION = True
# ═════════════════════════════════════════════════════════
# 8. RANDOM SEED
# ══════════════════════════════════════════════════════════
RANDOM_SEED = 42
# ══════════════════════════════════════════════════════════
# 9. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════
# fc2 output size
FEATURE_DIM = 256
# ══════════════════════════════════════════════════════════
# 10. DEBUG INFO
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n========== CONFIG ==========")
    print(f"BASE_DIR           : {BASE_DIR}")
    print(f"\nTRAIN_DIR          : {TRAIN_DIR}")
    print(f"VAL_DIR            : {VAL_DIR}")
    print(f"\nMODEL_PATH         : {MODEL_PATH}")
    print(f"\nTRAIN_FEATURE_PATH : {TRAIN_FEATURE_PATH}")
    print(f"VAL_FEATURE_PATH   : {VAL_FEATURE_PATH}")
    print(f"\nSVM_MODEL_PATH     : {SVM_MODEL_PATH}")
    print(f"\nBACKBONE           : {BACKBONE}")
    print(f"BATCH_SIZE         : {BATCH_SIZE}")
    print(f"EPOCHS             : {EPOCHS}")
    print("=" * 35)
