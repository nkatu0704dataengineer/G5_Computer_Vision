"""
config.py
─────────────────────────────────────────────────────────────
Cấu hình cho pipeline SVM Face Recognition
─────────────────────────────────────────────────────────────
"""

import os

# ══════════════════════════════════════════════════════════
# 1. ROOT DIRECTORY
# ══════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nếu folder SVM nằm trong project chính:
#
# G5_CNN_SVM/
# ├── CNN/
# ├── SVM/
# └── artifacts/
#
# thì cần lùi lên 1 cấp:

PROJECT_ROOT = os.path.dirname(BASE_DIR)

# ══════════════════════════════════════════════════════════
# 2. FEATURE PATHS (từ CNN)
# ══════════════════════════════════════════════════════════

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")

FEATURE_DIR = os.path.join(ARTIFACT_DIR, "features")

TRAIN_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "train_features.npz"
)

VAL_FEATURE_PATH = os.path.join(
    FEATURE_DIR,
    "val_features.npz"
)

# ══════════════════════════════════════════════════════════
# 3. MODEL SAVE PATH
# ══════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")

SVM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 4. SVM CONFIG
# ══════════════════════════════════════════════════════════

DEFAULT_CONFIG = {

    # ── PCA ───────────────────────────────
    "pca_enabled": True,

    # Giữ lại 95% variance
    "pca_variance": 0.95,

    # ── Unknown Face Rejection ────────────
    #
    # percentile:
    #   threshold dựa theo phân phối confidence
    #
    # fixed:
    #   threshold cố định
    #
    "rejection_strategy": "percentile",

    # Từ chối 10% mẫu khó nhất
    "threshold_value": 0.1,

    # ── Cross Validation ──────────────────
    "cv_folds": 5,

    # ── Hyperparameter Search ────────────
    "param_grid": [

        # Linear SVM
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100]
        },

        # RBF SVM
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto"]
        }
    ]
}

# ══════════════════════════════════════════════════════════
# 5. DEBUG INFO
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n========== SVM CONFIG ==========")

    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")

    print(f"\nTRAIN_FEATURE_PATH  : {TRAIN_FEATURE_PATH}")
    print(f"VAL_FEATURE_PATH    : {VAL_FEATURE_PATH}")

    print(f"\nSVM_MODEL_PATH      : {SVM_MODEL_PATH}")

    print(f"\nPCA ENABLED         : {DEFAULT_CONFIG['pca_enabled']}")
    print(f"PCA VARIANCE        : {DEFAULT_CONFIG['pca_variance']}")

    print(f"\nREJECTION STRATEGY  : {DEFAULT_CONFIG['rejection_strategy']}")
    print(f"THRESHOLD VALUE     : {DEFAULT_CONFIG['threshold_value']}")

    print("=" * 40)