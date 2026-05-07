"""
extract_feature.py

Trích xuất feature vector từ mô hình CNN đã train.

Vector đặc trưng này (256-D) là output của lớp Dense fc2
– trước lớp Dropout và Output – dùng làm input cho SVM
hoặc các bộ phân loại khác.

Hai chế độ:
  1. extract_feature(image_path)
  2. extract_features_from_dir(dir_path)

"""

import os
import numpy as np
import tensorflow as tf
import cnn_config as config
import dataset as ds
import model as model_module
# ─────────────────────────────────────────────
# CACHE MODEL
# ─────────────────────────────────────────────
_face_model = None
_feature_extractor = None
# ─────────────────────────────────────────────
# LOAD EXTRACTOR
# ─────────────────────────────────────────────
def _get_extractor():
    """
    Load model 1 lần duy nhất rồi cache lại.
    """
    global _face_model, _feature_extractor
    if _feature_extractor is None:

        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy model: {config.MODEL_PATH}"
            )
        print(f"[extract] Loading model: {config.MODEL_PATH}")

        _face_model = tf.keras.models.load_model(config.MODEL_PATH)

        _feature_extractor = model_module.build_feature_extractor(
            _face_model
        )
    return _feature_extractor
# ─────────────────────────────────────────────
# EXTRACT SINGLE IMAGE
# ─────────────────────────────────────────────
def extract_feature(image_path: str) -> np.ndarray:
    """
    Trích xuất feature vector từ 1 ảnh.
    Returns:
        vector shape (256,)
    """
    extractor = _get_extractor()
    img_array = ds.load_and_preprocess_image(image_path)
    feature_map = extractor.predict(img_array, verbose=0)
    feature_vector = feature_map.flatten()
    return feature_vector
# ─────────────────────────────────────────────
# EXTRACT BATCH
# ─────────────────────────────────────────────
def extract_features_batch(image_paths: list) -> np.ndarray:
    """
    Trích xuất feature cho nhiều ảnh.
    Returns:
        features shape (N, 256)
    """
    extractor = _get_extractor()
    features = []
    total = len(image_paths)
    for idx, path in enumerate(image_paths, 1):
        print(f"  [{idx:>4}/{total}] {os.path.basename(path)}")
        img_array = ds.load_and_preprocess_image(path)
        feat = extractor.predict(
            img_array,
            verbose=0
        ).flatte
        features.append(feat)
    return np.array(features)
# ─────────────────────────────────────────────
# EXTRACT DIRECTORY
# ─────────────────────────────────────────────
SUPPORTED_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}
def extract_features_from_dir(root_dir: str):
    """
    Trích xuất feature từ toàn bộ thư mục dataset.
    Structure:
        root/
            class1/
            class2/
    Returns:
        X
        y
        classes
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {root_dir}"
        )
    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }
    all_paths = []
    all_labels = []
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXT:
                all_paths.append(
                    os.path.join(cls_dir, fname)
                )
                all_labels.append(
                    class_to_idx[cls]
                )
    print("\n" + "═" * 55)
    print(f"[extract] Dataset      : {root_dir}")
    print(f"[extract] Total images : {len(all_paths)}")
    print(f"[extract] Classes      : {classes}")
    print("═" * 55)
    X = extract_features_batch(all_paths)
    y = np.array(all_labels)

    print("\n[extract] DONE")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y, classes
# ─────────────────────────────────────────────
# SAVE FEATURES
# ─────────────────────────────────────────────
def save_features(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str
):
    """
    Save features → .npz
    """
    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )
    np.savez_compressed(
        save_path,
        X=X,
        y=y
    )
    print(f"[extract] Saved: {save_path}")
# ─────────────────────────────────────────────
# LOAD FEATURES
# ─────────────────────────────────────────────
def load_features(save_path: str):
    """
    Load features từ .npz
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {save_path}"
        )
    data = np.load(save_path)
    print(f"[extract] Loaded: {save_path}")
    return data["X"], data["y"]
# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" FEATURE EXTRACTION DEMO ")
    print("=" * 55)
    if os.path.exists(config.TRAIN_DIR):
        X, y, classes = extract_features_from_dir(
            config.TRAIN_DIR
        )
        feature_path = os.path.join(
            config.FEATURE_DIR,
            "train_features.npz"
        )
        save_features(
            X,
            y,
            feature_path
        )
        print("\nFeature extraction completed.")
