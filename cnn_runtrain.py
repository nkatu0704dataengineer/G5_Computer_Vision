"""
run_train.py

FULL CNN TRAINING PIPELINE
Pipeline:
    1. Check environment
    2. Load dataset
    3. Build CNN model
    4. Train classification head
    5. Fine-tune backbone (optional)
    6. Save CNN model
    7. Extract CNN features
    8. Save features for SVM

Run:
    python run_train.py
"""

import os
import tensorflow as tf
import numpy as np
# project modules
import config
import dataset
import model as model_module
import cnn_train as train_module
import extract_feature

# ─────────────────────────────────────────────
# RANDOM SEED
# ─────────────────────────────────────────────
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
# ─────────────────────────────────────────────
# ENVIRONMENT CHECK
# ─────────────────────────────────────────────

def check_environment():
    print("\n" + "═" * 55)
    print("  ENVIRONMENT CHECK")
    print("═" * 55)
    print(f"TensorFlow version : {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available      : {len(gpus)}")
        print("Training device    : GPU")
    else:
        print("GPU available      : 0")
        print("Training device    : CPU")
    # dataset check
    for d in [config.TRAIN_DIR, config.VAL_DIR]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"\nDataset folder not found:\n{d}\n\n"
                f"Expected structure:\n"
                f"dataset/\n"
                f"   train/\n"
                f"      class_1/\n"
                f"      class_2/\n"
                f"   val/\n"
                f"      class_1/\n"
                f"      class_2/\n"
            )
    print("\nDataset structure OK ✓")
    print("═" * 55)
# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("     FACE RECOGNITION – CNN PIPELINE")
    print("=" * 60)
    # ==========================================
    # STEP 1 — ENV CHECK
    # ==========================================
    check_environment()

    # =========================================
    # STEP 2 — LOAD DATASET
    # ==========================================

    print("\n[STEP 2] LOAD DATASET")
    dataset.print_dataset_info()
    num_classes = dataset.get_num_classes()
    train_gen = dataset.get_train_generator()
    val_gen   = dataset.get_val_generator()
    print(f"\nClass indices:")
    print(train_gen.class_indices)

    # ==========================================
    # STEP 3 — BUILD MODEL
    # ==========================================

    print("\n" + "═" * 55)
    print("  BUILD CNN MODEL")
    print("═" * 55)
    model = model_module.build_model(
        num_classes=num_classes
    )
    model = train_module.compile_model(
        model,
        learning_rate=config.LEARNING_RATE
    )
    model_module.print_model_summary(model)
    # =========================================
    # STEP 4 — TRAIN HEAD
    # ==========================================
    history1 = train_module.train_head(
        model,
        train_gen,
        val_gen
    )
    train_module.print_training_results(
        history1,
        stage="Stage 1"
    )
    # ==========================================
    # STEP 5 — FINE-TUNE
    # ==========================================
    if config.FINE_TUNE:
        history2 = train_module.train_fine_tune(
            model,
            train_gen,
            val_gen
        )
        train_module.print_training_results(
            history2,
            stage="Stage 2 Fine-Tune"
        )
    # ==========================================
    # STEP 6 — SAVE CNN MODEL
    # ==========================================
    print("\n" + "═" * 55)
    print("  SAVE CNN MODEL")
    print("═" * 55)
    train_module.save_model(model)
    # ==========================================
    # STEP 7 — FEATURE EXTRACTION
    # ==========================================
    print("\n" + "═" * 55)
    print("  FEATURE EXTRACTION FOR SVM")
    print("═" * 55)
    # create directories
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # ==========================================
    # TRAIN FEATURES
    # ==========================================
    print("\n[1/2] Extract TRAIN features")
    X_train, y_train, classes = (
        extract_feature.extract_features_from_dir(
            config.TRAIN_DIR
        )
    )
    train_feature_path = os.path.join(
        config.FEATURE_DIR,
        "train_features.npz"
    )
    extract_feature.save_features(
        X_train,
        y_train,
        train_feature_path
    )
    # ==========================================
    # VALIDATION FEATURES
    # =========================================
    print("\n[2/2] Extract VALIDATION features")
    X_val, y_val, _ = (
        extract_feature.extract_features_from_dir(
            config.VAL_DIR
        )
    )
    val_feature_path = os.path.join(
        config.FEATURE_DIR,
        "val_features.npz"
    )
    extract_feature.save_features(
        X_val,
        y_val,
        val_feature_path
    )
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("                PIPELINE COMPLETED")
    print("=" * 60)

    print("\nCNN model saved:")
    print(config.MODEL_PATH)

    print("\nTrain features:")
    print(f"{train_feature_path}")
    print(f"Shape: {X_train.shape}")

    print("\nValidation features:")
    print(f"{val_feature_path}")
    print(f"Shape: {X_val.shape}")

    print("\nDetected classes:")
    print(classes)

    print("\nNext step:")
    print("python SVM/train.py")

    print("=" * 60)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()