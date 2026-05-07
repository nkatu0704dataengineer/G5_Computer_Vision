"""
train.py
Huấn luyện mô hình CNN:
  Giai đoạn 1 – Train classification head (backbone đóng băng)
  Giai đoạn 2 – Fine-tune backbone (tuỳ chọn)

Lưu model vào:
    artifacts/models/

TensorBoard logs:
    artifacts/models/logs_stage1/
    artifacts/models/logs_stage2/
"""

import os
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import cnn_config as config
import cnn_dataset as dataset
import cnn_model as model_module
# ─────────────────────────────────────────────
# 1. CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(stage: str = "stage1") -> list:
    """
    Tạo callback cho training.
    """
    # đảm bảo folder model tồn tại
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # tensorboard log
    log_dir = os.path.join(
        config.MODEL_DIR,
        f"logs_{stage}"
    )
    callbacks = [
        # save best model
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # early stopping
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # reduce learning rate
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # tensorboard
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=0
        )
    ]
    return callbacks


# ─────────────────────────────────────────────
# 2. TRAIN HEAD
# ─────────────────────────────────────────────
def train_head(
    model: tf.keras.Model,
    train_gen,
    val_gen
) -> tf.keras.callbacks.History:
    print("\n" + "═" * 55)
    print("  GIAI ĐOẠN 1: TRAIN CLASSIFICATION HEAD")
    print("═" * 55)
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(stage="stage1"),
        verbose=1
    )
    return history
# ─────────────────────────────────────────────
# 3. FINE-TUNE
# ─────────────────────────────────────────────
def train_fine_tune(
    model: tf.keras.Model,
    train_gen,
    val_gen
) -> tf.keras.callbacks.History:
    print("\n" + "═" * 55)
    print("  GIAI ĐOẠN 2: FINE-TUNE BACKBONE")
    print("═" * 55)
    # mở khóa backbone
    model = model_module.enable_fine_tune(model)
    history = model.fit(
        train_gen,
        epochs=config.FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(stage="stage2"),
        verbose=1
    )
    return history
# ─────────────────────────────────────────────
# 4. SAVE / LOAD MODEL
# ─────────────────────────────────────────────
def save_model(model: tf.keras.Model):
    """
    Lưu model CNN.
    """
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    print("\n[train] Saved CNN model:")
    print(f"         {config.MODEL_PATH}")

def load_model() -> tf.keras.Model:
    """
    Load model CNN đã train.
    """
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"\nKhông tìm thấy model:\n"
            f"{config.MODEL_PATH}\n"
            f"Hãy train model trước."
        )
    model = tf.keras.models.load_model(config.MODEL_PATH)
    print(f"\n[train] Loaded model:")
    print(f"         {config.MODEL_PATH}")
    return model

# ─────────────────────────────────────────────
# 5. TRAIN RESULT
# ─────────────────────────────────────────────
def print_training_results(
    history,
    stage: str = "Train"
):
    """
    In kết quả training tốt nhất.
    """
    val_acc = max(
        history.history.get("val_accuracy", [0])
    )
    val_loss = min(
        history.history.get("val_loss", [0])
    )
    print("\n" + "─" * 55)
    print(f"[{stage}] BEST RESULT")
    print("─" * 55)
    print(f"val_accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"val_loss     : {val_loss:.4f}")
    print("─" * 55)
