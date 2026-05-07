"""
model.py
─────────────────────────────────────────────────────────────
Xây dựng kiến trúc mô hình CNN cho nhận diện khuôn mặt.

Chiến lược:
  1. Dùng backbone pretrained (VGG16 / ResNet50) làm feature extractor
  2. Thêm classification head gồm các lớp Dense
  3. Hỗ trợ fine-tune backbone
  4. Hỗ trợ trích xuất feature cho SVM
─────────────────────────────────────────────────────────────
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
import cnn_config as config
# ─────────────────────────────────────────────
# 1. TẠO BACKBONE PRETRAINED
# ─────────────────────────────────────────────
def _build_backbone() -> tf.keras.Model:
    """
    Tải backbone pretrained từ ImageNet.
    """
    if config.BACKBONE.lower() == "vgg16":
        backbone = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(
                config.IMG_HEIGHT,
                config.IMG_WIDTH,
                config.CHANNELS
            )
        )
    elif config.BACKBONE.lower() == "resnet50":

        backbone = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(
                config.IMG_HEIGHT,
                config.IMG_WIDTH,
                config.CHANNELS
            )
        )
    else:
        raise ValueError(
            f"Backbone không hợp lệ: {config.BACKBONE}"
        )
    # freeze backbone giai đoạn đầu
    backbone.trainable = False
    print(
        f"[model] Backbone: "
        f"{config.BACKBONE.upper()} | "
        f"trainable=False"
    )
    return backbone
# ─────────────────────────────────────────────
# 2. BUILD FULL MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Xây dựng mô hình CNN:
        Backbone → GAP → Dense → Output
    """
    backbone = _build_backbone()
    x = backbone.output
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(
        name="gap"
    )(x)
    # FC1
    x = layers.Dense(
        512,
        activation="relu",
        name="fc1"
    )(x)
    x = layers.BatchNormalization(
        name="bn1"
    )(x)
    x = layers.Dropout(
        0.5,
        name="drop1"
    )(x)
    # FC2
    x = layers.Dense(
        256,
        activation="relu",
        name="fc2"
    )(x)
    x = layers.BatchNormalization(
        name="bn2"
    )(x)
    x = layers.Dropout(
        0.3,
        name="drop2"
    )(x)
    # Output
    output = layers.Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(x)
    model = Model(
        inputs=backbone.input,
        outputs=output,
        name="FaceCNN"
    )
    print(
        f"[model] Đã build model | "
        f"num_classes={num_classes}"
    )
    return model


# ─────────────────────────────────────────────
# 3. COMPILE MODEL
# ─────────────────────────────────────────────

def compile_model(
    model: tf.keras.Model,
    learning_rate: float = None
) -> tf.keras.Model:
    """
    Compile model.
    """
    lr = learning_rate or config.LEARNING_RATE
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"[model] Compiled | lr={lr:.0e}")
    return model


# ─────────────────────────────────────────────
# 4. ENABLE FINE-TUNE
# ─────────────────────────────────────────────

def enable_fine_tune(model: tf.keras.Model) -> tf.keras.Model:
    """
    Mở khoá một phần backbone để fine-tune.
    """
    backbone = None
    # tìm backbone model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    if backbone is None:
        raise ValueError(
            "Không tìm thấy backbone model"
        )
    # mở backbone
    backbone.trainable = True
    # freeze các layer đầu
    for layer in backbone.layers[:-config.FINE_TUNE_AT]:
        layer.trainable = False

    trainable_count = sum(
        1 for layer in backbone.layers
        if layer.trainable
    )
    print(
        f"[model] Fine-tune bật | "
        f"{trainable_count} layers được mở"
    )
    # compile lại với lr nhỏ
    model = compile_model(
        model,
        learning_rate=config.FINE_TUNE_LR
    )
    return model


# ─────────────────────────────────────────────
# 5. FEATURE EXTRACTOR
# ─────────────────────────────────────────────

def build_feature_extractor(
    model: tf.keras.Model
) -> tf.keras.Model:
    """
    Tạo model trích xuất feature 256-D.
    """
    feature_output = model.get_layer(
        "fc2"
    ).output
    extractor = Model(
        inputs=model.input,
        outputs=feature_output,
        name="FeatureExtractor"
    )
    print(
        f"[model] Feature extractor | "
        f"output shape: {extractor.output_shape}"
    )
    return extractor


# ─────────────────────────────────────────────
# 6. MODEL SUMMARY
# ─────────────────────────────────────────────

def print_model_summary(model: tf.keras.Model):
    """
    In tóm tắt model.
    """
    model.summary()
    trainable_params = sum(
        tf.size(v).numpy()
        for v in model.trainable_variables
    )
    non_trainable_params = sum(
        tf.size(v).numpy()
        for v in model.non_trainable_variables
    )
    print(
        f"\nTrainable params : "
        f"{trainable_params:,}"
    )
    print(
        f"Non-trainable    : "
        f"{non_trainable_params:,}"
    )
