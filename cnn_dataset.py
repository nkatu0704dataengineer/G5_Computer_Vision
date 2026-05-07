"""
dataset.py
─────────────────────────────────────────────────────────────
Xử lý dữ liệu:
- Load dataset
- Augmentation
- Preprocessing
- Data generators
─────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.preprocessing import (
    image as keras_image
)

import config


# ─────────────────────────────────────────────
# CHECK DATASET STRUCTURE
# ─────────────────────────────────────────────

def _validate_dataset_structure():

    for directory in [config.TRAIN_DIR, config.VAL_DIR]:

        if not os.path.exists(directory):

            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {directory}"
            )

        subdirs = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))
        ]

        if len(subdirs) == 0:

            raise ValueError(
                f"Dataset rỗng: {directory}"
            )


# ─────────────────────────────────────────────
# TRAIN GENERATOR
# ─────────────────────────────────────────────

def get_train_generator():

    _validate_dataset_structure()

    if config.USE_AUGMENTATION:

        datagen = ImageDataGenerator(

            rescale=1.0 / 255.0,

            horizontal_flip=True,

            rotation_range=15,

            zoom_range=0.15,

            width_shift_range=0.1,

            height_shift_range=0.1,

            brightness_range=[0.8, 1.2],

            fill_mode="nearest"
        )

    else:

        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0
        )

    generator = datagen.flow_from_directory(

        directory=config.TRAIN_DIR,

        target_size=(
            config.IMG_HEIGHT,
            config.IMG_WIDTH
        ),

        batch_size=config.BATCH_SIZE,

        class_mode="categorical",

        shuffle=True,

        seed=config.RANDOM_SEED
    )

    return generator


# ─────────────────────────────────────────────
# VALIDATION GENERATOR
# ─────────────────────────────────────────────

def get_val_generator():

    _validate_dataset_structure()

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    generator = datagen.flow_from_directory(

        directory=config.VAL_DIR,

        target_size=(
            config.IMG_HEIGHT,
            config.IMG_WIDTH
        ),

        batch_size=config.BATCH_SIZE,

        class_mode="categorical",

        shuffle=False
    )

    return generator


# ─────────────────────────────────────────────
# CLASS INFO
# ─────────────────────────────────────────────

def get_class_names():

    _validate_dataset_structure()

    classes = sorted([

        d for d in os.listdir(config.TRAIN_DIR)

        if os.path.isdir(
            os.path.join(config.TRAIN_DIR, d)
        )
    ])

    return classes


def get_num_classes():

    return len(get_class_names())


# ─────────────────────────────────────────────
# DATASET INFO
# ─────────────────────────────────────────────

def print_dataset_info():

    classes = get_class_names()

    total_train = 0
    total_val = 0

    for cls in classes:

        train_cls_dir = os.path.join(
            config.TRAIN_DIR,
            cls
        )

        val_cls_dir = os.path.join(
            config.VAL_DIR,
            cls
        )

        if os.path.exists(train_cls_dir):

            total_train += len([
                f for f in os.listdir(train_cls_dir)
                if os.path.isfile(
                    os.path.join(train_cls_dir, f)
                )
            ])

        if os.path.exists(val_cls_dir):

            total_val += len([
                f for f in os.listdir(val_cls_dir)
                if os.path.isfile(
                    os.path.join(val_cls_dir, f)
                )
            ])

    print("\n" + "═" * 55)
    print(" DATASET INFO ")
    print("═" * 55)

    print(f"Train dir   : {config.TRAIN_DIR}")
    print(f"Val dir     : {config.VAL_DIR}")

    print(f"\nNum classes : {len(classes)}")
    print(f"Classes     : {classes}")

    print(f"\nTrain images: {total_train}")
    print(f"Val images  : {total_val}")

    print("═" * 55)


# ─────────────────────────────────────────────
# SINGLE IMAGE PREPROCESS
# ─────────────────────────────────────────────

def load_and_preprocess_image(
    image_path: str
) -> np.ndarray:

    """
    Load + preprocess 1 image.

    Returns:
        shape:
            (1, H, W, 3)
    """

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {image_path}"
        )

    img = keras_image.load_img(

        image_path,

        target_size=(
            config.IMG_HEIGHT,
            config.IMG_WIDTH
        )
    )

    img_array = keras_image.img_to_array(img)

    img_array = img_array.astype("float32")

    img_array /= 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print_dataset_info()

    train_gen = get_train_generator()

    val_gen = get_val_generator()

    print("\nGenerator test successful.")

    print(f"Train batches: {len(train_gen)}")
    print(f"Val batches  : {len(val_gen)}")
