# G5_Computer_Vision

## CNN + SVM Hybrid Face Recognition Pipeline

Một hệ thống nhận diện khuôn mặt theo kiến trúc hybrid:

* CNN dùng để học đặc trưng ảnh khuôn mặt
* SVM dùng để phân loại trên feature vector trích xuất từ CNN
* Hỗ trợ:

  * Transfer Learning
  * Fine-tuning
  * Feature Extraction
  * PCA
  * Adaptive Rejection Threshold
  * Google Colab training
  * End-to-End pipeline

---

# 📌 Project Architecture

```text
RAW FACE IMAGE
       │
       ▼
 CNN Backbone (VGG16 / ResNet50)
       │
       ▼
Feature Vector (256-D)
       │
       ▼
 StandardScaler
       │
       ▼
 PCA (optional)
       │
       ▼
 SVM Classifier
       │
       ▼
Prediction / Rejection
```

---

# 🚀 Main Features

## CNN Module

✅ Transfer Learning với:

* VGG16
* ResNet50

✅ Two-stage training:

* Stage 1 → train classification head
* Stage 2 → fine-tune backbone

✅ Data augmentation

✅ TensorBoard logging

✅ Feature extraction cho SVM

✅ Google Colab compatible

---

## SVM Module

✅ SVM classifier:

* Linear kernel
* RBF kernel

✅ Hyperparameter tuning bằng GridSearchCV

✅ PCA dimensionality reduction

✅ Adaptive rejection threshold

✅ Confidence score + margin score

✅ Unknown / difficult sample rejection

---

# 📂 Project Structure

```text
G5_Computer_Vision/
│
├── CNN/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── run_train.py
│   └── extract_feature.py
│
├── SVM/
│   ├── config.py
│   ├── svm_model.py
│   └── train.py
│
├── dataset/
│   ├── train/
│   │   ├── person_1/
│   │   ├── person_2/
│   │   └── ...
│   │
│   └── val/
│       ├── person_1/
│       ├── person_2/
│       └── ...
│
├── artifacts/
│   ├── models/
│   └── features/
│
└── README.md
```

---

# 📁 Dataset Structure

Dataset phải có format:

```text
dataset/
    train/
        person_A/
            img1.jpg
            img2.jpg

        person_B/
            img1.jpg
            img2.jpg

    val/
        person_A/
        person_B/
```

Tên thư mục = class label.

---

# 🧠 CNN Training Pipeline

## Stage 1 — Train Classification Head

Backbone pretrained được đóng băng:

```python
backbone.trainable = False
```

Chỉ train:

* Dense
* BatchNorm
* Dropout
* Output layer

---

## Stage 2 — Fine-tuning

Mở khóa một phần backbone:

```python
for layer in backbone.layers[:-config.FINE_TUNE_AT]:
    layer.trainable = False
```

Fine-tune với learning rate nhỏ hơn.

---

# 🔥 Feature Extraction

Sau khi train CNN:

```text
IMAGE
  ↓
CNN Backbone
  ↓
fc2 layer
  ↓
256-D feature vector
```

Feature được lưu:

```text
artifacts/features/train_features.npz
artifacts/features/val_features.npz
```

---

# ⚡ SVM Pipeline

## Pipeline Architecture

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())
])
```

---

## Adaptive Threshold

Hệ thống tự tính threshold để reject sample khó:

```python
adaptive_threshold = np.quantile(
    max_oof_probs,
    threshold_value
)
```

Nếu confidence thấp:

* output = `"REJECTED"`

---

# 🛠 Installation

## Clone repository

```bash
git clone https://github.com/nkatu0704dataengineer/G5_Computer_Vision.git
cd G5_Computer_Vision
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

Hoặc:

```bash
pip install tensorflow scikit-learn numpy joblib
```

---

# ▶️ Run CNN Training

```bash
python CNN/run_train.py
```

Pipeline sẽ:

1. Load dataset
2. Build CNN
3. Train head
4. Fine-tune (optional)
5. Save model
6. Extract features

---

# ▶️ Run SVM Training

```bash
python SVM/train.py
```

Pipeline sẽ:

1. Load CNN features
2. Scale data
3. Apply PCA
4. Hyperparameter tuning
5. Train SVM
6. Evaluate model
7. Save trained model

---

# 💾 Saved Files

## CNN

```text
artifacts/models/face_cnn.keras
```

---

## Features

```text
artifacts/features/train_features.npz
artifacts/features/val_features.npz
```

---

## SVM

```text
artifacts/models/svm_model.pkl
```

---

# ⚙️ Configuration

## CNN Config

File:

```text
CNN/config.py
```

Có thể chỉnh:

* backbone
* epochs
* batch size
* augmentation
* fine-tuning
* learning rate
* dataset path

---

## SVM Config

File:

```text
SVM/config.py
```

Có thể chỉnh:

* PCA
* threshold
* rejection strategy
* param_grid
* CV folds

---

# 📊 Technologies Used

## Deep Learning

* TensorFlow
* Keras

## Machine Learning

* Scikit-learn

## Utilities

* NumPy
* Joblib

---

# 🎯 Why CNN + SVM?

CNN rất mạnh trong:

* feature extraction

SVM rất mạnh trong:

* classification trên feature embedding nhỏ
* hoạt động tốt với dataset vừa và nhỏ

Hybrid architecture thường:

* ổn định hơn pure CNN
* giảm overfitting
* dễ tune hơn

---

# 📈 Future Improvements

* Face detection integration
* Real-time webcam inference
* ArcFace / FaceNet embedding
* Triplet Loss
* ONNX export
* Docker deployment
* FastAPI inference API
* MLflow experiment tracking
* Airflow training pipeline

---

# 👨‍💻 Author

Developed by:
**nkatu0704dataengineer**

Project:
**G5 Computer Vision — CNN + SVM Face Recognition System**

---

# ⭐ Notes

Project được thiết kế theo hướng:

* modular
* production-friendly
* dễ mở rộng
* phù hợp research & deployment pipeline

Đặc biệt phù hợp để:

* học Computer Vision
* nghiên cứu feature extraction
* hybrid ML/DL systems
* portfolio AI Engineer / ML Engineer / Data Engineer oriented AI systems
