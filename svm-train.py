# train.py

import numpy as np
from sklearn.model_selection import train_test_split

from svm_model import SVMClassifierModule

# Fake CNN features
X_features = np.random.randn(1000, 2048)
y_labels = np.random.randint(0, 5, 1000)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_labels,
    test_size=0.2,
    stratify=y_labels,
    random_state=42
)

# Train
svm_mod = SVMClassifierModule()
svm_mod.fit(X_train, y_train)

# Evaluate
svm_mod.evaluate(X_test, y_test)