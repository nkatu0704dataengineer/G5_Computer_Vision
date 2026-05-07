# svm_model.py

import logging
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

from svm_config import DEFAULT_CONFIG

# --- LOGGING ---
logger = logging.getLogger("SVMClassifierModule")
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('svm_pipeline.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class SVMClassifierModule:

    def __init__(self, config=None):
        self.config = config if config else DEFAULT_CONFIG
        self.pipeline = None
        self.adaptive_threshold = 0.0
        self.classes_ = None

    def _build_pipeline(self):
        steps = [('scaler', StandardScaler())]

        if self.config['pca_enabled']:
            steps.append(('pca', PCA(n_components=self.config['pca_variance'])))

        steps.append(('svm', SVC(probability=True, cache_size=1000)))

        return Pipeline(steps)

    def fit(self, X_train, y_train):
        logger.info("Training model...")

        pipe = self._build_pipeline()

        grid_search = GridSearchCV(
            pipe,
            self.config['param_grid'],
            cv=self.config['cv_folds'],
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        self.pipeline = grid_search.best_estimator_
        self.classes_ = self.pipeline.named_steps['svm'].classes_

        logger.info(f"Best params: {grid_search.best_params_}")

        # --- Adaptive threshold (Out-of-fold) ---
        logger.info("Calculating adaptive threshold...")

        oof_probs = cross_val_predict(
            self.pipeline,
            X_train,
            y_train,
            cv=self.config['cv_folds'],
            method='predict_proba',
            n_jobs=-1
        )

        max_oof_probs = np.max(oof_probs, axis=1)

        if self.config['rejection_strategy'] == 'percentile':
            self.adaptive_threshold = np.quantile(
                max_oof_probs,
                self.config['threshold_value']
            )
        else:
            self.adaptive_threshold = self.config['threshold_value']

        logger.info(f"Threshold: {self.adaptive_threshold:.4f}")

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model not fitted!")

        probs = self.pipeline.predict_proba(X)
        sorted_probs = np.sort(probs, axis=1)

        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        max_probs = np.max(probs, axis=1)
        pred_indices = np.argmax(probs, axis=1)

        labels = self.classes_[pred_indices]

        is_rejected = max_probs < self.adaptive_threshold

        final_outputs = labels.astype(object)
        final_outputs[is_rejected] = "REJECTED"

        return [{
            "label": labels[i],
            "confidence": max_probs[i],
            "margin": margins[i],
            "is_rejected": is_rejected[i],
            "final_output": final_outputs[i]
        } for i in range(len(X))]

    def evaluate(self, X_test, y_test):
        results = self.predict(X_test)
        y_pred_raw = np.array([r['label'] for r in results])

        logger.info("=== EVALUATION ===")
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_raw):.4f}")

        valid_mask = ~np.array([r['is_rejected'] for r in results])

        if np.any(valid_mask):
            y_test_filt = np.array(y_test)[valid_mask]
            y_pred_filt = y_pred_raw[valid_mask]

            logger.info(f"Reject rate: {(~valid_mask).mean():.2%}")
            logger.info(f"Accuracy after reject: {accuracy_score(y_test_filt, y_pred_filt):.4f}")

            print(classification_report(y_test_filt, y_pred_filt))
        else:
            logger.warning("All samples rejected!")

    def save_model(self, path):
        joblib.dump({
            'pipeline': self.pipeline,
            'threshold': self.adaptive_threshold,
            'classes': self.classes_,
            'config': self.config
        }, path)

        logger.info(f"Saved model to {path}")

    def load_model(self, path):
        pack = joblib.load(path)

        self.pipeline = pack['pipeline']
        self.adaptive_threshold = pack['threshold']
        self.classes_ = pack['classes']
        self.config = pack['config']

        logger.info(f"Loaded model from {path}")
