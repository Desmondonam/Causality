"""Predictive modeling: train and evaluate several classifiers on the
causal-score-selected feature shortlist."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.config import RANDOM_STATE, TEST_SIZE

SCALED_MODELS = {"Logistic Regression", "SVM"}


def build_model_configs() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        # SVC's built-in `probability=True` is deprecated (sklearn >=1.9);
        # CalibratedClassifierCV is the supported way to get predict_proba.
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False),
    }


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_mean: float
    cv_std: float
    y_pred: np.ndarray = field(repr=False)
    y_pred_proba: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "Model": self.name,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1-Score": self.f1,
            "ROC-AUC": self.roc_auc,
            "CV_ROC_AUC_Mean": self.cv_mean,
            "CV_ROC_AUC_Std": self.cv_std,
        }


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, ModelResult], StandardScaler]:
    """Train every configured model, returning per-model results plus the
    (fit-on-train) scaler needed for the linear/SVM models at inference time."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: dict[str, ModelResult] = {}
    for name, model in build_model_configs().items():
        if name in SCALED_MODELS:
            fit_X, eval_X, cv_X = X_train_scaled, X_test_scaled, X_train_scaled
        else:
            fit_X, eval_X, cv_X = X_train, X_test, X_train

        model.fit(fit_X, y_train)
        y_pred = model.predict(eval_X)
        y_pred_proba = model.predict_proba(eval_X)[:, 1]

        cv_scores = cross_val_score(
            model, cv_X, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="roc_auc"
        )

        results[name] = ModelResult(
            name=name,
            model=model,
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )
    return results, scaler


def results_to_frame(results: dict[str, ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in results.values()])


def best_model_name(results: dict[str, ModelResult]) -> str:
    return max(results, key=lambda k: results[k].roc_auc)


def roc_curve_points(y_test: pd.Series, results: dict[str, ModelResult]) -> dict[str, tuple]:
    return {name: roc_curve(y_test, r.y_pred_proba)[:2] for name, r in results.items()}
