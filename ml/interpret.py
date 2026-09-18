"""SHAP-based interpretation of the best-performing tree model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_importance(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (importance_frame, raw_shap_values_for_positive_class)."""
    import shap

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        shap_values = raw[1]
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        # shap >= 0.45 returns (n_samples, n_features, n_classes)
        shap_values = raw[:, :, 1]
    else:
        shap_values = raw

    importance = pd.DataFrame(
        {"Feature": X.columns, "SHAP_Importance": np.abs(shap_values).mean(axis=0)}
    ).sort_values("SHAP_Importance", ascending=False)

    return importance, shap_values
