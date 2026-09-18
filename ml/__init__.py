"""Causal machine learning pipeline for the Wisconsin breast cancer dataset.

This package implements the full workflow described in the project README:

1. ``data``        - load / regenerate the raw dataset
2. ``features``     - statistical feature selection ("causal score")
3. ``causal_graph``  - domain-informed DAG + Pearl do-calculus effect estimation
4. ``modeling``      - multi-model training and evaluation
5. ``interpret``     - SHAP-based model interpretation
6. ``pipeline``      - orchestrates the steps above end to end
"""

from ml.config import PROJECT_ROOT, DATA_PATH, MODELS_DIR, OUTPUTS_DIR

__all__ = ["PROJECT_ROOT", "DATA_PATH", "MODELS_DIR", "OUTPUTS_DIR"]
