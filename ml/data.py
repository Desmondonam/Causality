"""Dataset loading / regeneration.

The project historically tracked ``data/data.csv`` via DVC against a Google
Drive remote. That remote isn't reachable from every environment, but the
underlying data is the public UCI/Kaggle "Wisconsin Diagnostic Breast
Cancer" dataset, which is also bundled with scikit-learn
(:func:`sklearn.datasets.load_breast_cancer`). ``generate_dataset`` rebuilds
a CSV with the original Kaggle column naming (``radius_mean``,
``concave points_worst``, ``diagnosis`` as ``M``/``B``, ...) so the rest of
the pipeline, the notebooks, and the tests can all run deterministically
without any external credentials.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml.config import DATA_PATH, TARGET


def _sklearn_name_to_kaggle(name: str) -> str:
    """Map a scikit-learn feature name to the original Kaggle column name.

    e.g. "mean radius" -> "radius_mean", "worst concave points" ->
    "concave points_worst", "radius error" -> "radius_se".
    """
    words = name.split()
    if words[0] in ("mean", "worst"):
        suffix = words[0]
        base = " ".join(words[1:])
    elif words[-1] == "error":
        suffix = "se"
        base = " ".join(words[:-1])
    else:  # pragma: no cover - defensive, sklearn names always match above
        raise ValueError(f"Unrecognized feature name: {name}")
    return f"{base}_{suffix}"


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    """Rebuild the Kaggle-schema breast cancer dataset from scikit-learn."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()

    rename_map = {name: _sklearn_name_to_kaggle(name) for name in bunch.feature_names}
    df = df.rename(columns=rename_map)

    # sklearn encodes target as 0=malignant, 1=benign; Kaggle uses M/B.
    df[TARGET] = np.where(df["target"] == 0, "M", "B")
    df = df.drop(columns=["target"])

    rng = np.random.default_rng(seed)
    df.insert(0, "id", rng.permutation(np.arange(100000, 100000 + len(df))))

    # Reproduce the trailing all-NaN "Unnamed: 32" artifact present in the
    # original Kaggle CSV export, since downstream code explicitly handles it.
    df["Unnamed: 32"] = np.nan

    # Order columns exactly like the original Kaggle file.
    ordered = ["id", TARGET] + [c for c in df.columns if c not in ("id", TARGET, "Unnamed: 32")] + ["Unnamed: 32"]
    return df[ordered]


def save_dataset(path=DATA_PATH, seed: int = 42) -> pd.DataFrame:
    df = generate_dataset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path=DATA_PATH, regenerate_if_missing: bool = True) -> pd.DataFrame:
    """Load the dataset from disk, regenerating it from sklearn if absent."""
    if not path.exists():
        if not regenerate_if_missing:
            raise FileNotFoundError(
                f"{path} not found. Run `python -m ml.data` or pass "
                "regenerate_if_missing=True to rebuild it from scikit-learn."
            )
        return save_dataset(path)
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop export artifacts / identifiers and encode the target as 0/1."""
    df = df.copy()
    drop_cols = [c for c in ("Unnamed: 32", "id") if c in df.columns]
    df = df.drop(columns=drop_cols)
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"M": 1, "B": 0}).astype(int)
    return df


if __name__ == "__main__":
    frame = save_dataset()
    print(f"Wrote {len(frame)} rows x {frame.shape[1]} columns to {DATA_PATH}")
