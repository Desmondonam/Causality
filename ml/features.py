"""Statistical feature selection used as a screening step before causal
graph construction.

None of these methods alone establishes causality — they establish
*association*. We combine four complementary statistical signals into a
single "causal score" used only to narrow 30 raw features down to a
tractable shortlist; the actual causal claims are made later, in
``ml.causal_graph``, via an explicit DAG and Pearl's backdoor-adjustment
criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression

from ml.config import N_TOP_FEATURES, RANDOM_STATE


def _minmax_norm(s: pd.Series) -> pd.Series:
    span = s.max() - s.min()
    if span == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / span


def compute_feature_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_scaled: np.ndarray,
) -> pd.DataFrame:
    """Compute ANOVA F, mutual information, RF importance and |LR coef|,
    and combine them into a single normalized composite ``Causal_Score``.
    """
    scores = pd.DataFrame(index=X_train.columns)

    f_selector = SelectKBest(score_func=f_classif, k="all").fit(X_train, y_train)
    scores["F_Score"] = f_selector.scores_
    scores["F_PValue"] = f_selector.pvalues_

    scores["MI_Score"] = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train_scaled, y_train)
    scores["RF_Importance"] = rf.feature_importances_

    lr = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    scores["LR_Coefficient"] = np.abs(lr.coef_[0])

    for col in ("F_Score", "MI_Score", "RF_Importance", "LR_Coefficient"):
        scores[f"{col}_Norm"] = _minmax_norm(scores[col])

    scores["Causal_Score"] = scores[
        ["F_Score_Norm", "MI_Score_Norm", "RF_Importance_Norm", "LR_Coefficient_Norm"]
    ].mean(axis=1)

    return scores.sort_values("Causal_Score", ascending=False)


def select_top_features(scores: pd.DataFrame, n: int = N_TOP_FEATURES) -> list[str]:
    return scores.head(n).index.tolist()
