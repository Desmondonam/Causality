from ml.config import N_TOP_FEATURES
from ml.features import select_top_features


def test_feature_scores_columns(trained_pipeline):
    scores = trained_pipeline["feature_scores"]
    for col in ("F_Score", "MI_Score", "RF_Importance", "LR_Coefficient", "Causal_Score"):
        assert col in scores.columns
    # Sorted descending by the composite score.
    assert scores["Causal_Score"].is_monotonic_decreasing


def test_select_top_features_length(trained_pipeline):
    top = select_top_features(trained_pipeline["feature_scores"])
    assert len(top) == N_TOP_FEATURES
    assert len(set(top)) == N_TOP_FEATURES  # no duplicates
