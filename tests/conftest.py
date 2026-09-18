import pytest

from ml.config import TARGET
from ml.data import clean_dataset, load_dataset
from ml.features import compute_feature_scores, select_top_features
from ml.modeling import split_data, train_and_evaluate
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="session")
def clean_df():
    return clean_dataset(load_dataset())


@pytest.fixture(scope="session")
def trained_pipeline(clean_df):
    """Train once per test session and share the result across tests that
    only need to assert on it - the full model sweep is not cheap."""
    X, y = clean_df.drop(columns=[TARGET]), clean_df[TARGET]
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = StandardScaler().fit(X_train)
    feature_scores = compute_feature_scores(X_train, y_train, scaler.transform(X_train))
    top_features = select_top_features(feature_scores)

    results, top_scaler = train_and_evaluate(X_train[top_features], X_test[top_features], y_train, y_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_scores": feature_scores,
        "top_features": top_features,
        "results": results,
        "scaler": top_scaler,
    }
