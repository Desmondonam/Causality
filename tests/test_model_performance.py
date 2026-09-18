import pytest

MIN_ACCURACY = 0.90
MIN_ROC_AUC = 0.95


def test_models_trained(trained_pipeline):
    results = trained_pipeline["results"]
    assert set(results) == {"Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"}


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"])
def test_model_accuracy_threshold(trained_pipeline, model_name):
    r = trained_pipeline["results"][model_name]
    assert r.accuracy >= MIN_ACCURACY, f"{model_name} accuracy {r.accuracy:.2f} below {MIN_ACCURACY}"


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"])
def test_model_roc_auc_threshold(trained_pipeline, model_name):
    r = trained_pipeline["results"][model_name]
    assert r.roc_auc >= MIN_ROC_AUC, f"{model_name} ROC-AUC {r.roc_auc:.2f} below {MIN_ROC_AUC}"
