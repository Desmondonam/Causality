"""API tests. Assumes `python -m ml.pipeline` has already been run from the
repo root so model artifacts exist under ../models (the CI workflow does
this before running these tests)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import get_model_service

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _require_artifacts():
    try:
        get_model_service()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Model artifacts not available: {exc}")


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info():
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_features"] > 0
    assert len(body["top_features"]) == body["n_features"]


def test_features():
    resp = client.get("/api/features")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) > 0
    assert {"name", "label", "range"} <= set(body[0].keys())


def test_predict_with_median_values():
    features_resp = client.get("/api/features").json()
    payload = {"features": {f["name"]: f["range"]["median"] for f in features_resp}}

    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert body["risk_level"] in ("low", "moderate", "high")


def test_predict_missing_feature_returns_422():
    resp = client.post("/api/predict", json={"features": {}})
    assert resp.status_code == 422


def test_causal_graph():
    resp = client.get("/api/causal-graph")
    assert resp.status_code == 200
    body = resp.json()
    assert "diagnosis" in body["nodes"]
    assert len(body["edges"]) > 0


def test_feature_importance():
    resp = client.get("/api/feature-importance")
    assert resp.status_code == 200
    assert len(resp.json()) > 0
