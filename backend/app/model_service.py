"""Loads the artifacts produced by `python -m ml.pipeline` (model, scaler,
feature list, metadata, causal-graph/SHAP reports) once at process startup
and serves predictions + explanations from memory."""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.config import settings


class ArtifactsNotFoundError(RuntimeError):
    pass


class ModelService:
    def __init__(self):
        self.models_dir = settings.models_dir
        self.reports_dir = settings.reports_dir

        self.model = None
        self.shap_model = None
        self.scaler = None
        self.top_features: list[str] = []
        self.metadata: dict = {}
        self._shap_explainer = None

        self._load()

    def _load(self):
        required = ["model.pkl", "scaler.pkl", "top_features.pkl", "metadata.json"]
        missing = [f for f in required if not (self.models_dir / f).exists()]
        if missing:
            raise ArtifactsNotFoundError(
                f"Missing model artifacts {missing} in {self.models_dir}. "
                "Run `python -m ml.pipeline` from the repo root first."
            )

        self.model = joblib.load(self.models_dir / "model.pkl")
        self.scaler = joblib.load(self.models_dir / "scaler.pkl")
        self.top_features = joblib.load(self.models_dir / "top_features.pkl")
        with open(self.models_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shap_path = self.models_dir / "shap_model.pkl"
        if shap_path.exists():
            self.shap_model = joblib.load(shap_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _needs_scaling(self) -> bool:
        # LogisticRegression / CalibratedClassifierCV(SVC) were fit on scaled
        # features; tree ensembles were fit on raw features. Detect via class name.
        return type(self.model).__name__ in {"LogisticRegression", "CalibratedClassifierCV"}

    def _to_frame(self, features: dict[str, float]) -> pd.DataFrame:
        missing = [f for f in self.top_features if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return pd.DataFrame([[features[f] for f in self.top_features]], columns=self.top_features)

    def predict(self, features: dict[str, float]) -> dict:
        X = self._to_frame(features)
        X_input = self.scaler.transform(X) if self._needs_scaling() else X

        proba_malignant = float(self.model.predict_proba(X_input)[0, 1])
        prediction = int(proba_malignant >= 0.5)

        if proba_malignant < 0.3:
            risk = "low"
        elif proba_malignant < 0.7:
            risk = "moderate"
        else:
            risk = "high"

        contributions = self._top_contributions(X)

        return {
            "prediction": prediction,
            "label": "malignant" if prediction == 1 else "benign",
            "probability_malignant": proba_malignant,
            "risk_level": risk,
            "model_used": self.metadata.get("best_model", type(self.model).__name__),
            "top_contributions": contributions,
        }

    def _top_contributions(self, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
        if self.shap_model is None:
            return []
        import shap

        explainer = self._get_shap_explainer()
        raw = explainer.shap_values(X)
        if isinstance(raw, list):
            values = np.array(raw[1])[0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            values = raw[0, :, 1]
        else:
            values = np.array(raw)[0]

        order = np.argsort(-np.abs(values))[:top_n]
        return [
            {
                "feature": self.top_features[i],
                "value": float(X.iloc[0, i]),
                "shap_contribution": float(values[i]),
            }
            for i in order
        ]

    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            import shap

            self._shap_explainer = shap.TreeExplainer(self.shap_model)
        return self._shap_explainer

    def feature_info(self) -> list[dict]:
        ranges = self.metadata.get("feature_ranges", {})
        return [
            {
                "name": f,
                "label": f.replace("_", " ").title(),
                "range": ranges.get(f, {"min": 0.0, "max": 1.0, "median": 0.5}),
            }
            for f in self.top_features
        ]

    def model_info(self) -> dict:
        return {
            "best_model": self.metadata.get("best_model", "unknown"),
            "n_features": self.metadata.get("n_features", len(self.top_features)),
            "top_features": self.top_features,
            "metrics": self.metadata.get("metrics", {}),
        }

    def causal_report(self) -> dict:
        from ml.causal_graph import build_causal_graph

        graph = build_causal_graph()
        nodes = list(graph.nodes)
        edges = [{"source": s, "target": t} for s, t in graph.edges]

        effects_path = self.reports_dir / "causal_effects.csv"
        effects = []
        if effects_path.exists():
            df = pd.read_csv(effects_path)
            for _, row in df.iterrows():
                effects.append(
                    {
                        "treatment": row["treatment"],
                        "standardized_ate": float(row["standardized_ATE"]),
                        "adjustment_set": row["adjustment_set"],
                        "robust_to_placebo": bool(row["robust_to_placebo"]),
                    }
                )
        return {"nodes": nodes, "edges": edges, "effects": effects}

    def feature_importance(self) -> list[dict]:
        causal_path = self.reports_dir / "feature_scores.csv"
        shap_path = self.reports_dir / "shap_importance.csv"

        causal_scores = pd.read_csv(causal_path, index_col=0) if causal_path.exists() else None
        shap_scores = pd.read_csv(shap_path) if shap_path.exists() else None

        features = set(self.top_features)
        if causal_scores is not None:
            features |= set(causal_scores.index)
        if shap_scores is not None:
            features |= set(shap_scores["Feature"])

        items = []
        for f in features:
            causal_score = None
            if causal_scores is not None and f in causal_scores.index:
                causal_score = float(causal_scores.loc[f, "Causal_Score"])
            shap_importance = None
            if shap_scores is not None and f in set(shap_scores["Feature"]):
                shap_importance = float(shap_scores.loc[shap_scores["Feature"] == f, "SHAP_Importance"].iloc[0])
            items.append({"feature": f, "causal_score": causal_score, "shap_importance": shap_importance})

        items.sort(key=lambda x: x["shap_importance"] or 0, reverse=True)
        return items


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
