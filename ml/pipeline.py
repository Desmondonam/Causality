"""End-to-end orchestration: data -> features -> causal graph -> models ->
SHAP -> saved artifacts.

Run with:

    python -m ml.pipeline

Produces everything the Streamlit app (``app.py``) and the FastAPI backend
(``backend/``) need at inference time, plus the CSV/PNG report artifacts
referenced by the README and the notebooks.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from ml.causal_graph import build_causal_graph, draw_causal_graph, estimate_causal_effects, results_to_frame
from ml.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TARGET
from ml.data import clean_dataset, load_dataset
from ml.features import compute_feature_scores, select_top_features
from ml.interpret import compute_shap_importance
from ml.modeling import best_model_name, split_data, train_and_evaluate
from ml.visuals import plot_feature_importance, plot_model_comparison
from sklearn.preprocessing import StandardScaler


def run(verbose: bool = True) -> dict:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def log(msg: str):
        if verbose:
            print(msg)

    log("=" * 80)
    log("1. LOADING DATA")
    log("=" * 80)
    df = clean_dataset(load_dataset())
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    log(f"{X.shape[0]} samples, {X.shape[1]} features, "
        f"{y.sum()} malignant / {(y == 0).sum()} benign")

    X_train, X_test, y_train, y_test = split_data(X, y)
    full_scaler = StandardScaler().fit(X_train)
    X_train_scaled = full_scaler.transform(X_train)

    log("\n" + "=" * 80)
    log("2. CAUSAL FEATURE SELECTION (statistical screening)")
    log("=" * 80)
    feature_scores = compute_feature_scores(X_train, y_train, X_train_scaled)
    top_features = select_top_features(feature_scores)
    feature_scores.to_csv(REPORTS_DIR / "feature_scores.csv")
    log(f"Selected top {len(top_features)} features:\n  " + "\n  ".join(top_features))

    log("\n" + "=" * 80)
    log("3. CAUSAL GRAPH (Pearl do-calculus via DoWhy)")
    log("=" * 80)
    dag = build_causal_graph()
    draw_causal_graph(dag, FIGURES_DIR / "causal_graph.png")
    effects = estimate_causal_effects(df, dag)
    effects_frame = results_to_frame(effects)
    effects_frame.to_csv(REPORTS_DIR / "causal_effects.csv", index=False)
    log(effects_frame.to_string(index=False))

    log("\n" + "=" * 80)
    log("4. MODEL TRAINING")
    log("=" * 80)
    X_train_top, X_test_top = X_train[top_features], X_test[top_features]
    results, top_scaler = train_and_evaluate(X_train_top, X_test_top, y_train, y_test)
    for name, r in results.items():
        log(f"{name:22s} acc={r.accuracy:.4f} roc_auc={r.roc_auc:.4f} "
            f"cv_auc={r.cv_mean:.4f}(+/-{r.cv_std:.4f})")

    best_name = best_model_name(results)
    best_model = results[best_name].model
    log(f"\nBest model: {best_name} (ROC-AUC={results[best_name].roc_auc:.4f})")

    log("\n" + "=" * 80)
    log("5. SHAP INTERPRETATION")
    log("=" * 80)
    shap_source = results["Random Forest"].model
    shap_importance, _ = compute_shap_importance(shap_source, X_test_top)
    shap_importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    log(shap_importance.head(10).to_string(index=False))

    log("\n" + "=" * 80)
    log("6. VISUALIZATIONS")
    log("=" * 80)
    plot_model_comparison(results, y_test, FIGURES_DIR / "model_performance.png")
    plot_feature_importance(feature_scores, shap_importance, FIGURES_DIR / "feature_importance.png")
    log(f"Saved figures to {FIGURES_DIR}")

    log("\n" + "=" * 80)
    log("7. SAVING ARTIFACTS")
    log("=" * 80)
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    joblib.dump(top_scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(top_features, MODELS_DIR / "top_features.pkl")
    # Saved separately (even when it isn't the best predictive model) so the
    # backend can always produce SHAP TreeExplainer explanations.
    joblib.dump(shap_source, MODELS_DIR / "shap_model.pkl")

    meta = {
        "best_model": best_name,
        "n_features": len(top_features),
        "top_features": top_features,
        "metrics": {
            name: {k: v for k, v in r.as_dict().items() if k != "Model"} for name, r in results.items()
        },
        "feature_ranges": {
            col: {"min": float(X[col].min()), "max": float(X[col].max()), "median": float(X[col].median())}
            for col in top_features
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved model, scaler, top_features and metadata.json to {MODELS_DIR}")

    model_comparison = pd.DataFrame([r.as_dict() for r in results.values()])
    model_comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    report = _build_text_report(df, X, y, feature_scores, results, best_name, shap_importance, effects_frame)
    (REPORTS_DIR / "causal_analysis_report.txt").write_text(report, encoding="utf-8")
    log(f"Saved text report to {REPORTS_DIR / 'causal_analysis_report.txt'}")

    log("\nPIPELINE COMPLETE")
    return {"results": results, "best_model_name": best_name, "meta": meta}


def _build_text_report(df, X, y, feature_scores, results, best_name, shap_importance, effects_frame) -> str:
    lines = [
        "BREAST CANCER CAUSAL MACHINE LEARNING ANALYSIS",
        "=" * 80,
        "",
        "1. DATASET SUMMARY",
        f"   - Total Samples: {len(df)}",
        f"   - Features: {X.shape[1]}",
        f"   - Malignant Cases: {int(y.sum())} ({y.mean() * 100:.1f}%)",
        f"   - Benign Cases: {int((y == 0).sum())} ({(1 - y.mean()) * 100:.1f}%)",
        "",
        "2. STATISTICAL FEATURE SELECTION",
        "   Top 5 features by composite causal score:",
    ]
    for i, (idx, row) in enumerate(feature_scores.head(5).iterrows(), 1):
        lines.append(f"     {i}. {idx}: {row['Causal_Score']:.4f}")

    lines += ["", "3. CAUSAL EFFECT ESTIMATION (Pearl backdoor adjustment via DoWhy)"]
    for _, row in effects_frame.iterrows():
        lines.append(
            f"     {row['treatment']}: standardized ATE={row['standardized_ATE']:.4f}, "
            f"adjusted for [{row['adjustment_set']}], robust_to_placebo={row['robust_to_placebo']}"
        )

    lines += ["", "4. MODEL PERFORMANCE"]
    for name, r in results.items():
        lines.append(
            f"   {name}: accuracy={r.accuracy:.4f} precision={r.precision:.4f} "
            f"recall={r.recall:.4f} f1={r.f1:.4f} roc_auc={r.roc_auc:.4f} "
            f"cv_auc={r.cv_mean:.4f}(+/-{r.cv_std:.4f})"
        )

    lines += ["", f"5. BEST MODEL: {best_name} (ROC-AUC={results[best_name].roc_auc:.4f})"]

    lines += ["", "6. SHAP INTERPRETATION (top 5)"]
    for _, row in shap_importance.head(5).iterrows():
        lines.append(f"     {row['Feature']}: {row['SHAP_Importance']:.4f}")

    lines += ["", "=" * 80, "Analysis completed successfully."]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
