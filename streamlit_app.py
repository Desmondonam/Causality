"""Streamlit demo app for the Causality project.

Unlike the React frontend (the primary, Vercel-deployed product), this app
talks directly to the artifacts produced by `python -m ml.pipeline` - no
separate backend needed, which makes it convenient to deploy to Streamlit
Community Cloud as a lightweight, data-scientist-facing companion demo.
Run `python -m ml.pipeline` at least once before launching this app.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TARGET
from ml.data import clean_dataset, load_dataset

st.set_page_config(
    page_title="Causality | Breast Cancer Causal ML",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #db2777; text-align: center; padding-top: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.4rem;
                   border-radius: 12px; color: white; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return clean_dataset(load_dataset())


@st.cache_resource
def get_model_artifacts():
    required = ["model.pkl", "scaler.pkl", "top_features.pkl", "metadata.json"]
    if any(not (MODELS_DIR / f).exists() for f in required):
        return None
    model = joblib.load(MODELS_DIR / "model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    top_features = joblib.load(MODELS_DIR / "top_features.pkl")
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, top_features, metadata


@st.cache_data
def get_report(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


df = get_data()
artifacts = get_model_artifacts()

st.markdown('<p class="main-header">🎗️ Causality: Breast Cancer Causal ML</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#64748b'>Statistical screening -> causal graph (Pearl/DoWhy) "
    "-> predictive modeling -> SHAP interpretation</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select page",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Analysis", "🔮 Prediction Tool", "🧭 Causal Graph", "📄 Report"],
    )
    st.markdown("---")
    st.info(
        "This is the lightweight Streamlit companion to the React app. "
        "Run `python -m ml.pipeline` first to populate real model artifacts."
    )
    if artifacts is None:
        st.warning("No trained model found. Run `python -m ml.pipeline` from the repo root.")

if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{len(df)}</h2><p>Total Samples</p></div>', unsafe_allow_html=True)
    with col2:
        best_acc = artifacts[3]["metrics"][artifacts[3]["best_model"]]["Accuracy"] * 100 if artifacts else 0
        st.markdown(f'<div class="metric-card"><h2>{best_acc:.1f}%</h2><p>Best Model Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        n_feat = artifacts[3]["n_features"] if artifacts else "-"
        st.markdown(f'<div class="metric-card"><h2>{n_feat}</h2><p>Causal Features Used</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Project overview")
        st.markdown(
            """
            This project applies **Pearl's causal framework**, not just correlation, to breast cancer
            diagnosis:

            - **Statistical screening**: ANOVA F-test, mutual information, Random Forest importance and
              logistic coefficients combine into a composite causal score.
            - **Causal graph**: a domain-informed DAG + DoWhy's backdoor-adjustment criterion estimate the
              causal effect of each "worst" measurement on malignancy, stress-tested with refutation checks.
            - **Modeling + SHAP**: four classifiers are compared and explained.
            """
        )
    with col2:
        st.markdown("### 🔬 Key findings")
        if artifacts:
            model, scaler, top_features, metadata = artifacts
            best = metadata["best_model"]
            st.success(f"**Best model:** {best}\n\nROC-AUC: {metadata['metrics'][best]['ROC-AUC']:.4f}")
            st.info(f"**Top causal features:**\n\n" + "\n".join(f"- {f}" for f in top_features[:5]))
        else:
            st.warning("Run `python -m ml.pipeline` to populate real results here.")

elif page == "📊 Data Explorer":
    st.header("📊 Data Explorer")
    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Statistics", "🔍 Distribution"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", df.shape[0])
        c2.metric("Total Features", df.shape[1] - 1)
        c3.metric("Malignant %", f"{df[TARGET].mean() * 100:.1f}%")

    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

    with tab3:
        counts = df[TARGET].value_counts().rename({1: "Malignant", 0: "Benign"})
        fig = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color=["#10b981", "#ef4444"])])
        fig.update_layout(title="Diagnosis Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Analysis":
    st.header("🤖 Model Analysis")
    if artifacts is None:
        st.warning("No trained model found. Run `python -m ml.pipeline` from the repo root.")
    else:
        model, scaler, top_features, metadata = artifacts
        comparison = pd.DataFrame(metadata["metrics"]).T.reset_index().rename(columns={"index": "Model"})

        tab1, tab2 = st.tabs(["🎯 Performance", "🔍 Feature Importance"])
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=comparison["Model"], y=comparison["Accuracy"], marker_color="#3b82f6"))
                fig.update_layout(title="Model Accuracy", height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure(go.Bar(x=comparison["Model"], y=comparison["ROC-AUC"], marker_color="#10b981"))
                fig.update_layout(title="Model ROC-AUC", height=400)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comparison, use_container_width=True)

        with tab2:
            shap_importance = get_report("shap_importance.csv")
            if shap_importance is not None:
                fig = go.Figure(
                    go.Bar(
                        x=shap_importance["SHAP_Importance"].head(10)[::-1],
                        y=shap_importance["Feature"].head(10)[::-1],
                        orientation="h",
                        marker_color="#8b5cf6",
                    )
                )
                fig.update_layout(title="Top 10 Features by SHAP Importance", height=500)
                st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Prediction Tool":
    st.header("🔮 Cancer Prediction Tool")
    st.markdown("Enter cell nucleus measurements to get a prediction.")

    if artifacts is None:
        st.warning("No trained model found. Run `python -m ml.pipeline` from the repo root.")
    else:
        model, scaler, top_features, metadata = artifacts
        ranges = metadata["feature_ranges"]

        cols = st.columns(3)
        inputs = {}
        for i, feature in enumerate(top_features):
            r = ranges[feature]
            with cols[i % 3]:
                inputs[feature] = st.slider(
                    feature.replace("_", " ").title(),
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=float(r["median"]),
                )

        if st.button("🔍 Predict", type="primary"):
            X = pd.DataFrame([inputs])[top_features]
            needs_scaling = type(model).__name__ in {"LogisticRegression", "CalibratedClassifierCV"}
            X_input = scaler.transform(X) if needs_scaling else X
            proba = float(model.predict_proba(X_input)[0, 1])
            prediction = int(proba >= 0.5)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.error("### ⚠️ MALIGNANT")
                    st.markdown(f"**Confidence:** {proba * 100:.1f}%")
                else:
                    st.success("### ✅ BENIGN")
                    st.markdown(f"**Confidence:** {(1 - proba) * 100:.1f}%")
                st.caption(f"Model used: {metadata['best_model']}")
            with c2:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,
                        title={"text": "Malignancy Risk"},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkred" if prediction == 1 else "green"},
                            "steps": [
                                {"range": [0, 30], "color": "lightgreen"},
                                {"range": [30, 70], "color": "yellow"},
                                {"range": [70, 100], "color": "lightcoral"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.info("⚕️ **Note:** This is a research/education tool and must not replace professional diagnosis.")

elif page == "🧭 Causal Graph":
    st.header("🧭 Causal Graph (Pearl / DoWhy)")
    st.markdown(
        "Domain-informed DAG over tumor cytology features, with backdoor-adjusted causal effect "
        "estimates and refutation tests. See the **Methodology** page of the React app, or "
        "`notebooks/02_Causal_Inference_and_Modeling.ipynb`, for the full rationale."
    )
    graph_path = FIGURES_DIR / "causal_graph.png"
    if graph_path.exists():
        st.image(str(graph_path), use_container_width=True)
    effects = get_report("causal_effects.csv")
    if effects is not None:
        st.dataframe(effects, use_container_width=True)
        fig = px.bar(
            effects.sort_values("standardized_ATE"),
            x="standardized_ATE",
            y="treatment",
            orientation="h",
            color="robust_to_placebo",
            title="Standardized backdoor-adjusted ATE on malignancy",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run `python -m ml.pipeline` to generate causal effect estimates.")

elif page == "📄 Report":
    st.header("📄 Analysis Report")
    report_path = REPORTS_DIR / "causal_analysis_report.txt"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        st.text(report_text)
        st.download_button("📥 Download report", data=report_text, file_name="causal_analysis_report.txt")
    else:
        st.warning("Run `python -m ml.pipeline` to generate the report.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b'>🎗️ Causality | Streamlit companion demo &mdash; "
    "see the React app for the primary product</p>",
    unsafe_allow_html=True,
)
