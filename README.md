# Causality

**Causal machine learning for breast cancer diagnosis.** This project goes beyond "which features predict
malignancy" and asks "which features *cause* it, under an explicit, falsifiable causal model" — implemented
with Judea Pearl's do-calculus (via [DoWhy](https://www.pywhy.org/dowhy/)), a domain-informed causal graph,
interpretable predictive models, and SHAP explanations, wrapped in a full-stack app (FastAPI + React) and a
lightweight Streamlit companion demo.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Why "causal" and not just "predictive"

Most ML feature-importance tools (ANOVA scores, mutual information, SHAP) measure **association** — rung 1
of Pearl's [Ladder of Causation](https://en.wikipedia.org/wiki/Ladder_of_causation). They can't tell you
whether a feature is a cause of the outcome, a symptom of a shared cause, or a proxy for something else
entirely. This project makes that distinction explicit:

1. **Statistical screening** narrows 30 raw cytology measurements to 15 candidates (association only).
2. **A causal graph**, built from domain knowledge about tumor cytology, is handed to DoWhy, which
   identifies a valid *backdoor adjustment set* per treatment and estimates its causal effect on diagnosis —
   then stress-tests that estimate with refutation checks (does it survive a random common cause? does it
   collapse under a placebo/permuted treatment?).
3. **Predictive modeling + SHAP** trains and compares four classifiers and explains their predictions — a
   different, complementary kind of insight from the causal estimates in step 2.

See [`ml/causal_graph.py`](ml/causal_graph.py) and
[`notebooks/02_Causal_Inference_and_Modeling.ipynb`](notebooks/02_Causal_Inference_and_Modeling.ipynb) for
the full reasoning, or the **Methodology** page in the deployed app.

## Architecture

```
                     ┌─────────────────────┐
                     │   ml/ (Python)       │
                     │  data → features →   │
                     │  causal_graph →      │
                     │  modeling → interpret│
                     └──────────┬───────────┘
                                │ python -m ml.pipeline
                                ▼
                  models/ + outputs/reports/ + outputs/figures/
                     │                              │
     ┌───────────────┴───────────────┐              │
     ▼                                ▼              ▼
┌─────────────┐              ┌───────────────┐  notebooks/*.ipynb
│ backend/     │◄────REST────┤ frontend/      │  (narrated walkthrough)
│ FastAPI      │              │ Next.js+React  │
│ (Render/Fly) │              │ (Vercel)       │
└─────────────┘              └───────────────┘

              ┌─────────────────────────────────┐
              │ streamlit_app.py — Streamlit demo │  (reads models/ + outputs/ directly,
              │ (Streamlit Community Cloud)       │   no backend needed)
              └─────────────────────────────────┘
```

| Layer | Tech | Why |
|---|---|---|
| Causal inference | [DoWhy](https://www.pywhy.org/dowhy/), [NetworkX](https://networkx.org/) | Pearl's graphical do-calculus: identification + estimation + refutation, without a native Graphviz dependency |
| ML / stats | scikit-learn, statsmodels, SHAP | Feature screening, modeling, interpretability |
| API | FastAPI, Pydantic v2, Uvicorn | Typed, async, auto-documented (`/docs`) |
| Frontend | Next.js 16 (App Router), TypeScript, Tailwind CSS v4, Recharts | Modern React stack, deploys natively to Vercel |
| Secondary UI | Streamlit | Zero-frontend-code demo for a data-science audience |
| Data versioning | DVC (+ regeneration from `sklearn.datasets`) | Reproducible without requiring remote storage credentials |
| CI/CD | GitHub Actions (backend + frontend workflows) | Pipeline + tests run on every push; Docker image build verified |
| Containerization | Docker, docker-compose | Backend deploys anywhere containers run |

## Repository layout

```
ml/                   Reusable causal ML pipeline (the actual "engine")
  data.py             Load/regenerate the dataset (sklearn.datasets, Kaggle-schema CSV)
  features.py         Statistical "causal score" feature screening
  causal_graph.py      Domain DAG + DoWhy backdoor estimation + refutation
  modeling.py          Multi-model training/evaluation
  interpret.py          SHAP importance
  visuals.py             Matplotlib report figures
  pipeline.py             Orchestrates the above, saves all artifacts
  run_notebooks.py          Executes notebooks/*.ipynb in place (CI + local)
notebooks/            Narrated EDA + causal-inference/modeling walkthrough
backend/              FastAPI service serving trained artifacts to the frontend
frontend/             Next.js + TypeScript + Tailwind app (deploy target: Vercel)
tests/                pytest suite for ml/
streamlit_app.py      Streamlit companion demo
data/, models/, outputs/   Generated (gitignored) - see "Reproducing" below
```

## Reproducing the analysis

```bash
python -m venv venv && venv\Scripts\activate        # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

python -m ml.pipeline          # data → causal graph → models → SHAP → models/ + outputs/
python -m ml.run_notebooks     # (optional) re-executes notebooks/*.ipynb in place
pytest tests/ -v
```

`python -m ml.pipeline` is fully deterministic and self-contained: it regenerates `data/data.csv` from
`sklearn.datasets.load_breast_cancer` (reshaped to the original Kaggle column schema) if it isn't already
present, so nothing here depends on the DVC remote being reachable. If you do have access to the original
`storage` DVC remote, `dvc pull` will fetch the authoritative CSV instead.

### Results (current run)

| Model | Accuracy | ROC-AUC | F1-Score |
|---|---|---|---|
| SVM (calibrated) | 98.2% | 99.6% | 97.6% |
| Logistic Regression | 97.4% | 99.7% | 96.4% |
| Random Forest | 97.4% | 99.6% | 96.3% |
| **Gradient Boosting** (best ROC-AUC) | 95.6% | **99.8%** | 93.7% |

All five backdoor-adjusted causal-effect estimates (`outputs/reports/causal_effects.csv`) survived placebo
refutation in the current run — see the Insights page or `causal_analysis_report.txt` for details.

## Running the full stack locally

```bash
# 1. Train and export artifacts (see above)
python -m ml.pipeline

# 2. Backend
pip install -r backend/requirements.txt
uvicorn app.main:app --reload --app-dir backend    # http://localhost:8000, docs at /docs

# 3. Frontend (new terminal)
cd frontend
npm install
cp .env.example .env.local                         # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                                         # http://localhost:3000

# 4. (optional) Streamlit companion, no backend required
streamlit run streamlit_app.py
```

Or with Docker Compose (backend + frontend, after step 1):

```bash
docker compose up --build
```

## Deployment

| Component | Target | Notes |
|---|---|---|
| Frontend | **Vercel** | Root directory = `frontend`; set `NEXT_PUBLIC_API_URL` to the deployed backend URL. See [`frontend/README.md`](frontend/README.md). |
| Backend | Any container host (Render, Fly.io, Railway, ...) | Build `backend/Dockerfile` from the repo root (it needs `ml/`, `models/`, `outputs/reports/`). See [`backend/README.md`](backend/README.md). |
| Streamlit demo | Streamlit Community Cloud | Point at `streamlit_app.py`; run `python -m ml.pipeline` in a pre-deploy step or commit the artifacts if your host doesn't support one. |

CI (`.github/workflows/backend-ci.yml`, `frontend-ci.yml`) runs the pipeline, the full test suite, and a
Docker build on every push, so a red build means the deployable artifact is actually broken.

## Notebooks

- [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb) — class balance, summary statistics, correlation
  structure, effect sizes.
- [`notebooks/02_Causal_Inference_and_Modeling.ipynb`](notebooks/02_Causal_Inference_and_Modeling.ipynb) —
  statistical screening, the causal graph, DoWhy identification/estimation/refutation, model training, SHAP.

Both import from `ml/` rather than duplicating logic, and are re-executed by `python -m ml.run_notebooks`
(and in CI) so they never drift from the code.

## License

[MIT](LICENSE)

## Disclaimer

This project is for education and research into causal machine learning methodology. It is not a validated
medical device and must not be used for real diagnostic decisions.

## Author

**Desmond Onam** — [@Desmondonam](https://github.com/Desmondonam) ·
[LinkedIn](https://www.linkedin.com/in/desmond-onam-b64702175/)
