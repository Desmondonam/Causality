# Causality backend

FastAPI service that serves the trained causal ML pipeline (`../ml`) to the React frontend.

## Endpoints

| Method | Path                    | Description                                            |
| ------ | ------------------------ | -------------------------------------------------------- |
| GET    | `/health`                | Liveness + whether model artifacts are loaded            |
| GET    | `/api/model-info`        | Best model name, feature list, evaluation metrics        |
| GET    | `/api/features`          | Feature names/labels/ranges, for building an input form  |
| POST   | `/api/predict`           | `{"features": {...}}` -> prediction + SHAP contributions |
| GET    | `/api/causal-graph`      | DAG nodes/edges + DoWhy backdoor-adjusted effects         |
| GET    | `/api/feature-importance`| Causal score vs. SHAP importance per feature              |

Interactive docs at `/docs` once running.

## Local development

```bash
# from the repo root
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate on macOS/Linux
pip install -r requirements.txt -r backend/requirements.txt
python -m ml.pipeline                          # trains models, writes ../models and ../outputs/reports

cp backend/.env.example backend/.env           # adjust CORS origins if needed
uvicorn app.main:app --reload --app-dir backend
```

The API reads model artifacts from `../models` and report CSVs from `../outputs/reports` (both produced by
`python -m ml.pipeline`); it does not train anything itself.

## Tests

```bash
pytest backend/tests -v
```

## Docker

```bash
# from the repo root, after `python -m ml.pipeline` has produced ../models
docker build -f backend/Dockerfile -t causality-backend .
docker run -p 8000:8000 --env CAUSALITY_CORS_ORIGINS=http://localhost:3000 causality-backend
```

## Deployment

Any container host works (Render, Railway, Fly.io, Azure Container Apps, ...). The general recipe:

1. Run `python -m ml.pipeline` (locally or in CI) so `models/` and `outputs/reports/` are populated.
2. Build the image from `backend/Dockerfile` (build context = repo root, since it copies `ml/`, `models/`
   and `outputs/reports/` alongside `backend/app/`).
3. Set `CAUSALITY_CORS_ORIGINS` to your deployed frontend's origin(s).
4. Point the frontend's `NEXT_PUBLIC_API_URL` at the deployed backend URL.
