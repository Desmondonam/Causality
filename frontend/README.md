# Causality frontend

Next.js (App Router) + TypeScript + Tailwind CSS v4 + Recharts frontend for the Causality project. It talks
to the FastAPI backend in `../backend` over a REST API; see the root [README](../README.md) for the full
architecture.

## Pages

- `/` — landing page with live headline model metrics
- `/predict` — interactive prediction tool (calls `POST /api/predict`)
- `/insights` — model comparison, feature importance, causal graph and causal-effect charts
- `/methodology` — narrative explanation of the causal-inference approach

## Local development

```bash
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL to your backend
npm run dev
```

Requires the backend to be running (see `../backend/README.md`) and `../models` / `../outputs/reports` to
be populated by `python -m ml.pipeline` from the repo root.

## Deployment (Vercel)

1. Import this repository into Vercel and set the project **Root Directory** to `frontend`.
2. Set the environment variable `NEXT_PUBLIC_API_URL` to your deployed backend URL (e.g. a Render/Fly.io
   FastAPI service — see `../backend/README.md`).
3. Deploy. Vercel auto-detects Next.js; no extra build configuration is required.
