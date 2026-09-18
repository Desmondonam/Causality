FROM python:3.12-slim

WORKDIR /srv

# System deps for scientific python wheels that occasionally need to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt

# The FastAPI app imports `ml.causal_graph` for the /api/causal-graph route,
# so the `ml` package comes along too. Model artifacts are baked in from
# `models/` and `outputs/reports/` (produced by `python -m ml.pipeline`).
COPY ml ./ml
COPY models ./models
COPY outputs/reports ./outputs/reports
COPY backend/app ./app

ENV PYTHONPATH=/srv
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
