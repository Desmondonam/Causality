"""FastAPI application exposing the trained causal ML pipeline to the React
frontend. Run locally with:

    uvicorn app.main:app --reload --app-dir backend

or via Docker (see backend/Dockerfile).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_service import ArtifactsNotFoundError, ModelService, get_model_service
from app.schemas import (
    CausalGraphResponse,
    FeatureImportanceItem,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Causal machine learning API for breast cancer diagnosis prediction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> ModelService:
    try:
        return get_model_service()
    except ArtifactsNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        service = get_model_service()
        return HealthResponse(status="ok", model_loaded=service.loaded, best_model=service.metadata.get("best_model"))
    except ArtifactsNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False, best_model=None)


@app.get("/api/model-info", response_model=ModelInfo, tags=["model"])
def model_info(service: ModelService = Depends(get_service)):
    return service.model_info()


@app.get("/api/features", tags=["model"])
def features(service: ModelService = Depends(get_service)):
    return service.feature_info()


@app.post("/api/predict", response_model=PredictionResponse, tags=["model"])
def predict(request: PredictionRequest, service: ModelService = Depends(get_service)):
    try:
        return service.predict(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/causal-graph", response_model=CausalGraphResponse, tags=["causal"])
def causal_graph(service: ModelService = Depends(get_service)):
    return service.causal_report()


@app.get("/api/feature-importance", response_model=list[FeatureImportanceItem], tags=["causal"])
def feature_importance(service: ModelService = Depends(get_service)):
    return service.feature_importance()
