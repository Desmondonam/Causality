from pydantic import BaseModel, Field


class FeatureRange(BaseModel):
    min: float
    max: float
    median: float


class FeatureInfo(BaseModel):
    name: str
    label: str
    range: FeatureRange


class ModelInfo(BaseModel):
    best_model: str
    n_features: int
    top_features: list[str]
    metrics: dict[str, dict[str, float]]


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Mapping of feature name -> value, e.g. {'radius_worst': 18.2, ...}"
    )


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = malignant, 0 = benign")
    label: str
    probability_malignant: float
    risk_level: str
    model_used: str
    top_contributions: list[FeatureContribution]


class CausalEdge(BaseModel):
    source: str
    target: str


class CausalEffect(BaseModel):
    treatment: str
    standardized_ate: float
    adjustment_set: str
    robust_to_placebo: bool


class CausalGraphResponse(BaseModel):
    nodes: list[str]
    edges: list[CausalEdge]
    effects: list[CausalEffect]


class FeatureImportanceItem(BaseModel):
    feature: str
    causal_score: float | None = None
    shap_importance: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    best_model: str | None = None
