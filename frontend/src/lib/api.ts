export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface FeatureRange {
  min: number;
  max: number;
  median: number;
}

export interface FeatureInfo {
  name: string;
  label: string;
  range: FeatureRange;
}

export interface ModelInfo {
  best_model: string;
  n_features: number;
  top_features: string[];
  metrics: Record<string, Record<string, number>>;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  shap_contribution: number;
}

export interface PredictionResponse {
  prediction: number;
  label: string;
  probability_malignant: number;
  risk_level: "low" | "moderate" | "high";
  model_used: string;
  top_contributions: FeatureContribution[];
}

export interface CausalEdge {
  source: string;
  target: string;
}

export interface CausalEffect {
  treatment: string;
  standardized_ate: number;
  adjustment_set: string;
  robust_to_placebo: boolean;
}

export interface CausalGraphResponse {
  nodes: string[];
  edges: CausalEdge[];
  effects: CausalEffect[];
}

export interface FeatureImportanceItem {
  feature: string;
  causal_score: number | null;
  shap_importance: number | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  best_model: string | null;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/health"),
  modelInfo: () => apiFetch<ModelInfo>("/api/model-info"),
  features: () => apiFetch<FeatureInfo[]>("/api/features"),
  predict: (features: Record<string, number>) =>
    apiFetch<PredictionResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),
  causalGraph: () => apiFetch<CausalGraphResponse>("/api/causal-graph"),
  featureImportance: () => apiFetch<FeatureImportanceItem[]>("/api/feature-importance"),
};
