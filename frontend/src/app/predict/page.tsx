"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { api, type FeatureInfo, type PredictionResponse } from "@/lib/api";
import { RiskGauge } from "@/components/RiskGauge";

export default function PredictPage() {
  const [features, setFeatures] = useState<FeatureInfo[] | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .features()
      .then((fs) => {
        setFeatures(fs);
        const initial: Record<string, number> = {};
        fs.forEach((f) => (initial[f.name] = f.range.median));
        setValues(initial);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict(values);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const resetToMedians = () => {
    if (!features) return;
    const initial: Record<string, number> = {};
    features.forEach((f) => (initial[f.name] = f.range.median));
    setValues(initial);
    setResult(null);
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold tracking-tight">Prediction tool</h1>
      <p className="mt-2 max-w-2xl text-muted">
        Enter cell nucleus measurements from a digitized fine needle aspirate to estimate malignancy risk.
        Sliders are pre-filled with dataset medians &mdash; this is a research/education demo, not a
        diagnostic device.
      </p>

      {error && (
        <div className="mt-6 flex items-center gap-2 rounded-xl border border-malignant/30 bg-malignant/10 px-4 py-3 text-sm text-malignant">
          <AlertTriangle size={16} /> {error}
          {error.includes("fetch") && (
            <span className="text-muted"> &mdash; is the backend running at NEXT_PUBLIC_API_URL?</span>
          )}
        </div>
      )}

      <div className="mt-8 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
        <form onSubmit={handleSubmit} className="space-y-4 rounded-2xl border border-border bg-surface p-6">
          {!features && !error && (
            <div className="flex items-center gap-2 text-muted">
              <Loader2 className="animate-spin" size={16} /> Loading feature ranges...
            </div>
          )}
          {features?.map((f) => (
            <FeatureSlider
              key={f.name}
              feature={f}
              value={values[f.name] ?? f.range.median}
              onChange={(v) => setValues((prev) => ({ ...prev, [f.name]: v }))}
            />
          ))}
          {features && (
            <div className="flex gap-3 pt-2">
              <button
                type="submit"
                disabled={loading}
                className="flex items-center gap-2 rounded-full bg-brand px-6 py-2.5 text-sm font-semibold text-brand-foreground disabled:opacity-60"
              >
                {loading && <Loader2 className="animate-spin" size={15} />}
                Predict
              </button>
              <button
                type="button"
                onClick={resetToMedians}
                className="rounded-full border border-border px-6 py-2.5 text-sm font-semibold hover:bg-background"
              >
                Reset to medians
              </button>
            </div>
          )}
        </form>

        <div className="space-y-4">
          <div className="rounded-2xl border border-border bg-surface p-6">
            {result ? (
              <>
                <RiskGauge probabilityMalignant={result.probability_malignant} />
                <div className="mt-3 text-center">
                  <span
                    className={
                      "rounded-full px-3 py-1 text-sm font-semibold " +
                      (result.prediction === 1
                        ? "bg-malignant/15 text-malignant"
                        : "bg-benign/15 text-benign")
                    }
                  >
                    {result.label.toUpperCase()}
                  </span>
                  <p className="mt-2 text-xs text-muted">model: {result.model_used}</p>
                </div>
              </>
            ) : (
              <p className="py-10 text-center text-sm text-muted">
                Submit the form to see a prediction and risk gauge here.
              </p>
            )}
          </div>

          {result && result.top_contributions.length > 0 && (
            <div className="rounded-2xl border border-border bg-surface p-6">
              <h3 className="text-sm font-semibold">Top feature contributions</h3>
              <p className="mt-1 text-xs text-muted">SHAP values from a Random Forest surrogate explainer</p>
              <ul className="mt-3 space-y-2">
                {result.top_contributions.map((c) => (
                  <ContributionBar key={c.feature} contribution={c} />
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FeatureSlider({
  feature,
  value,
  onChange,
}: {
  feature: FeatureInfo;
  value: number;
  onChange: (v: number) => void;
}) {
  const { min, max } = feature.range;
  const step = useMemo(() => Math.max((max - min) / 200, 0.0001), [min, max]);

  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <label className="font-medium">{feature.label}</label>
        <span className="tabular-nums text-muted">{value.toFixed(3)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1.5 w-full accent-[var(--brand)]"
      />
    </div>
  );
}

function ContributionBar({ contribution }: { contribution: { feature: string; shap_contribution: number } }) {
  const magnitude = Math.min(Math.abs(contribution.shap_contribution) * 100, 100);
  const positive = contribution.shap_contribution >= 0;
  return (
    <li>
      <div className="flex justify-between text-xs">
        <span>{contribution.feature}</span>
        <span className={positive ? "text-malignant" : "text-benign"}>
          {positive ? "+" : ""}
          {contribution.shap_contribution.toFixed(3)}
        </span>
      </div>
      <div className="mt-1 h-1.5 w-full rounded-full bg-background">
        <div
          className={"h-1.5 rounded-full " + (positive ? "bg-malignant" : "bg-benign")}
          style={{ width: `${magnitude}%` }}
        />
      </div>
    </li>
  );
}
