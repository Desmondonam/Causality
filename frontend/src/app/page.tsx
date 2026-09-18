import Link from "next/link";
import { Activity, GitBranch, ShieldCheck, Sparkles } from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { api } from "@/lib/api";

export const revalidate = 300;

export default async function HomePage() {
  let modelInfo = null;
  try {
    modelInfo = await api.modelInfo();
  } catch {
    modelInfo = null;
  }

  const bestMetrics = modelInfo?.metrics?.[modelInfo.best_model];

  return (
    <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
      <section className="text-center">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted">
          <Sparkles size={13} className="text-brand" /> Pearl-framework causal ML
        </span>
        <h1 className="mt-5 text-4xl font-bold tracking-tight sm:text-5xl">
          Causal machine learning for
          <span className="text-brand"> breast cancer diagnosis</span>
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
          Beyond correlation: this project builds an explicit causal graph over tumor cytology features,
          identifies backdoor-adjusted treatment effects with DoWhy, and combines that with interpretable
          predictive models explained via SHAP.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/predict"
            className="rounded-full bg-brand px-6 py-3 text-sm font-semibold text-brand-foreground shadow-sm transition hover:opacity-90"
          >
            Try the prediction tool
          </Link>
          <Link
            href="/methodology"
            className="rounded-full border border-border px-6 py-3 text-sm font-semibold text-foreground transition hover:bg-surface"
          >
            Read the methodology
          </Link>
        </div>
      </section>

      <section className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Samples" value="569" hint="Wisconsin Diagnostic Breast Cancer" icon={<Activity size={18} />} />
        <StatCard
          label="Best model"
          value={modelInfo?.best_model ?? "—"}
          hint={bestMetrics ? `ROC-AUC ${(bestMetrics["ROC-AUC"] * 100).toFixed(1)}%` : "run the pipeline to populate"}
          icon={<ShieldCheck size={18} />}
        />
        <StatCard
          label="Accuracy"
          value={bestMetrics ? `${(bestMetrics["Accuracy"] * 100).toFixed(1)}%` : "—"}
          hint="held-out test set"
          icon={<Sparkles size={18} />}
        />
        <StatCard
          label="Causal features"
          value={modelInfo ? String(modelInfo.n_features) : "—"}
          hint="of 30 raw measurements"
          icon={<GitBranch size={18} />}
        />
      </section>

      <section className="mt-16 grid gap-6 sm:grid-cols-3">
        <FeatureBlock
          title="1. Statistical screening"
          body="ANOVA F-test, mutual information, Random Forest importance and logistic coefficients are combined into a composite causal score to shrink 30 features to a tractable shortlist."
        />
        <FeatureBlock
          title="2. Causal graph (DoWhy)"
          body="A domain-informed DAG encodes tumor cytology mechanics. DoWhy identifies a backdoor adjustment set per treatment and estimates effects, stress-tested with placebo and random-common-cause refutation."
        />
        <FeatureBlock
          title="3. Modeling + SHAP"
          body="Logistic Regression, Random Forest, Gradient Boosting and SVM are compared; SHAP explains individual predictions on top of the causally-motivated feature set."
        />
      </section>
    </div>
  );
}

function FeatureBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-2xl border border-border bg-surface p-6">
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
    </div>
  );
}
