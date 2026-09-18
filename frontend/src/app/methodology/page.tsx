const SECTIONS = [
  {
    title: "1. The Ladder of Causation",
    body: [
      "Judea Pearl frames causal reasoning in three rungs: (1) association — what do the data show, e.g. correlation and predictive power; (2) intervention — what happens if we do X, i.e. change a variable and observe the effect while holding confounders fixed; (3) counterfactuals — what would have happened had X been different for this specific case.",
      "Most \"feature importance\" techniques, including the four statistical scores and SHAP used in this project, live on rung 1: they are powerful for prediction but do not by themselves establish that a feature causes the outcome.",
    ],
  },
  {
    title: "2. Statistical screening (rung 1)",
    body: [
      "Before any causal claims are made, 30 raw cytology measurements are ranked by a composite causal score: the normalized average of an ANOVA F-test, mutual information, Random Forest importance, and absolute logistic-regression coefficients. This is purely a dimensionality-reduction step — it narrows the feature set to 15 candidates for the next stage.",
    ],
  },
  {
    title: "3. A domain-informed causal graph (rung 2)",
    body: [
      "The causal graph is not learned from the data — it is specified from domain knowledge about tumor cytology, then used to derive testable statistical implications:",
      "• radius mechanically determines perimeter and area (perimeter ≈ 2πr, area ≈ πr²), so radius is modeled as their common cause.",
      "• concavity (severity of concave regions of the cell boundary) drives concave points (count of such regions).",
      "• the *_mean (average cell) and *_worst (most extreme cell) versions of each measurement share an unobserved 'tumor severity' cause, modeled as *_mean → *_worst.",
      "• the four \"worst\" shape/contour measurements are modeled as direct causes of diagnosis, matching how pathologists actually grade malignancy from the most abnormal cells in a sample.",
    ],
  },
  {
    title: "4. Identification and estimation with DoWhy",
    body: [
      "Given the graph, DoWhy applies Pearl's backdoor criterion to identify — for each treatment feature — the minimal set of other variables that must be statistically controlled for to isolate its causal effect on diagnosis. A linear-regression estimator then produces a standardized average treatment effect (ATE) on the adjusted data.",
      "Each estimate is stress-tested with two refutation checks: adding a random common cause (a genuinely causal estimate should barely move), and permuting (placebo-ing) the treatment (a genuinely causal estimate should collapse toward zero). An effect is only reported as 'robust' if the placebo-refuted effect is under half the magnitude of the original.",
    ],
  },
  {
    title: "5. Predictive modeling & SHAP",
    body: [
      "Four classifiers — Logistic Regression, Random Forest, Gradient Boosting, and a calibrated SVM — are trained on the 15 shortlisted features and compared on accuracy, precision, recall, F1, and 5-fold cross-validated ROC-AUC. SHAP (TreeExplainer on the Random Forest) explains individual predictions: a genuinely useful diagnostic aid, but — like the screening step — a rung-1, model-specific explanation rather than a causal claim.",
    ],
  },
  {
    title: "6. Reproducing this analysis",
    body: [
      "The entire pipeline — data generation, feature screening, causal graph construction and estimation, model training, SHAP, and artifact export — runs end to end with a single command from the repo root: python -m ml.pipeline. See notebooks/01_EDA.ipynb and notebooks/02_Causal_Inference_and_Modeling.ipynb for a narrated walkthrough, or README.md for the full architecture.",
    ],
  },
];

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-bold tracking-tight">Methodology</h1>
      <p className="mt-2 text-muted">
        How this project moves from raw cytology measurements to a causally-motivated, interpretable
        diagnosis model.
      </p>

      <div className="mt-10 space-y-10">
        {SECTIONS.map((s) => (
          <section key={s.title}>
            <h2 className="text-lg font-semibold">{s.title}</h2>
            <div className="mt-2 space-y-2 text-sm leading-relaxed text-muted">
              {s.body.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
          </section>
        ))}
      </div>

      <div className="mt-12 rounded-2xl border border-border bg-surface p-6 text-sm text-muted">
        <strong className="text-foreground">Disclaimer:</strong> this project is for education and research
        into causal machine learning methodology. It is not validated for clinical use and must not be used
        to make real diagnostic decisions.
      </div>
    </div>
  );
}
