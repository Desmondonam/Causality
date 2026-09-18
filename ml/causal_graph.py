"""Pearl-style causal inference on the breast-cancer feature set.

This module is the actual "causality" part of the project (the earlier
statistical feature-selection step only measures association). We:

1. Encode domain knowledge about tumor cytology as a directed acyclic graph
   (DAG) with :mod:`networkx`.
2. Hand that DAG to `DoWhy <https://www.pywhy.org/dowhy/>`_, which uses
   Pearl's graphical do-calculus to identify a valid *backdoor adjustment
   set* for each treatment -> outcome pair.
3. Estimate the adjusted average treatment effect (ATE) of each "worst"
   measurement on malignancy.
4. Stress-test each estimate with refutation tests (adding a random common
   cause, and a placebo/permuted treatment) - if the effect doesn't survive
   these, it likely wasn't causal to begin with.

Domain rationale for the DAG (see the causal-inference notebook for the
full discussion):

- ``radius`` is the primary geometric measurement of a cell nucleus, and
  mechanically determines ``perimeter`` and ``area`` (perimeter ~ 2*pi*r,
  area ~ pi*r^2), so radius is modeled as their common cause.
- ``concavity`` (severity of concave portions of the contour) drives
  ``concave points`` (number of concave portions), since more severe
  concavities produce more concave points.
- ``*_mean`` describes the average cell in a sample; ``*_worst`` describes
  the most extreme cell. A more severe underlying tumor tends to produce
  both a higher mean *and* a higher worst value, so we model
  ``*_mean -> *_worst``.
- Malignancy (``diagnosis``) is caused by irregular, large nuclei: we treat
  the four "worst" shape/contour features as direct causes of diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from ml.config import TARGET


def build_causal_graph() -> nx.DiGraph:
    """Return the domain-informed causal DAG over the shortlisted features."""
    g = nx.DiGraph()
    edges = [
        # Geometric mechanics: radius determines perimeter & area.
        ("radius_mean", "perimeter_mean"),
        ("radius_mean", "area_mean"),
        ("radius_worst", "perimeter_worst"),
        ("radius_worst", "area_worst"),
        # Contour irregularity: concavity severity drives concave-point count.
        ("concavity_mean", "concave points_mean"),
        ("concavity_worst", "concave points_worst"),
        # Underlying tumor severity links the "mean" cell to the "worst" cell.
        ("radius_mean", "radius_worst"),
        ("concavity_mean", "concavity_worst"),
        ("concave points_mean", "concave points_worst"),
        ("area_mean", "area_worst"),
        ("perimeter_mean", "perimeter_worst"),
        # Direct causes of malignancy: shape/contour of the most extreme cell.
        ("radius_worst", TARGET),
        ("perimeter_worst", TARGET),
        ("area_worst", TARGET),
        ("concavity_worst", TARGET),
        ("concave points_worst", TARGET),
    ]
    g.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(g), "Causal graph must be acyclic"
    return g


def graph_to_gml(graph: nx.DiGraph) -> str:
    """Serialize to GML, the format DoWhy expects for a raw graph string."""
    return "\n".join(nx.generate_gml(graph))


@dataclass
class CausalEffectResult:
    treatment: str
    ate: float
    refute_random_common_cause: float
    refute_placebo: float
    adjustment_set: list[str] = field(default_factory=list)

    @property
    def robust(self) -> bool:
        """Heuristic: the effect survives refutation if the placebo effect
        collapses towards zero relative to the original estimate."""
        if self.ate == 0:
            return False
        return abs(self.refute_placebo) < 0.5 * abs(self.ate)


def estimate_causal_effects(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    treatments: list[str] | None = None,
) -> list[CausalEffectResult]:
    """Estimate the backdoor-adjusted ATE of each treatment on diagnosis.

    ``df`` must contain every node in ``graph`` plus the binary target
    column. Continuous features are standardized first so effect sizes are
    comparable to one another (a "standardized ATE").
    """
    # Imported lazily: dowhy pulls in a heavy dependency graph and touches
    # stdout with unicode math notation that misbehaves on cp1252 consoles.
    import sys

    from dowhy import CausalModel

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if treatments is None:
        treatments = [n for n in graph.predecessors(TARGET)]

    data = df.copy()
    feature_cols = [c for c in graph.nodes if c != TARGET]
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    gml = graph_to_gml(graph)
    results: list[CausalEffectResult] = []

    for treatment in treatments:
        model = CausalModel(data=data, treatment=treatment, outcome=TARGET, graph=gml)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

        # num_simulations kept modest (default is 100): each simulation
        # refits the estimator, and 5 treatments x 2 refuters x 100 sims is
        # slow enough to be impractical for a local `python -m ml.pipeline`
        # run or CI. 10 simulations is still enough to see whether an effect
        # collapses under a placebo/random-cause perturbation.
        try:
            random_cause = model.refute_estimate(
                estimand, estimate, method_name="random_common_cause", num_simulations=10
            )
            random_cause_value = float(random_cause.new_effect)
        except Exception:
            random_cause_value = float("nan")

        try:
            placebo = model.refute_estimate(
                estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=10,
            )
            placebo_value = float(placebo.new_effect)
        except Exception:
            placebo_value = float("nan")

        adjustment_set = sorted(
            {v for values in estimand.backdoor_variables.values() for v in values}
        ) if isinstance(estimand.backdoor_variables, dict) else list(estimand.backdoor_variables or [])

        results.append(
            CausalEffectResult(
                treatment=treatment,
                ate=float(estimate.value),
                refute_random_common_cause=random_cause_value,
                refute_placebo=placebo_value,
                adjustment_set=adjustment_set,
            )
        )

    return results


def results_to_frame(results: list[CausalEffectResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "treatment": r.treatment,
                "standardized_ATE": r.ate,
                "refute_random_common_cause": r.refute_random_common_cause,
                "refute_placebo": r.refute_placebo,
                "adjustment_set": ", ".join(r.adjustment_set) if r.adjustment_set else "(none)",
                "robust_to_placebo": r.robust,
            }
            for r in results
        ]
    ).sort_values("standardized_ATE", ascending=False)


def draw_causal_graph(graph: nx.DiGraph, path):
    """Render the DAG with matplotlib (no Graphviz/pygraphviz dependency,
    so this works unmodified in CI and on Windows)."""
    import matplotlib.pyplot as plt

    generations = list(nx.topological_generations(graph))
    pos = {}
    for depth, layer in enumerate(generations):
        for i, node in enumerate(sorted(layer)):
            pos[node] = (depth, -(i - (len(layer) - 1) / 2))

    fig, ax = plt.subplots(figsize=(13, 7))
    node_colors = ["#ef4444" if n == TARGET else "#3b82f6" for n in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrowstyle="-|>", arrowsize=18, edge_color="#64748b", connectionstyle="arc3,rad=0.05"
    )
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    ax.set_title("Domain-informed causal DAG (Pearl framework)", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from ml.config import FIGURES_DIR, REPORTS_DIR
    from ml.data import clean_dataset, load_dataset

    dag = build_causal_graph()
    draw_causal_graph(dag, FIGURES_DIR / "causal_graph.png")

    df = clean_dataset(load_dataset())
    effects = estimate_causal_effects(df, dag)
    frame = results_to_frame(effects)
    frame.to_csv(REPORTS_DIR / "causal_effects.csv", index=False)
    print(frame.to_string(index=False))
