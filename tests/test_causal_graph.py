import networkx as nx

from ml.causal_graph import build_causal_graph, graph_to_gml
from ml.config import TARGET


def test_causal_graph_is_a_dag():
    g = build_causal_graph()
    assert nx.is_directed_acyclic_graph(g)


def test_diagnosis_has_only_worst_measurement_parents():
    g = build_causal_graph()
    parents = set(g.predecessors(TARGET))
    assert parents == {
        "radius_worst",
        "perimeter_worst",
        "area_worst",
        "concavity_worst",
        "concave points_worst",
    }


def test_graph_serializes_to_gml():
    g = build_causal_graph()
    gml = graph_to_gml(g)
    assert "graph" in gml
    assert TARGET in gml
