import networkx as nx
import numpy as np
import pandas as pd

from src.graph_features import (
    build_nx_graph,
    compute_basic_features,
    compute_community_features,
    compute_triangle_features,
)


def _small_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])
    return G


def test_build_nx_graph_from_edge_df():
    edge_df = pd.DataFrame({
        "source_id": ["A", "B", "A"],
        "target_id": ["B", "C", "C"],
        "year": [2000, 2001, 2002],
    })
    G = build_nx_graph(edge_df)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3


def test_compute_basic_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("A", "D")]
    features = compute_basic_features(G, pairs)
    assert features.shape == (2, 3)  # common_neighbors, pref_attachment, jaccard


def test_compute_basic_features_values():
    G = _small_graph()
    # A and C share neighbor B (in undirected view) and A→C is direct but let's check common_neighbors
    pairs = [("A", "C")]
    features = compute_basic_features(G, pairs)
    # In undirected: A-B, B-C, A-C, C-D. Common neighbors of A,C = {B}
    assert features[0, 0] >= 1  # at least one common neighbor


def test_compute_triangle_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("B", "D")]
    features = compute_triangle_features(G, pairs)
    assert features.shape == (2, 2)  # triangles_source, clustering_source


def test_compute_community_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("A", "D")]
    features = compute_community_features(G, pairs)
    assert features.shape == (2, 2)  # same_louvain, same_label_prop


def test_compute_community_features_same_node_is_same_community():
    G = _small_graph()
    # A vs A should always be same community
    pairs = [("A", "A")]
    features = compute_community_features(G, pairs)
    assert features[0, 0] == 1  # same_louvain
    assert features[0, 1] == 1  # same_label_prop


def test_features_handle_unknown_nodes_gracefully():
    G = _small_graph()
    pairs = [("X", "Y")]  # neither X nor Y in graph
    basic = compute_basic_features(G, pairs)
    triangle = compute_triangle_features(G, pairs)
    community = compute_community_features(G, pairs)
    assert basic.shape == (1, 3)
    assert triangle.shape == (1, 2)
    assert community.shape == (1, 2)
    # No crash, reasonable defaults (zeros or -1-based same community)
