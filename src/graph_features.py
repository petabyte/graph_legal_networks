from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import label_propagation_communities, louvain_communities


def build_nx_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from the citation edge DataFrame."""
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(str(row["source_id"]), str(row["target_id"]))
    return G


def compute_basic_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [common_neighbors, pref_attachment, jaccard].
    Uses undirected projection for neighbor-based metrics.
    Returns shape (N, 3).
    """
    U = G.to_undirected()
    rows = []
    for u, v in pairs:
        if U.has_node(u) and U.has_node(v):
            cn = len(list(nx.common_neighbors(U, u, v)))
            pa = U.degree(u) * U.degree(v)
            jac = next(nx.jaccard_coefficient(U, [(u, v)]))[2]
        else:
            cn, pa, jac = 0, 0, 0.0
        rows.append([cn, pa, jac])
    return np.array(rows, dtype=float)


def compute_triangle_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [triangles_u, clustering_u].
    Returns shape (N, 2).
    """
    U = G.to_undirected()
    triangles = nx.triangles(U)
    clustering = nx.clustering(U)
    rows = []
    for u, _ in pairs:
        rows.append([triangles.get(u, 0), clustering.get(u, 0.0)])
    return np.array(rows, dtype=float)


def compute_community_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [same_louvain_community, same_label_prop_community].
    Returns shape (N, 2).
    """
    U = G.to_undirected()

    # Louvain (built-in networkx)
    louvain_comms = louvain_communities(U, seed=42)
    louvain: dict[str, int] = {}
    for i, comm in enumerate(louvain_comms):
        for node in comm:
            louvain[node] = i

    # Label propagation
    lp_comms = label_propagation_communities(U)
    lp: dict[str, int] = {}
    for i, comm in enumerate(lp_comms):
        for node in comm:
            lp[node] = i

    rows = []
    for u, v in pairs:
        same_louvain = int(louvain.get(u, -1) == louvain.get(v, -2))
        same_lp = int(lp.get(u, -1) == lp.get(v, -2))
        rows.append([same_louvain, same_lp])
    return np.array(rows, dtype=float)
