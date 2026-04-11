"""
Experiment 1: Graph Construction Analysis
Validates the pipeline: graph statistics, degree distribution,
top cases by in-degree, and Louvain community structure.

Output files (in results/exp1/):
  graph_stats.csv          — node count, edge count, avg degree, density
  degree_distribution.png  — log-log plot of in-degree rank
  top10_indegree.csv       — top 10 most-cited cases by in-degree
  top_case_per_community.csv — top case per Louvain community
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure the project root is on sys.path when running from experiments/
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
from pathlib import Path

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.graph_features import build_nx_graph

RESULTS_DIR = Path("results/exp1")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scotus_cases()
    edges = build_edge_list(df)
    G = build_nx_graph(edges)
    U = G.to_undirected()

    # Build id → case name mapping
    # The dataset may have a 'case_name' or 'name' column; fall back to id string
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    if name_col:
        id_to_name = {
            str(int(float(row["id"]))): str(row[name_col])
            for _, row in df.iterrows()
        }
    else:
        id_to_name = {str(int(float(row["id"]))): str(int(float(row["id"]))) for _, row in df.iterrows()}

    # --- Graph statistics ---
    num_nodes = G.number_of_nodes()
    in_degrees = dict(G.in_degree())
    stats = {
        "nodes": num_nodes,
        "edges": G.number_of_edges(),
        "avg_in_degree": sum(in_degrees.values()) / num_nodes if num_nodes else 0,
        "density": nx.density(G),
    }
    print("Graph statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    pd.DataFrame([stats]).to_csv(RESULTS_DIR / "graph_stats.csv", index=False)

    # --- Degree distribution (log-log) ---
    sorted_degrees = sorted(in_degrees.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(range(1, len(sorted_degrees) + 1), sorted_degrees, ".", markersize=3, alpha=0.6)
    ax.set_xlabel("Rank")
    ax.set_ylabel("In-degree")
    ax.set_title("SCOTUS Citation In-Degree Distribution (log-log)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "degree_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved degree_distribution.png")

    # --- Top 10 most-cited cases ---
    top10 = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_df = pd.DataFrame(
        [(id_to_name.get(cid, cid), deg) for cid, deg in top10],
        columns=["case_name", "in_degree"],
    )
    print("\nTop 10 most-cited cases:")
    print(top10_df.to_string(index=False))
    top10_df.to_csv(RESULTS_DIR / "top10_indegree.csv", index=False)

    # --- Louvain community detection ---
    comms = louvain_communities(U, seed=42)
    num_communities = len(comms)
    print(f"\nLouvain communities detected: {num_communities}")

    # Assign community IDs to nodes
    node_to_comm: dict[str, int] = {}
    for i, comm in enumerate(comms):
        for node in comm:
            node_to_comm[node] = i

    # Top case per community by in-degree
    community_rows = [
        {"case_name": id_to_name.get(node, node), "community": comm_id, "in_degree": in_degrees.get(node, 0)}
        for node, comm_id in node_to_comm.items()
    ]
    comm_df = pd.DataFrame(community_rows)
    top_per_community = (
        comm_df.sort_values("in_degree", ascending=False)
        .groupby("community")
        .first()
        .reset_index()[["community", "case_name", "in_degree"]]
        .sort_values("community")
    )
    print("\nTop case per community (first 10 communities):")
    print(top_per_community.head(10).to_string(index=False))
    top_per_community.to_csv(RESULTS_DIR / "top_case_per_community.csv", index=False)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
