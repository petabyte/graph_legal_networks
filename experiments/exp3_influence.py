"""
Experiment 3: Influence Analysis
PageRank + betweenness centrality on the full SCOTUS citation graph.
Validates that the graph recovers the known legal canon.

Output files (in results/exp3/):
  top20_pagerank.csv    — top 20 cases by PageRank score
  top20_betweenness.csv — top 20 cases by betweenness centrality
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import networkx as nx
import pandas as pd

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.graph_features import build_nx_graph

RESULTS_DIR = _PROJECT_ROOT / "results" / "exp3"

# Known landmark SCOTUS cases for qualitative validation
LANDMARK_CASES = {
    "Marbury v. Madison",
    "McCulloch v. Maryland",
    "Brown v. Board of Education",
    "Roe v. Wade",
    "Miranda v. Arizona",
    "Gideon v. Wainwright",
    "Mapp v. Ohio",
    "United States v. Nixon",
    "Bush v. Gore",
    "Obergefell v. Hodges",
}


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scotus_cases()
    edges = build_edge_list(df)
    G = build_nx_graph(edges)

    # Build id → case name mapping
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    if name_col:
        id_to_name = {
            str(int(float(row["id"]))): str(row[name_col])
            for _, row in df.iterrows()
        }
    else:
        id_to_name = {}

    def display_name(node_id: str) -> str:
        return id_to_name.get(node_id, node_id)

    # --- PageRank ---
    print("Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)
    top20_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_pr_df = pd.DataFrame(
        [(display_name(cid), round(score, 6)) for cid, score in top20_pr],
        columns=["case_name", "pagerank"],
    )
    print("Top 20 by PageRank:")
    print(top20_pr_df.to_string(index=False))
    top20_pr_df.to_csv(RESULTS_DIR / "top20_pagerank.csv", index=False)

    # --- Betweenness Centrality (approximate, k=500 for speed) ---
    print("\nComputing betweenness centrality (k=500 approximation)...")
    betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), normalized=True, seed=42)
    top20_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_bc_df = pd.DataFrame(
        [(display_name(cid), round(score, 6)) for cid, score in top20_bc],
        columns=["case_name", "betweenness"],
    )
    print("\nTop 20 by Betweenness Centrality:")
    print(top20_bc_df.to_string(index=False))
    top20_bc_df.to_csv(RESULTS_DIR / "top20_betweenness.csv", index=False)

    # --- Qualitative validation ---
    pr_names = set(top20_pr_df["case_name"])
    bc_names = set(top20_bc_df["case_name"])
    pr_hits = LANDMARK_CASES & pr_names
    bc_hits = LANDMARK_CASES & bc_names

    print(f"\nLandmark cases in top-20 PageRank: {len(pr_hits)}/{len(LANDMARK_CASES)}")
    for name in sorted(pr_hits):
        print(f"  + {name}")

    print(f"\nLandmark cases in top-20 Betweenness: {len(bc_hits)}/{len(LANDMARK_CASES)}")
    for name in sorted(bc_hits):
        print(f"  + {name}")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
