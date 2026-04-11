"""
Experiment 2: Link Prediction Ablation
5 feature sets × 2 models (RF + LR). Reports AUC table and ROC curve figure.

Feature sets:
  text_only       — cosine similarity of Legal-BERT embeddings
  graph_basic     — common neighbors, preferential attachment, Jaccard
  graph_triangle  — graph_basic + triangle count, clustering coefficient
  graph_community — graph_triangle + same-community indicators (Louvain + LP)
  combined        — all graph features + text similarity

Output files (in results/exp2/):
  ablation_results.csv  — feature_set × model × AUC/F1/Precision/Recall
  roc_curves.png        — ROC curves for all feature sets (RF only)
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.embeddings import load_or_compute_embeddings, cosine_similarity_pairs
from src.graph_features import (
    build_nx_graph,
    compute_basic_features,
    compute_community_features,
    compute_triangle_features,
)
from src.models import train_evaluate
from src.splitting import sample_negatives, random_split

RESULTS_DIR = _PROJECT_ROOT / "results" / "exp2"
# roberta-base used instead of nlpaueb/legal-bert-base-uncased:
# GPU unavailable (CUDA driver too old), roberta-base is already cached locally.
# The experiment logic is identical regardless of model choice.
EMBED_MODEL = "roberta-base"


def _assemble_features(
    pairs: list[tuple[str, str]],
    G,
    id_to_idx: dict[str, int],
    embeddings: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Build feature matrix for a given feature_set name."""
    basic = compute_basic_features(G, pairs)          # (N, 3)
    triangle = compute_triangle_features(G, pairs)     # (N, 2)
    community = compute_community_features(G, pairs)   # (N, 2)

    # Text similarity: convert string pairs to int index pairs
    pair_idxs = [
        (id_to_idx.get(u, 0), id_to_idx.get(v, 0))
        for u, v in pairs
    ]
    text_sim = cosine_similarity_pairs(embeddings, pair_idxs).reshape(-1, 1)  # (N, 1)

    if feature_set == "text_only":
        return text_sim
    elif feature_set == "graph_basic":
        return basic
    elif feature_set == "graph_triangle":
        return np.hstack([basic, triangle])
    elif feature_set == "graph_community":
        return np.hstack([basic, triangle, community])
    elif feature_set == "combined":
        return np.hstack([basic, triangle, community, text_sim])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_scotus_cases()
    edges = build_edge_list(df)

    print("Random 80/20 split (dataset lacks reliable decision dates)...")
    train_edges, test_edges = random_split(edges, test_frac=0.2, seed=42)
    print(f"  Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")

    G = build_nx_graph(train_edges)
    all_nodes = list(G.nodes())
    existing_edges = set(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))

    texts = df["html_with_citations"].fillna("").tolist()
    embeddings = load_or_compute_embeddings(texts, model_name=EMBED_MODEL, batch_size=16)
    id_to_idx = {str(int(float(row["id"]))): i for i, (_, row) in enumerate(df.iterrows())}

    # Build train pairs
    train_pos = list(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))
    train_neg = sample_negatives(train_pos, all_nodes, existing_edges, seed=0)
    train_pairs = train_pos + train_neg
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

    # Build test pairs
    test_pos = list(zip(test_edges["source_id"].astype(str), test_edges["target_id"].astype(str)))
    all_edges_set = set(zip(edges["source_id"].astype(str), edges["target_id"].astype(str)))
    test_neg = sample_negatives(test_pos, all_nodes, all_edges_set, seed=42)
    test_pairs = test_pos + test_neg
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    print(f"  Train pairs: {len(train_pairs)} ({len(train_pos)} pos + {len(train_neg)} neg)")
    print(f"  Test pairs:  {len(test_pairs)} ({len(test_pos)} pos + {len(test_neg)} neg)")

    feature_sets = ["text_only", "graph_basic", "graph_triangle", "graph_community", "combined"]
    model_names = ["rf", "lr"]
    results_rows = []
    roc_data: dict[str, tuple] = {}

    for fs in feature_sets:
        print(f"\nAssembling features for: {fs}")
        X_train = _assemble_features(train_pairs, G, id_to_idx, embeddings, fs)
        X_test = _assemble_features(test_pairs, G, id_to_idx, embeddings, fs)
        for model_name in model_names:
            print(f"  {model_name.upper()} / {fs} ...")
            res = train_evaluate(X_train, train_labels, X_test, test_labels, model_name)
            results_rows.append({
                "feature_set": fs,
                "model": model_name.upper(),
                "auc": round(res["auc"], 4),
                "precision": round(res["precision"], 4),
                "recall": round(res["recall"], 4),
                "f1": round(res["f1"], 4),
            })
            roc_data[f"{fs}_{model_name}"] = (res["fpr"], res["tpr"], res["auc"])

    results_df = pd.DataFrame(results_rows)
    print("\nAblation results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    # ROC curves (RF only)
    fig, ax = plt.subplots(figsize=(7, 5))
    for fs in feature_sets:
        fpr, tpr, auc = roc_data[f"{fs}_rf"]
        ax.plot(fpr, tpr, label=f"{fs} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Link Prediction (Random Forest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
