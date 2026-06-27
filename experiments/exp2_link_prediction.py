"""
Experiment 2: Link Prediction Ablation
5 feature sets × 2 models (RF + LR). Reports AUC table and ROC curve figure.

Feature sets:
  text_only       — cosine similarity of Legal-BERT embeddings
  graph_basic     — common neighbors, preferential attachment, Jaccard
  graph_triangle  — graph_basic + triangle count, clustering coefficient
  graph_community — graph_triangle + same-community indicators (Louvain + LP)
  combined        — all graph features + text similarity

Negative sampling:
  Training: 50 % hard (same Louvain community as positive target, not cited)
            + 50 % random non-edges, 1:1 ratio.
  Binary test: same hard/random mix.
  Ranking test: 10 negatives per positive (same-community preferred).

Evaluation:
  Binary classification: AUC, Precision, Recall, F1 (saved to ablation_results.csv)
  Ranking (1 pos vs 10 neg): MRR, Hits@1, Hits@5, Hits@10
           (saved to ranking_metrics.csv)

Graph features (common neighbours, Louvain communities, etc.) are computed
from the training subgraph only — no test edges are present in G.

Note on temporal split: the dataset's `date_created` field records the
CourtListener database-ingestion timestamp (clusters ~2010), not the actual
SCOTUS decision date.  A meaningful temporal split would require extracting
decision years from opinion text (e.g. via LLM); this is left for future work.

Output files (in results/exp2/):
  ablation_results.csv  — feature_set × model × AUC/F1/Precision/Recall
  ranking_metrics.csv   — feature_set × model × MRR/Hits@1/Hits@5/Hits@10
  roc_curves.png        — ROC curves for all feature sets (LR only)
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
    get_louvain_communities,
)
from src.kuzu_features import load_entity_mentions, compute_entity_overlap
from src.models import train_evaluate, ranking_metrics
from src.splitting import (
    random_split,
    sample_hard_negatives,
    sample_negatives_ranked,
)

RESULTS_DIR = _PROJECT_ROOT / "results" / "exp2"
EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"
N_RANK_NEG = 10  # negatives per positive in the ranking evaluation


def _assemble_features(
    pairs: list[tuple[str, str]],
    G,
    id_to_idx: dict[str, int],
    embeddings: np.ndarray,
    mentions: dict,
    feature_set: str,
) -> np.ndarray:
    """Build feature matrix for a given feature_set name."""
    basic = compute_basic_features(G, pairs)          # (N, 3)
    triangle = compute_triangle_features(G, pairs)     # (N, 2)
    community = compute_community_features(G, pairs)   # (N, 2)
    semantic = compute_entity_overlap(mentions, pairs) # (N, 2)

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
    elif feature_set == "semantic":
        return semantic
    elif feature_set == "combined_full":
        return np.hstack([basic, triangle, community, text_sim, semantic])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_scotus_cases()
    edges = build_edge_list(df)

    print("Random 80/20 split...")
    train_edges, test_edges = random_split(edges, test_frac=0.2, seed=42)
    print(f"  Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")

    # Build training-only graph.  All graph features are computed from G so
    # that no test-edge neighbourhoods leak into the feature matrix.
    G = build_nx_graph(train_edges)
    all_nodes = list(G.nodes())
    existing_edges = set(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))

    # Community assignments from the training graph — used for hard-negative sampling.
    print("Computing Louvain communities on training graph...")
    community = get_louvain_communities(G)

    texts = df["html_with_citations"].fillna("").tolist()
    embeddings = load_or_compute_embeddings(texts, model_name=EMBED_MODEL, batch_size=16)
    id_to_idx = {str(int(float(row["id"]))): i for i, (_, row) in enumerate(df.iterrows())}

    print("Loading entity mentions from Kuzu graph...")
    mentions = load_entity_mentions()
    coverage = sum(1 for n in all_nodes if n in mentions)
    print(f"  Entity mentions loaded for {len(mentions)} cases "
          f"({coverage}/{len(all_nodes)} graph nodes covered)")

    all_edges_set = set(zip(edges["source_id"].astype(str), edges["target_id"].astype(str)))

    # --- Training pairs: 1:1 hard/random mix ---
    train_pos = list(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))
    train_neg = sample_hard_negatives(
        train_pos, community, all_nodes, existing_edges, seed=0, hard_frac=0.5
    )
    train_pairs = train_pos + train_neg
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

    # --- Binary test pairs: same hard/random mix ---
    test_pos = list(zip(test_edges["source_id"].astype(str), test_edges["target_id"].astype(str)))
    test_neg = sample_hard_negatives(
        test_pos, community, all_nodes, all_edges_set, seed=42, hard_frac=0.5
    )
    test_pairs = test_pos + test_neg
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    print(f"  Train pairs: {len(train_pairs)} ({len(train_pos)} pos + {len(train_neg)} neg)")
    print(f"  Test pairs:  {len(test_pairs)} ({len(test_pos)} pos + {len(test_neg)} neg)")

    # --- Ranking test set: 1 positive + N_RANK_NEG negatives per query ---
    print(f"Building ranking test set (1 pos + {N_RANK_NEG} hard/random neg per query)...")
    ranking_neg_groups = sample_negatives_ranked(
        test_pos, community, all_nodes, all_edges_set, n_neg=N_RANK_NEG, seed=42
    )
    ranking_pairs: list[tuple[str, str]] = []
    ranking_labels_list: list[int] = []
    for (u, v), neg_list in zip(test_pos, ranking_neg_groups):
        ranking_pairs.append((u, v))
        ranking_labels_list.append(1)
        ranking_pairs.extend(neg_list)
        ranking_labels_list.extend([0] * len(neg_list))
    ranking_labels = np.array(ranking_labels_list)
    ranking_group_size = N_RANK_NEG + 1  # 1 pos + N_RANK_NEG neg

    feature_sets = [
        "text_only", "graph_basic", "graph_triangle", "graph_community",
        "combined", "semantic", "combined_full",
    ]
    model_names = ["rf", "lr"]
    results_rows = []
    ranking_rows = []
    roc_data: dict[str, tuple] = {}

    for fs in feature_sets:
        print(f"\nAssembling features for: {fs}")
        X_train = _assemble_features(train_pairs, G, id_to_idx, embeddings, mentions, fs)
        X_test = _assemble_features(test_pairs, G, id_to_idx, embeddings, mentions, fs)
        X_ranking = _assemble_features(ranking_pairs, G, id_to_idx, embeddings, mentions, fs)

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

            # Score ranking test set with the same fitted model / scaler
            clf = res["clf"]
            scaler = res["scaler"]
            X_r = scaler.transform(X_ranking) if scaler is not None else X_ranking
            rank_scores = clf.predict_proba(X_r)[:, 1]
            r_metrics = ranking_metrics(ranking_labels, rank_scores, group_size=ranking_group_size)
            ranking_rows.append({
                "feature_set": fs,
                "model": model_name.upper(),
                **r_metrics,
            })

    results_df = pd.DataFrame(results_rows)
    print("\nAblation results (binary classification):")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    ranking_df = pd.DataFrame(ranking_rows)
    print(f"\nRanking metrics (1 pos vs {N_RANK_NEG} hard/random neg):")
    print(ranking_df.to_string(index=False))
    ranking_df.to_csv(RESULTS_DIR / "ranking_metrics.csv", index=False)

    # ROC curves (LR only — LR outperforms RF across all feature sets)
    fig, ax = plt.subplots(figsize=(7, 5))
    for fs in feature_sets:
        fpr, tpr, auc = roc_data[f"{fs}_lr"]
        ax.plot(fpr, tpr, label=f"{fs} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Link Prediction (Logistic Regression)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
