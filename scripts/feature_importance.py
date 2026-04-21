"""
Feature importance analysis for citation link prediction.

Trains a Random Forest on the combined_full feature set and plots
feature importances with confidence intervals (std over trees).

Output: results/exp2/feature_importance.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.embeddings import load_or_compute_embeddings, cosine_similarity_pairs
from src.graph_features import (
    build_nx_graph,
    compute_basic_features,
    compute_community_features,
    compute_triangle_features,
)
from src.kuzu_features import load_entity_mentions, compute_entity_overlap
from src.splitting import sample_negatives, random_split

RESULTS_DIR = PROJECT_ROOT / "results" / "exp2"
EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"

FEATURE_NAMES = [
    "Common Neighbors",
    "Preferential Attachment",
    "Jaccard Coefficient",
    "Triangles (source)",
    "Clustering Coeff. (source)",
    "Same Louvain Comm.",
    "Same LabelProp Comm.",
    "Legal-BERT Similarity",
    "Entity Jaccard",
    "Entity Common Count",
]

FEATURE_GROUPS = {
    "Structural\n(basic)": [0, 1, 2],
    "Structural\n(triangle)": [3, 4],
    "Community": [5, 6],
    "Semantic\n(text)": [7],
    "Semantic\n(entity)": [8, 9],
}

GROUP_COLORS = {
    "Structural\n(basic)": "#4C72B0",
    "Structural\n(triangle)": "#55A868",
    "Community": "#C44E52",
    "Semantic\n(text)": "#8172B2",
    "Semantic\n(entity)": "#CCB974",
}


def build_combined_full(pairs, G, id_to_idx, embeddings, mentions):
    basic = compute_basic_features(G, pairs)
    triangle = compute_triangle_features(G, pairs)
    community = compute_community_features(G, pairs)
    semantic = compute_entity_overlap(mentions, pairs)
    pair_idxs = [(id_to_idx.get(u, 0), id_to_idx.get(v, 0)) for u, v in pairs]
    text_sim = cosine_similarity_pairs(embeddings, pair_idxs).reshape(-1, 1)
    return np.hstack([basic, triangle, community, text_sim, semantic])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_scotus_cases()
    edges = build_edge_list(df)

    print("Splitting...")
    train_edges, test_edges = random_split(edges, test_frac=0.2, seed=42)
    G = build_nx_graph(train_edges)
    all_nodes = list(G.nodes())
    existing_edges = set(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))

    print("Loading embeddings...")
    texts = df["html_with_citations"].fillna("").tolist()
    embeddings = load_or_compute_embeddings(texts, model_name=EMBED_MODEL, batch_size=16)
    id_to_idx = {str(int(float(row["id"]))): i for i, (_, row) in enumerate(df.iterrows())}

    print("Loading entity mentions...")
    mentions = load_entity_mentions()

    train_pos = list(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))
    train_neg = sample_negatives(train_pos, all_nodes, existing_edges, seed=0)
    train_pairs = train_pos + train_neg
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

    print("Building features (combined_full)...")
    X_train = build_combined_full(train_pairs, G, id_to_idx, embeddings, mentions)

    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, train_labels)

    importances = clf.feature_importances_
    std = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)

    # Assign group color per feature
    colors = []
    for i in range(len(FEATURE_NAMES)):
        for group, indices in FEATURE_GROUPS.items():
            if i in indices:
                colors.append(GROUP_COLORS[group])
                break

    # Sort by importance descending
    order = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_imp = importances[order]
    sorted_std = std[order]
    sorted_colors = [colors[i] for i in order]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FEATURE_NAMES))
    bars = ax.bar(x, sorted_imp, color=sorted_colors, yerr=sorted_std,
                  capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Decrease in Impurity", fontsize=10)
    ax.set_title("Feature Importance — Citation Link Prediction (Random Forest)", fontsize=11)
    ax.set_ylim(0, sorted_imp.max() * 1.25)

    # Annotate bars with value
    for bar, imp in zip(bars, sorted_imp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{imp:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Legend for groups
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              title="Feature group", title_fontsize=8)

    fig.tight_layout()
    out_path = RESULTS_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # Print table
    print("\nFeature importances (sorted):")
    print(f"{'Feature':<30} {'Importance':>10} {'Std':>8}")
    print("-" * 52)
    for i in order:
        print(f"{FEATURE_NAMES[i]:<30} {importances[i]:>10.4f} {std[i]:>8.4f}")


if __name__ == "__main__":
    main()
