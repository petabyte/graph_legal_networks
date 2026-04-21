"""
DeLong's test for comparing AUC of two classifiers on the same test set.
Compares text_only vs combined_full (LR) using the paired structure of the test set.

Also reports bootstrap 95% CIs for all feature sets.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.embeddings import load_or_compute_embeddings, cosine_similarity_pairs
from src.graph_features import build_nx_graph, compute_basic_features, compute_community_features, compute_triangle_features
from src.kuzu_features import load_entity_mentions, compute_entity_overlap
from src.splitting import sample_negatives, random_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"
N_BOOTSTRAP = 2000
SEED = 42


def delong_test(y_true, prob_a, prob_b):
    """
    DeLong's test for paired AUC comparison.
    Returns (z_stat, p_value, auc_a, auc_b).
    Implements the fast DeLong variance estimator (Sun & Xu, 2014).
    """
    def auc_and_var(y, p):
        pos = p[y == 1]
        neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        # Placement values
        v10 = np.array([(p_i > neg).mean() + 0.5 * (p_i == neg).mean() for p_i in pos])
        v01 = np.array([(p_j < pos).mean() + 0.5 * (p_j == pos).mean() for p_j in neg])
        auc = v10.mean()
        s10 = v10.var() / n1
        s01 = v01.var() / n0
        var = s10 + s01
        return auc, var, v10, v01

    auc_a, var_a, v10_a, v01_a = auc_and_var(y_true, prob_a)
    auc_b, var_b, v10_b, v01_b = auc_and_var(y_true, prob_b)

    pos_idx = y_true == 1
    neg_idx = y_true == 0
    n1 = pos_idx.sum()
    n0 = neg_idx.sum()

    # Covariance
    cov = (np.cov(v10_a, v10_b)[0, 1] / n1 +
           np.cov(v01_a, v01_b)[0, 1] / n0)

    var_diff = var_a + var_b - 2 * cov
    if var_diff <= 0:
        return 0.0, 1.0, auc_a, auc_b
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p, auc_a, auc_b


def bootstrap_ci(y_true, y_prob, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def fit_lr(X_train, y_train, X_test):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te)[:, 1]


def main():
    print("Loading data...")
    df = load_scotus_cases()
    edges = build_edge_list(df)
    train_edges, test_edges = random_split(edges, test_frac=0.2, seed=42)
    G = build_nx_graph(train_edges)
    all_nodes = list(G.nodes())
    existing = set(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))

    import pandas as pd
    texts = df["html_with_citations"].fillna("").tolist()
    embeddings = load_or_compute_embeddings(texts, model_name=EMBED_MODEL, batch_size=16)
    id_to_idx = {str(int(float(row["id"]))): i for i, (_, row) in enumerate(df.iterrows())}
    mentions = load_entity_mentions()

    train_pos = list(zip(train_edges["source_id"].astype(str), train_edges["target_id"].astype(str)))
    train_neg = sample_negatives(train_pos, all_nodes, existing, seed=0)
    train_pairs = train_pos + train_neg
    y_train = np.array([1]*len(train_pos) + [0]*len(train_neg))

    all_edges = set(zip(edges["source_id"].astype(str), edges["target_id"].astype(str)))
    test_pos = list(zip(test_edges["source_id"].astype(str), test_edges["target_id"].astype(str)))
    test_neg = sample_negatives(test_pos, all_nodes, all_edges, seed=42)
    test_pairs = test_pos + test_neg
    y_test = np.array([1]*len(test_pos) + [0]*len(test_neg))

    def feats(pairs, fs):
        basic = compute_basic_features(G, pairs)
        tri = compute_triangle_features(G, pairs)
        comm = compute_community_features(G, pairs)
        sem = compute_entity_overlap(mentions, pairs)
        pidxs = [(id_to_idx.get(u,0), id_to_idx.get(v,0)) for u,v in pairs]
        txt = cosine_similarity_pairs(embeddings, pidxs).reshape(-1,1)
        if fs == "text_only": return txt
        if fs == "graph_basic": return basic
        if fs == "combined_full": return np.hstack([basic, tri, comm, txt, sem])
        return np.hstack([basic, tri, comm, txt, sem])

    print("\nFitting text_only and combined_full...")
    prob_text = fit_lr(feats(train_pairs, "text_only"), y_train, feats(test_pairs, "text_only"))
    prob_full = fit_lr(feats(train_pairs, "combined_full"), y_train, feats(test_pairs, "combined_full"))

    z, p, auc_a, auc_b = delong_test(y_test, prob_full, prob_text)
    ci_text = bootstrap_ci(y_test, prob_text)
    ci_full = bootstrap_ci(y_test, prob_full)

    print(f"\n{'='*55}")
    print(f"  text_only     AUC = {auc_b:.4f}  95% CI [{ci_text[0]:.4f}, {ci_text[1]:.4f}]")
    print(f"  combined_full AUC = {auc_a:.4f}  95% CI [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    print(f"  DeLong z = {z:.3f},  p = {p:.2e}")
    if p < 0.001:
        print(f"  *** Highly significant (p < 0.001)")
    elif p < 0.05:
        print(f"  * Significant (p < 0.05)")
    else:
        print(f"  Not significant (p >= 0.05)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
