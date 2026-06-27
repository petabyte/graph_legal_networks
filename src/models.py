from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "rf",
) -> dict:
    """
    Train a classifier and return evaluation metrics.
    model_name: "rf" for RandomForest, "lr" for LogisticRegression.
    Returns dict with keys: auc, precision, recall, f1, fpr, tpr,
    feature_importances, clf, scaler (scaler is None for RF).
    fpr and tpr are Python lists (JSON-serializable).
    """
    scaler = None
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif model_name == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Use 'rf' or 'lr'.")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "feature_importances": getattr(clf, "feature_importances_", None),
        "clf": clf,
        "scaler": scaler,
    }


def ranking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    group_size: int,
) -> dict:
    """
    Compute MRR, Hits@1, Hits@5, Hits@10 over grouped ranking lists.

    y_true and y_scores must be laid out as consecutive groups of `group_size`
    pairs, each group containing exactly one positive (y_true == 1) and
    group_size-1 negatives.  The positive need not be first.

    Rank is defined as (number of negatives scoring strictly above the positive)
    + 1, so the best possible rank is 1.
    """
    n_groups = len(y_true) // group_size
    mrr_total = 0.0
    h1 = h5 = h10 = 0
    for i in range(n_groups):
        g_true = y_true[i * group_size : (i + 1) * group_size]
        g_scores = y_scores[i * group_size : (i + 1) * group_size]
        pos_idx = int(np.where(g_true == 1)[0][0])
        pos_score = g_scores[pos_idx]
        rank = int(np.sum(g_scores > pos_score)) + 1
        mrr_total += 1.0 / rank
        if rank <= 1:
            h1 += 1
        if rank <= 5:
            h5 += 1
        if rank <= 10:
            h10 += 1
    n = max(n_groups, 1)
    return {
        "mrr": round(mrr_total / n, 4),
        "hits@1": round(h1 / n, 4),
        "hits@5": round(h5 / n, 4),
        "hits@10": round(h10 / n, 4),
    }
