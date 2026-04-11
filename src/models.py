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
    Returns dict with keys: auc, precision, recall, f1, fpr, tpr, feature_importances.
    fpr and tpr are Python lists (JSON-serializable).
    """
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
    }
