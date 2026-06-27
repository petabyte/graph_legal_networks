import numpy as np
from src.models import train_evaluate, ranking_metrics


def test_train_evaluate_rf_returns_all_metrics():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 5))
    y_train = rng.integers(0, 2, 200)
    X_test = rng.standard_normal((100, 5))
    y_test = rng.integers(0, 2, 100)

    results = train_evaluate(X_train, y_train, X_test, y_test, model_name="rf")
    assert "auc" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "fpr" in results
    assert "tpr" in results
    assert 0.0 <= results["auc"] <= 1.0


def test_train_evaluate_lr_returns_all_metrics():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 5))
    y_train = rng.integers(0, 2, 200)
    X_test = rng.standard_normal((100, 5))
    y_test = rng.integers(0, 2, 100)

    results = train_evaluate(X_train, y_train, X_test, y_test, model_name="lr")
    assert "auc" in results
    assert 0.0 <= results["auc"] <= 1.0


def test_train_evaluate_unknown_model_raises():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3))
    y = rng.integers(0, 2, 50)
    try:
        train_evaluate(X, y, X, y, model_name="xgboost")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_train_evaluate_fpr_tpr_are_lists():
    rng = np.random.default_rng(5)
    X_train = rng.standard_normal((100, 4))
    y_train = rng.integers(0, 2, 100)
    X_test = rng.standard_normal((50, 4))
    y_test = rng.integers(0, 2, 50)
    results = train_evaluate(X_train, y_train, X_test, y_test, model_name="rf")
    assert isinstance(results["fpr"], list)
    assert isinstance(results["tpr"], list)
    assert len(results["fpr"]) == len(results["tpr"])


def test_train_evaluate_exposes_clf_and_scaler():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 4))
    y = rng.integers(0, 2, 100)
    res_rf = train_evaluate(X, y, X, y, model_name="rf")
    assert "clf" in res_rf
    assert res_rf["scaler"] is None

    res_lr = train_evaluate(X, y, X, y, model_name="lr")
    assert "clf" in res_lr
    assert res_lr["scaler"] is not None


# --- ranking_metrics ---

def test_ranking_metrics_perfect():
    # Positive always scores highest → rank 1 for every group
    group_size = 5
    n_groups = 10
    y_true = np.array(([1] + [0] * (group_size - 1)) * n_groups)
    y_scores = np.array(([1.0, 0.5, 0.4, 0.3, 0.2]) * n_groups)
    m = ranking_metrics(y_true, y_scores, group_size)
    assert m["mrr"] == 1.0
    assert m["hits@1"] == 1.0
    assert m["hits@5"] == 1.0


def test_ranking_metrics_worst():
    # Positive always scores lowest → rank = group_size
    group_size = 5
    n_groups = 4
    y_true = np.array(([1] + [0] * (group_size - 1)) * n_groups)
    y_scores = np.array(([0.1, 0.9, 0.8, 0.7, 0.6]) * n_groups)
    m = ranking_metrics(y_true, y_scores, group_size)
    # rank 5 → MRR = 0.2, Hits@1 = 0, Hits@5 = 1
    assert abs(m["mrr"] - round(1 / group_size, 4)) < 1e-6
    assert m["hits@1"] == 0.0
    assert m["hits@5"] == 1.0


def test_ranking_metrics_positive_not_first():
    # Positive is last element in group, but scores highest
    group_size = 3
    y_true = np.array([0, 0, 1,  0, 0, 1])
    y_scores = np.array([0.2, 0.3, 0.9,  0.1, 0.4, 0.8])
    m = ranking_metrics(y_true, y_scores, group_size)
    assert m["mrr"] == 1.0
    assert m["hits@1"] == 1.0


def test_ranking_metrics_keys():
    y_true = np.array([1, 0, 0])
    y_scores = np.array([0.9, 0.5, 0.3])
    m = ranking_metrics(y_true, y_scores, group_size=3)
    assert set(m.keys()) == {"mrr", "hits@1", "hits@5", "hits@10"}
