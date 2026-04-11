import numpy as np
from src.models import train_evaluate


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
