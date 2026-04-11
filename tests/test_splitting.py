import pandas as pd
from src.splitting import temporal_split, sample_negatives


def _make_edge_df() -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": ["A", "B", "C", "D", "E"],
        "target_id": ["B", "C", "D", "E", "A"],
        "year": [1995, 1998, 2001, 2005, 2010],
    })


def test_temporal_split_correct_years():
    edges = _make_edge_df()
    train, test = temporal_split(edges, split_year=2000)
    assert (train["year"] < 2000).all()
    assert (test["year"] >= 2000).all()


def test_temporal_split_no_overlap():
    edges = _make_edge_df()
    train, test = temporal_split(edges, split_year=2000)
    train_pairs = set(zip(train["source_id"], train["target_id"]))
    test_pairs = set(zip(test["source_id"], test["target_id"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_temporal_split_none_years_go_to_train():
    edges = pd.DataFrame({
        "source_id": ["A", "B"],
        "target_id": ["B", "C"],
        "year": [None, 2005],
    })
    train, test = temporal_split(edges, split_year=2000)
    # None year should not appear in test
    assert len(test) == 1
    assert test.iloc[0]["source_id"] == "B"


def test_sample_negatives_balance():
    positive_pairs = [("A", "B"), ("C", "D")]
    all_nodes = ["A", "B", "C", "D", "E", "F"]
    existing_edges = {("A", "B"), ("C", "D")}
    negatives = sample_negatives(positive_pairs, all_nodes, existing_edges, seed=42)
    assert len(negatives) == len(positive_pairs)
    for u, v in negatives:
        assert (u, v) not in existing_edges
        assert u != v


def test_sample_negatives_are_not_positive():
    positive_pairs = [("A", "B"), ("A", "C"), ("A", "D")]
    all_nodes = ["A", "B", "C", "D", "E"]
    existing_edges = {("A", "B"), ("A", "C"), ("A", "D")}
    negatives = sample_negatives(positive_pairs, all_nodes, existing_edges, seed=0)
    assert len(negatives) == 3
    for u, v in negatives:
        assert (u, v) not in existing_edges


def test_sample_negatives_reproducible():
    positive_pairs = [("A", "B")]
    all_nodes = ["A", "B", "C", "D"]
    existing_edges = {("A", "B")}
    n1 = sample_negatives(positive_pairs, all_nodes, existing_edges, seed=7)
    n2 = sample_negatives(positive_pairs, all_nodes, existing_edges, seed=7)
    assert n1 == n2
