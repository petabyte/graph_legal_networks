import pandas as pd
from src.splitting import temporal_split, sample_negatives, random_split, sample_hard_negatives, sample_negatives_ranked


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


def test_random_split_sizes():
    edges = _make_edge_df()
    train, test = random_split(edges, test_frac=0.2, seed=42)
    assert len(train) + len(test) == len(edges)
    assert len(test) == 1  # 20% of 5 = 1


def test_random_split_no_overlap():
    edges = _make_edge_df()
    train, test = random_split(edges, test_frac=0.4, seed=0)
    train_pairs = set(zip(train["source_id"], train["target_id"]))
    test_pairs = set(zip(test["source_id"], test["target_id"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_random_split_reproducible():
    edges = _make_edge_df()
    t1_train, t1_test = random_split(edges, seed=99)
    t2_train, t2_test = random_split(edges, seed=99)
    assert list(t1_train["source_id"]) == list(t2_train["source_id"])
    assert list(t1_test["source_id"]) == list(t2_test["source_id"])


# --- sample_hard_negatives ---

def _community() -> dict[str, int]:
    # A, B in community 0; C, D, E, F in community 1
    return {"A": 0, "B": 0, "C": 1, "D": 1, "E": 1, "F": 1}


def test_sample_hard_negatives_length():
    community = _community()
    positive_pairs = [("A", "C"), ("B", "D")]
    all_nodes = list(community.keys())
    existing_edges: set[tuple[str, str]] = set()
    result = sample_hard_negatives(positive_pairs, community, all_nodes, existing_edges, seed=0)
    assert len(result) == len(positive_pairs)


def test_sample_hard_negatives_no_existing_edge():
    community = _community()
    positive_pairs = [("A", "C")]
    all_nodes = list(community.keys())
    existing_edges = {("A", "C")}
    result = sample_hard_negatives(positive_pairs, community, all_nodes, existing_edges, seed=1, hard_frac=1.0)
    for u, w in result:
        assert (u, w) not in existing_edges


def test_sample_hard_negatives_hard_frac_one_prefers_same_community():
    community = _community()
    # Positive target is C (community 1); hard neg should be from {D, E, F}
    positive_pairs = [("A", "C")]
    all_nodes = list(community.keys())
    existing_edges = {("A", "C")}
    results = [
        sample_hard_negatives(positive_pairs, community, all_nodes, existing_edges, seed=s, hard_frac=1.0)[0]
        for s in range(20)
    ]
    # With hard_frac=1.0, all negatives should come from community 1 (same as C)
    for u, w in results:
        assert community.get(w) == 1, f"Expected community-1 hard neg, got {w} (community {community.get(w)})"


def test_sample_hard_negatives_reproducible():
    community = _community()
    positive_pairs = [("A", "C"), ("B", "D")]
    all_nodes = list(community.keys())
    existing_edges: set[tuple[str, str]] = set()
    r1 = sample_hard_negatives(positive_pairs, community, all_nodes, existing_edges, seed=7)
    r2 = sample_hard_negatives(positive_pairs, community, all_nodes, existing_edges, seed=7)
    assert r1 == r2


# --- sample_negatives_ranked ---

def test_sample_negatives_ranked_length():
    community = _community()
    positive_pairs = [("A", "C"), ("B", "D")]
    all_nodes = list(community.keys())
    existing_edges: set[tuple[str, str]] = set()
    result = sample_negatives_ranked(positive_pairs, community, all_nodes, existing_edges, n_neg=3, seed=0)
    assert len(result) == len(positive_pairs)
    for neg_list in result:
        assert len(neg_list) == 3


def test_sample_negatives_ranked_no_duplicates_within_group():
    community = _community()
    positive_pairs = [("A", "C")]
    all_nodes = list(community.keys())
    existing_edges: set[tuple[str, str]] = set()
    result = sample_negatives_ranked(positive_pairs, community, all_nodes, existing_edges, n_neg=4, seed=0)
    targets = [w for _, w in result[0]]
    assert len(targets) == len(set(targets)), "Duplicates found within ranking group"


def test_sample_negatives_ranked_no_existing_edge():
    community = _community()
    positive_pairs = [("A", "C"), ("A", "D")]
    all_nodes = list(community.keys())
    existing_edges = {("A", "C"), ("A", "D")}
    result = sample_negatives_ranked(positive_pairs, community, all_nodes, existing_edges, n_neg=2, seed=0)
    for neg_list in result:
        for u, w in neg_list:
            assert (u, w) not in existing_edges
