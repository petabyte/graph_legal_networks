from __future__ import annotations

import random

import pandas as pd


def temporal_split(
    edge_df: pd.DataFrame, split_year: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split edges chronologically.
    Train = edges with year < split_year (including None years).
    Test = edges with year >= split_year.
    """
    year_col = pd.to_numeric(edge_df["year"], errors="coerce")
    test_mask = year_col >= split_year
    train = edge_df[~test_mask].reset_index(drop=True)
    test = edge_df[test_mask].reset_index(drop=True)
    return train, test


def random_split(
    edge_df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random 80/20 train/test split.
    Used when reliable decision dates are unavailable.
    """
    shuffled = edge_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_test = int(len(shuffled) * test_frac)
    test = shuffled.iloc[:n_test].reset_index(drop=True)
    train = shuffled.iloc[n_test:].reset_index(drop=True)
    return train, test


def sample_negatives(
    positive_pairs: list[tuple[str, str]],
    all_nodes: list[str],
    existing_edges: set[tuple[str, str]],
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    For each positive pair (u, v), sample one negative pair (u, w) where
    w != u and (u, w) is not in existing_edges.
    Returns a list of negative pairs the same length as positive_pairs.
    """
    rng = random.Random(seed)
    negatives: list[tuple[str, str]] = []
    for u, _ in positive_pairs:
        candidates = [n for n in all_nodes if n != u and (u, n) not in existing_edges]
        if not candidates:
            candidates = [n for n in all_nodes if n != u]
        negatives.append((u, rng.choice(candidates)))
    return negatives


def sample_hard_negatives(
    positive_pairs: list[tuple[str, str]],
    community: dict[str, int],
    all_nodes: list[str],
    existing_edges: set[tuple[str, str]],
    seed: int = 42,
    hard_frac: float = 0.5,
) -> list[tuple[str, str]]:
    """
    For each positive (u, v), sample one negative (u, w).
    With probability hard_frac, w is drawn from the same Louvain community as v
    (hard negative); otherwise w is random.  Falls back to random when no
    same-community non-edge exists.
    """
    rng = random.Random(seed)
    negatives: list[tuple[str, str]] = []
    for u, v in positive_pairs:
        hard_candidates: list[str] = []
        if rng.random() < hard_frac:
            v_comm = community.get(v, -1)
            hard_candidates = [
                n for n in all_nodes
                if n != u and community.get(n, -2) == v_comm and (u, n) not in existing_edges
            ]
        if hard_candidates:
            negatives.append((u, rng.choice(hard_candidates)))
        else:
            rand_candidates = [n for n in all_nodes if n != u and (u, n) not in existing_edges]
            if not rand_candidates:
                rand_candidates = [n for n in all_nodes if n != u]
            negatives.append((u, rng.choice(rand_candidates)))
    return negatives


def sample_negatives_ranked(
    positive_pairs: list[tuple[str, str]],
    community: dict[str, int],
    all_nodes: list[str],
    existing_edges: set[tuple[str, str]],
    n_neg: int = 10,
    seed: int = 42,
) -> list[list[tuple[str, str]]]:
    """
    For each positive (u, v), sample n_neg unique negatives without replacement.
    Same-community candidates (hard negatives) are prioritised; any remaining
    slots are filled with random non-edges.
    Returns a list of lists: result[i] contains n_neg pairs for positive_pairs[i].
    """
    rng = random.Random(seed)
    result: list[list[tuple[str, str]]] = []
    for u, v in positive_pairs:
        v_comm = community.get(v, -1)
        hard_pool = [
            n for n in all_nodes
            if n != u and community.get(n, -2) == v_comm and (u, n) not in existing_edges
        ]
        rand_pool = [
            n for n in all_nodes
            if n != u and (u, n) not in existing_edges
        ]
        rng.shuffle(hard_pool)
        rng.shuffle(rand_pool)
        seen: set[str] = set()
        negs: list[tuple[str, str]] = []
        for pool in (hard_pool, rand_pool):
            for n in pool:
                if len(negs) >= n_neg:
                    break
                if n not in seen:
                    negs.append((u, n))
                    seen.add(n)
            if len(negs) >= n_neg:
                break
        result.append(negs)
    return result
