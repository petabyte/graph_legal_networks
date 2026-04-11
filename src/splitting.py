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
            # Fallback: any different node
            candidates = [n for n in all_nodes if n != u]
        negatives.append((u, rng.choice(candidates)))
    return negatives
