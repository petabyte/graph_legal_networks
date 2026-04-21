from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
MENTIONS_FILE = DATA_DIR / "entity_mentions.jsonl"


def load_entity_mentions(mentions_file: Path = MENTIONS_FILE) -> dict[str, set[str]]:
    """Load entity mentions from the JSONL file produced by ingest_parallel.py.

    Returns a dict mapping case_id (str) → set of entity name strings (lower-cased).
    Falls back to empty dict if file does not exist.
    """
    if not mentions_file.exists():
        return {}

    mentions: dict[str, set[str]] = {}
    with open(mentions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for case_id, entities in obj.items():
                if entities:
                    mentions[str(case_id)] = set(str(e).lower() for e in entities)
    return mentions


def compute_entity_overlap(
    mentions: dict[str, set[str]],
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    """Compute entity-based features for each (source, target) pair.

    Returns (N, 2) float array with columns:
      [0] entity_jaccard   — |intersection| / |union| of entity sets
      [1] entity_common    — |intersection| (raw count of shared entities)

    Falls back to zeros for pairs where either case has no entity mentions.
    """
    n = len(pairs)
    out = np.zeros((n, 2), dtype=float)

    for i, (u, v) in enumerate(pairs):
        u_ents = mentions.get(u, set())
        v_ents = mentions.get(v, set())
        if not u_ents or not v_ents:
            continue
        intersection = len(u_ents & v_ents)
        union = len(u_ents | v_ents)
        out[i, 0] = intersection / union if union > 0 else 0.0
        out[i, 1] = float(intersection)

    return out
