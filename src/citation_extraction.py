from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from eyecite import get_citations
from eyecite.models import FullCaseCitation
from tqdm import tqdm

DATA_DIR = Path("data")
_OPINION_ID_RE = re.compile(r"/opinions/(\d+)/")


def extract_citations_from_text(text: str) -> list[str]:
    """Return list of citation strings found in text using eyecite."""
    citations = get_citations(text)
    return [
        str(c.token)
        for c in citations
        if isinstance(c, FullCaseCitation)
    ]


def _extract_opinion_id(uri: str) -> str | None:
    """Extract numeric opinion ID from a CourtListener URI."""
    m = _OPINION_ID_RE.search(uri)
    return m.group(1) if m else None


def build_edge_list(
    df: pd.DataFrame,
    cache_path: Path | None = DATA_DIR / "citations.csv",
) -> pd.DataFrame:
    """
    Build citation edges from the opinions_cited column.
    Each row's opinions_cited is a list of CourtListener URIs.
    Returns DataFrame with columns: source_id, target_id, year.
    Pass cache_path=None to skip caching (useful in tests).
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    # Build a set of known opinion IDs for fast lookup.
    # The id column may be stored as float (e.g. 12345.0), so normalise to int-string.
    def _norm_id(x: object) -> str:
        try:
            return str(int(float(x)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return str(x)

    known_ids: set[str] = {_norm_id(row_id) for row_id in df["id"]}

    edges: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citation edges"):
        source_id = _norm_id(row["id"])
        year = pd.to_datetime(row["date_created"], utc=True).year
        raw = row.get("opinions_cited")
        if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
            cited_uris = []
        else:
            cited_uris = list(raw)
        for uri in cited_uris:
            if not isinstance(uri, str):
                continue
            target_id = _extract_opinion_id(uri)
            if target_id and target_id != source_id and target_id in known_ids:
                edges.append({"source_id": source_id, "target_id": target_id, "year": year})

    edge_df = (
        pd.DataFrame(edges).drop_duplicates()
        if edges
        else pd.DataFrame(columns=["source_id", "target_id", "year"])
    )

    if cache_path is not None:
        DATA_DIR.mkdir(exist_ok=True)
        edge_df.to_csv(cache_path, index=False)

    return edge_df
