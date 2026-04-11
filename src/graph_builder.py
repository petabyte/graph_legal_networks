from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
# GRAPH_DIR is the path to the Kuzu database file (not a directory).
# Kuzu stores the database as a single file-tree at this path.
GRAPH_DIR = DATA_DIR / "scotus_graph" / "kuzu.db"
MAX_TEXT_CHARS = 4000  # truncate per case to control LLM cost


def _strip_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _make_embedder():
    """Return an embedder client.

    Uses OpenAIEmbedder with OPENAI_API_KEY when set, otherwise returns a stub
    that raises on actual use (suitable for schema-only operations).
    """
    import os
    from graphiti_core.embedder.client import EmbedderClient

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from graphiti_core.embedder.openai import OpenAIEmbedder
        return OpenAIEmbedder()

    # Stub: works for instantiation/schema setup, raises if embeddings requested.
    class _StubEmbedder(EmbedderClient):
        async def create(
            self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
        ) -> list[float]:
            raise RuntimeError(
                "No OPENAI_API_KEY set; cannot generate embeddings. "
                "Set OPENAI_API_KEY before calling build_semantic_graph()."
            )

    return _StubEmbedder()


def _make_cross_encoder():
    """Return a cross-encoder client.

    Returns a stub that raises on actual use (suitable for schema-only operations).
    The cross-encoder is only used during search, not during ingestion.
    """
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class _StubCrossEncoder(CrossEncoderClient):
        async def rank(
            self, query: str, passages: list[str]
        ) -> list[float]:
            raise RuntimeError(
                "StubCrossEncoder: not available for ranking. "
                "Provide a real CrossEncoderClient for search operations."
            )

    return _StubCrossEncoder()


def _get_graphiti(graph_dir: Path = GRAPH_DIR):
    """Create a Graphiti instance backed by a local Kuzu database.

    Args:
        graph_dir: Path to the Kuzu database file (Kuzu creates the file if it
                   does not exist; the parent directory must exist).

    Notes:
        - Reads ANTHROPIC_API_KEY from the environment for the LLM client.
          Instantiation succeeds without the key; actual ingestion requires it.
        - Reads OPENAI_API_KEY for the embedder if set; falls back to a stub
          that raises when embeddings are actually requested.
        - Import paths verified against graphiti-core installed package:
            graphiti_core.driver.kuzu_driver.KuzuDriver
            graphiti_core.llm_client.anthropic_client.AnthropicClient
            graphiti_core.Graphiti  (uses 'graph_driver=' kwarg, not 'driver=')
    """
    graph_dir.parent.mkdir(parents=True, exist_ok=True)
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.anthropic_client import AnthropicClient

    # KuzuDriver accepts a file path for the Kuzu database (not a directory).
    driver = KuzuDriver(str(graph_dir))

    # AnthropicClient reads ANTHROPIC_API_KEY from env; does not raise on
    # missing key at construction time — only fails when API calls are made.
    llm_client = AnthropicClient()

    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=_make_embedder(),
        cross_encoder=_make_cross_encoder(),
    )


async def _ingest_cases(df: pd.DataFrame, graphiti) -> None:
    """Async ingestion of all cases into Graphiti."""
    from graphiti_core.nodes import EpisodeType

    await graphiti.build_indices_and_constraints()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting into Graphiti"):
        text = _strip_html(str(row.get("html_with_citations", "")))[:MAX_TEXT_CHARS]
        if not text.strip():
            continue
        ts = pd.to_datetime(row["date_created"], utc=True)
        if ts is pd.NaT:
            continue
        await graphiti.add_episode(
            name=str(int(float(row["id"]))),
            episode_body=text,
            source=EpisodeType.text,
            reference_time=ts.to_pydatetime(),
            source_description="SCOTUS opinion",
        )


def build_semantic_graph(df: pd.DataFrame, graph_dir: Path = GRAPH_DIR) -> None:
    """
    One-time ingestion of SCOTUS cases into the local Kuzu graph.
    Requires ANTHROPIC_API_KEY env var (used by Graphiti's AnthropicClient).
    Skips if graph_dir already exists.

    Args:
        df: DataFrame returned by load_scotus_cases().
        graph_dir: Path to the Kuzu database file. Defaults to GRAPH_DIR.
    """
    if graph_dir.exists():
        print(f"Graph already exists at {graph_dir}, skipping ingestion.")
        return
    graphiti = _get_graphiti(graph_dir)
    asyncio.run(_ingest_cases(df, graphiti))
    print(f"Semantic graph persisted to {graph_dir}")
