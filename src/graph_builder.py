from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
# Kuzu stores the database as a file-tree at this path.
GRAPH_DIR = DATA_DIR / "scotus_graph" / "kuzu.db"
MAX_TEXT_CHARS = 4000  # truncate per case to control LLM cost

# Ollama endpoints
OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"  # native API: supports think=false
OLLAMA_BASE_URL = "http://localhost:11434/v1"          # OpenAI-compat: used for embeddings
OLLAMA_LLM_MODEL = "qwen3.5:latest"  # local 6.6GB model — no rate limits, no cloud dependency
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def _strip_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _get_graphiti(graph_dir: Path = GRAPH_DIR):
    """Create a Graphiti instance backed by a local Kuzu database.

    Uses Ollama for both LLM (qwen3.5:cloud) and embeddings (nomic-embed-text)
    via Ollama's OpenAI-compatible API — no external API keys required.
    """
    graph_dir.parent.mkdir(parents=True, exist_ok=True)

    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from pydantic import BaseModel

    # Common field aliases local models use instead of Graphiti's expected names.
    _FIELD_ALIASES: dict[str, str] = {
        # entity list key aliases
        "entities": "extracted_entities",
        "entity_list": "extracted_entities",
        "nodes": "extracted_entities",
        "extracted_nodes": "extracted_entities",
        # edge list key aliases
        "triples": "edges",
        "relationships": "edges",
        "relations": "edges",
        "edge_list": "edges",
        # entity object field aliases
        "entity": "name",
        "entity_name": "name",
        "entity_text": "name",
        "text": "name",
        "value": "name",
        # edge field aliases
        "type": "relation_type",
        "relationship": "relation_type",
        "source": "source_entity_name",
        "target": "target_entity_name",
        "statement": "fact",
        "description": "fact",
    }

    # Fields that should always contain a list (the list wrapper field name → item fields)
    _LIST_FIELDS: dict[str, set[str]] = {
        "extracted_entities": {"name", "entity_type_id"},
        "entity_resolutions": {"id", "name", "duplicate_name"},
        "edges": {"source_entity_name", "target_entity_name", "relation_type", "fact"},
        "triples": {"source_entity_name", "target_entity_name", "relation_type", "fact"},
        "summaries": {"name", "summary"},
    }

    def _normalize(data: object) -> object:
        """Recursively rename aliased keys in dicts/lists."""
        if isinstance(data, list):
            return [_normalize(item) for item in data]
        if isinstance(data, dict):
            normalized = {
                _FIELD_ALIASES.get(k, k): _normalize(v)
                for k, v in data.items()
            }
            # If a list field contains a bare dict instead of a list, wrap it
            for list_field in _LIST_FIELDS:
                if list_field in normalized and isinstance(normalized[list_field], dict):
                    normalized[list_field] = [normalized[list_field]]
            return normalized
        return data

    class _OllamaLLMClient(OpenAIGenericClient):
        """Wraps OpenAIGenericClient to fix Ollama cloud-model structured-output quirks.

        Cloud models tunnelled through Ollama don't reliably honour json_schema
        response format. They may:
          1. Wrap JSON in markdown fences (```json ... ```)
          2. Return a bare list instead of {"field": [...]}
          3. Use field aliases (e.g. "entity" instead of "name")
          4. Return an empty string for certain prompts

        This subclass overrides _generate_response to:
          - Force json_object format (universally supported)
          - Strip markdown fences before parsing
          - Wrap bare lists under the first field of response_model
          - Normalize aliased field names
          - Return an empty model instance on empty/unrecoverable responses
        """
        async def _generate_response(self, messages, response_model=None, max_tokens=None, model_size=None):
            import re as _re

            ollama_messages = []
            for m in messages:
                content = self._clean_input(m.content)
                if m.role in ('user', 'system'):
                    ollama_messages.append({'role': m.role, 'content': content})

            # Use native Ollama API with think=false — the OpenAI-compat endpoint
            # does not forward that parameter, causing qwen3.5 to return empty content.
            payload = {
                'model': self.model,
                'think': False,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': self.temperature,
                    'num_predict': max_tokens or self.max_tokens,
                },
                'messages': ollama_messages,
            }
            try:
                async with httpx.AsyncClient() as hclient:
                    resp = await hclient.post(OLLAMA_NATIVE_URL, json=payload, timeout=180.0)
                    resp.raise_for_status()
                    raw = resp.json().get('message', {}).get('content', '') or ''
            except Exception:
                raise

            # Strip markdown code fences
            raw = _re.sub(r'^```(?:json)?\s*', '', raw.strip())
            raw = _re.sub(r'\s*```$', '', raw.strip())

            if not raw.strip():
                # Empty response — return a minimal valid structure
                if response_model is not None:
                    first_field = next(iter(response_model.model_fields))
                    return {first_field: []}
                return {}

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Try extracting first JSON object from the string
                m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if m:
                    try:
                        result = json.loads(m.group())
                    except json.JSONDecodeError:
                        result = {}
                else:
                    result = {}

            # Step 1: Wrap bare list under first field name
            if isinstance(result, list) and response_model is not None:
                first_field = next(iter(response_model.model_fields))
                result = {first_field: result}

            # Step 2: Normalize field aliases first (entities→extracted_entities etc.)
            result = _normalize(result)

            # Step 3: If the normalized dict still doesn't match the model's fields,
            # it may be a single item that should be wrapped in a list field.
            if isinstance(result, dict) and response_model is not None:
                model_fields = set(response_model.model_fields.keys())
                if result and not model_fields.intersection(result.keys()):
                    for field, info in response_model.model_fields.items():
                        if 'list' in str(info.annotation).lower():
                            result = {field: [result]}
                            break

            # Step 4: Empty dict fallback
            if result == {} and response_model is not None:
                result = {
                    field: ([] if info.annotation and 'list' in str(info.annotation).lower() else None)
                    for field, info in response_model.model_fields.items()
                }

            return result

    driver = KuzuDriver(str(graph_dir))

    # graphiti-core's KuzuDriver.build_indices_and_constraints is a no-op;
    # the FTS indexes must be created explicitly after setup_schema().
    _fts_queries = [
        "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
        "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
        "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
        "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
    ]
    import kuzu as _kuzu
    _conn = _kuzu.Connection(driver.db)
    for _q in _fts_queries:
        try:
            _conn.execute(_q)
        except Exception:
            pass  # index already exists — safe to ignore
    _conn.close()

    llm_config = LLMConfig(
        api_key="ollama",           # Ollama ignores the key but the field is required
        model=OLLAMA_LLM_MODEL,
        small_model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_tokens=4096,
    )
    llm_client = _OllamaLLMClient(config=llm_config)

    embed_config = OpenAIEmbedderConfig(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
        embedding_model=OLLAMA_EMBED_MODEL,
    )
    embedder = OpenAIEmbedder(config=embed_config)

    # Cross-encoder is only used during search, not ingestion — stub is fine.
    class _StubCrossEncoder(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[float]:
            raise RuntimeError("StubCrossEncoder: not used during ingestion.")

    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=_StubCrossEncoder(),
    )


async def _get_ingested_ids(graph_dir: Path) -> set[str]:
    """Return set of case IDs already present in the Kuzu graph."""
    import kuzu as _kuzu
    db = _kuzu.Database(str(graph_dir))
    conn = _kuzu.Connection(db)
    try:
        result = conn.execute("MATCH (e:Episodic) RETURN e.name")
        ids = set()
        while result.has_next():
            ids.add(str(result.get_next()[0]))
        return ids
    except Exception:
        return set()
    finally:
        conn.close()


async def _ingest_cases(df: pd.DataFrame, graphiti, graph_dir: Path) -> None:
    """Async ingestion of all cases into Graphiti. Resumes from last checkpoint."""
    from graphiti_core.nodes import EpisodeType

    await graphiti.build_indices_and_constraints()

    already_done = await _get_ingested_ids(graph_dir)
    if already_done:
        print(f"  Resuming — {len(already_done)} cases already ingested, skipping.")

    todo = df[~df["id"].astype(str).apply(
        lambda x: str(int(float(x))) if x else x
    ).isin(already_done)]

    failed = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Ingesting into Graphiti"):
        text = _strip_html(str(row.get("html_with_citations", "")))[:MAX_TEXT_CHARS]
        if not text.strip():
            continue
        ts = pd.to_datetime(row["date_created"], utc=True)
        if pd.isnull(ts):
            continue
        try:
            await graphiti.add_episode(
                name=str(int(float(row["id"]))),
                episode_body=text,
                source=EpisodeType.text,
                reference_time=ts.to_pydatetime(),
                source_description="SCOTUS opinion",
            )
        except Exception as exc:
            failed += 1
            # Log but continue — partial ingestion is better than crashing
            tqdm.write(f"[skip] case {row['id']}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"Ingestion complete. {failed} cases failed and were skipped.")


def build_semantic_graph(df: pd.DataFrame, graph_dir: Path = GRAPH_DIR) -> None:
    """
    One-time ingestion of SCOTUS cases into the local Kuzu graph via Graphiti.
    Uses Ollama (qwen3.5:cloud + nomic-embed-text) — no API keys required.
    Skips silently if graph_dir already exists.
    """
    # Only skip if fully ingested (count episodic nodes ≥ len(df))
    if graph_dir.exists():
        import kuzu as _kuzu
        _db = _kuzu.Database(str(graph_dir))
        _conn = _kuzu.Connection(_db)
        try:
            _r = _conn.execute("MATCH (e:Episodic) RETURN count(e)")
            _count = _r.get_next()[0]
        except Exception:
            _count = 0
        finally:
            _conn.close()
        if _count >= len(df):
            print(f"Graph fully ingested ({_count} episodes). Skipping.")
            return
        print(f"Graph exists but incomplete ({_count}/{len(df)} episodes). Resuming...")
    graphiti = _get_graphiti(graph_dir)
    asyncio.run(_ingest_cases(df, graphiti, graph_dir))
    print(f"Semantic graph persisted to {graph_dir}")
