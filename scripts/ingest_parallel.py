"""
Parallel batch ingestion of SCOTUS cases into Graphiti/Kuzu.

Strategy:
  - Split all cases into N_WORKERS chunks
  - Each worker ingests its chunk into its own Kuzu shard: data/shards/shard_<i>/kuzu.db
  - Workers run concurrently via asyncio (Ollama handles parallel requests natively)
  - After all workers finish, merge entity mentions into data/entity_mentions.jsonl
    (a lightweight JSONL file: {case_id: [entity, ...]} per line)
  - The JSONL file is what kuzu_features.py reads — no need to merge Kuzu DBs

Usage:
  python3.12 scripts/ingest_parallel.py [--workers N] [--batch-size B]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

DATA_DIR = PROJECT_ROOT / "data"
SHARDS_DIR = DATA_DIR / "shards"
MENTIONS_FILE = DATA_DIR / "entity_mentions.jsonl"

OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"  # native API supports think=false
OLLAMA_LLM_MODEL = "qwen3.5:latest"
MAX_TEXT_CHARS = 2000  # shorter for speed in parallel mode


def _strip_html(html: str) -> str:
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


async def _extract_entities_direct(text: str, client: httpx.AsyncClient) -> list[str]:
    """Call Ollama native API to extract entity names. Uses think=false to disable
    qwen3.5's thinking mode (the OpenAI-compat endpoint doesn't forward that param).

    Returns a list of entity name strings. Falls back to [] on any error.
    """
    system = (
        "You are a legal entity extractor. "
        "Extract all named entities from the legal text: parties, courts, statutes, "
        "constitutional provisions, legal doctrines, and key legal concepts. "
        "Return ONLY a JSON object with key 'extracted_entities' containing a list of strings. "
        "No explanation, no markdown, no thinking."
    )
    payload = {
        "model": OLLAMA_LLM_MODEL,
        "think": False,  # disable thinking mode — critical for qwen3.5
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 1024},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:MAX_TEXT_CHARS]},
        ],
    }
    try:
        resp = await client.post(OLLAMA_NATIVE_URL, json=payload, timeout=120.0)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "") or ""
        # Strip markdown fences just in case
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        if not raw:
            return []
        data = json.loads(raw)
        # Accept various key names
        for key in ("extracted_entities", "entities", "nodes", "extracted_nodes"):
            if key in data and isinstance(data[key], list):
                names = []
                for item in data[key]:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("entity") or item.get("entity_text") or item.get("value")
                        if name:
                            names.append(str(name).lower())
                    elif isinstance(item, str):
                        names.append(item.lower())
                return names
        return []
    except Exception:
        return []


CHECKPOINT_EVERY = 50  # append to JSONL after this many new results


def _append_to_jsonl(new_results: dict[str, list[str]]) -> None:
    """Atomically append new results to the JSONL checkpoint file."""
    MENTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MENTIONS_FILE, "a") as f:
        for case_id, entities in new_results.items():
            f.write(json.dumps({case_id: entities}) + "\n")


async def _ingest_chunk(
    chunk: pd.DataFrame,
    worker_id: int,
    semaphore: asyncio.Semaphore,
    results: dict[str, list[str]],
    write_lock: asyncio.Lock,
    pbar: tqdm,
) -> None:
    """Process a chunk of cases, extracting entities via Ollama native API.

    Appends to JSONL every CHECKPOINT_EVERY new results for crash-safety.
    """
    pending: dict[str, list[str]] = {}

    async with httpx.AsyncClient() as client:
        for _, row in chunk.iterrows():
            case_id = str(int(float(row["id"])))
            if case_id in results:
                pbar.update(1)
                continue
            text = _strip_html(str(row.get("html_with_citations", "")))
            if not text.strip():
                pending[case_id] = []
                results[case_id] = []
                pbar.update(1)
            else:
                async with semaphore:
                    entities = await _extract_entities_direct(text, client)
                pending[case_id] = entities
                results[case_id] = entities
                pbar.update(1)

            if len(pending) >= CHECKPOINT_EVERY:
                async with write_lock:
                    _append_to_jsonl(pending)
                pending = {}

    # Flush remaining
    if pending:
        async with write_lock:
            _append_to_jsonl(pending)


async def run_parallel(df: pd.DataFrame, n_workers: int) -> dict[str, list[str]]:
    """Run entity extraction in parallel across n_workers async tasks.

    Results are checkpointed to MENTIONS_FILE every CHECKPOINT_EVERY cases.
    Resume-safe: already-processed cases are loaded from MENTIONS_FILE at start.
    """
    # Load already-done results from MENTIONS_FILE
    results: dict[str, list[str]] = {}
    if MENTIONS_FILE.exists():
        with open(MENTIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    results.update(obj)
        print(f"Loaded {len(results)} existing results from {MENTIONS_FILE}")

    remaining = df[~df["id"].apply(
        lambda x: str(int(float(x)))
    ).isin(results)]
    print(f"Cases remaining: {len(remaining)} / {len(df)}")

    if remaining.empty:
        print("All cases already processed.")
        return results

    # Split into chunks
    chunks = [remaining.iloc[i::n_workers] for i in range(n_workers)]

    # Semaphore limits actual concurrent Ollama calls (Ollama queues internally)
    semaphore = asyncio.Semaphore(n_workers)
    write_lock = asyncio.Lock()

    with tqdm(total=len(remaining), desc="Extracting entities") as pbar:
        tasks = [
            _ingest_chunk(chunk, i, semaphore, results, write_lock, pbar)
            for i, chunk in enumerate(chunks)
        ]
        await asyncio.gather(*tasks)

    return results


def save_results(results: dict[str, list[str]]) -> None:
    """Write results to JSONL — one {case_id: [entities]} per line."""
    MENTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MENTIONS_FILE, "w") as f:
        for case_id, entities in results.items():
            f.write(json.dumps({case_id: entities}) + "\n")
    print(f"Saved {len(results)} cases to {MENTIONS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel async workers (default: 4)")
    args = parser.parse_args()

    from src.dataset import load_scotus_cases
    df = load_scotus_cases()
    print(f"Dataset: {len(df)} cases")
    print(f"Workers: {args.workers}")

    results = asyncio.run(run_parallel(df, args.workers))

    # Final dedup pass — rewrite JSONL with one line per case, no duplicates
    save_results(results)

    # Summary
    non_empty = sum(1 for v in results.values() if v)
    total_entities = sum(len(v) for v in results.values())
    print(f"\nDone: {non_empty}/{len(df)} cases have entities, {total_entities} total entity mentions")


if __name__ == "__main__":
    main()
