"""Parallel Graphiti ingestion of SCOTUS cases into per-shard Kuzu DBs.

Strategy:
  - Split all cases into N_SHARDS chunks
  - Each shard runs in its own thread with its own asyncio event loop and Kuzu DB
    (Kuzu requires a single writer per DB; shards are fully isolated)
  - Resume-safe: checks existing Episodic node count per shard at startup
  - After ingestion: extracts triples from all shards into data/triples.jsonl
    Format: {case_id: [{s, p, o, fact}, ...]} per line

Usage:
  python3.12 scripts/graphiti_parallel.py [--shards N]
  python3.12 scripts/graphiti_parallel.py --extract-only [--shards N]
  python3.12 scripts/graphiti_parallel.py --status [--shards N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

DATA_DIR = PROJECT_ROOT / "data"
SHARDS_DIR = DATA_DIR / "shards"
TRIPLES_FILE = DATA_DIR / "triples.jsonl"
CRASH_SENTINEL = DATA_DIR / "ingest_last_case.txt"   # overwritten before each add_episode
SKIP_FILE = DATA_DIR / "ingest_skip.txt"             # one case_id per line to skip


# ---------------------------------------------------------------------------
# Kuzu helpers
# ---------------------------------------------------------------------------

def _count_episodes(graph_dir: Path) -> int:
    """Count Episodic nodes in a Kuzu shard. Returns 0 if shard is missing."""
    if not graph_dir.exists():
        return 0
    try:
        import kuzu
        db = kuzu.Database(str(graph_dir))
        conn = kuzu.Connection(db)
        r = conn.execute("MATCH (e:Episodic) RETURN count(e)")
        count = r.get_next()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _get_ingested_names(graph_dir: Path) -> set[str]:
    """Return set of episode names (case IDs) already in the shard."""
    if not graph_dir.exists():
        return set()
    try:
        import kuzu
        db = kuzu.Database(str(graph_dir))
        conn = kuzu.Connection(db)
        r = conn.execute("MATCH (e:Episodic) RETURN e.name")
        names = set()
        while r.has_next():
            names.add(str(r.get_next()[0]))
        conn.close()
        return names
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Per-shard ingestion (runs in its own thread / event loop)
# ---------------------------------------------------------------------------

def _load_skip_set() -> set[str]:
    """Load case IDs to skip from SKIP_FILE (one ID per line)."""
    if not SKIP_FILE.exists():
        return set()
    with open(SKIP_FILE) as f:
        return {line.strip() for line in f if line.strip()}


async def _ingest_shard_async(shard_id: int, chunk: pd.DataFrame) -> None:
    """Async ingestion of one shard. Each case is one Graphiti episode."""
    from graphiti_core.nodes import EpisodeType
    from src.graph_builder import _get_graphiti, _strip_html, MAX_TEXT_CHARS

    graph_dir = SHARDS_DIR / f"shard_{shard_id}" / "kuzu.db"
    graph_dir.parent.mkdir(parents=True, exist_ok=True)

    skip_set = _load_skip_set()
    if skip_set:
        tqdm.write(f"[shard {shard_id}] skipping {len(skip_set)} known-bad case(s): {skip_set}")

    ingested = _get_ingested_names(graph_dir)
    todo = chunk[~chunk["id"].apply(
        lambda x: str(int(float(x)))
    ).isin(ingested | skip_set)]

    if todo.empty:
        tqdm.write(f"[shard {shard_id}] complete ({len(ingested)} episodes already ingested)")
        return

    tqdm.write(f"[shard {shard_id}] {len(ingested)} done, {len(todo)} remaining")

    graphiti = _get_graphiti(graph_dir)
    await graphiti.build_indices_and_constraints()

    failed = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo),
                       desc=f"shard {shard_id}", position=shard_id, leave=True):
        case_id = str(int(float(row["id"])))
        text = _strip_html(str(row.get("html_with_citations", "")))[:MAX_TEXT_CHARS]
        if not text.strip():
            continue
        ts = pd.to_datetime(row["date_created"], utc=True)
        if pd.isnull(ts):
            continue
        # Write sentinel before the call — if a segfault occurs, this file
        # identifies the offending case so it can be added to SKIP_FILE.
        CRASH_SENTINEL.write_text(f"shard={shard_id} case_id={case_id}\n")
        try:
            await graphiti.add_episode(
                name=case_id,
                episode_body=text,
                source=EpisodeType.text,
                reference_time=ts.to_pydatetime(),
                source_description="SCOTUS opinion",
            )
        except Exception as exc:
            failed += 1
            tqdm.write(f"[shard {shard_id}] skip {case_id}: {type(exc).__name__}: {exc}")

    CRASH_SENTINEL.unlink(missing_ok=True)
    tqdm.write(f"[shard {shard_id}] finished. {failed} failed.")


def _run_shard(shard_id: int, chunk: pd.DataFrame) -> None:
    """Entry point for each thread — creates its own event loop."""
    asyncio.run(_ingest_shard_async(shard_id, chunk))


# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

def extract_triples(n_shards: int) -> None:
    """Query all shards for (ep, s, p, o, fact) triples → data/triples.jsonl.

    Schema (confirmed from existing 5-case kuzu.db):
      (Episodic)-[:MENTIONS]->(Entity)-[:RELATES_TO]->(RelatesToNode_)<-[:RELATES_TO]-(Entity)
    """
    import kuzu

    all_triples: dict[str, list[dict]] = {}

    for shard_id in range(n_shards):
        graph_dir = SHARDS_DIR / f"shard_{shard_id}" / "kuzu.db"
        if not graph_dir.exists():
            print(f"[shard {shard_id}] not found, skipping")
            continue

        n_ep = _count_episodes(graph_dir)
        if n_ep == 0:
            print(f"[shard {shard_id}] empty, skipping")
            continue

        print(f"[shard {shard_id}] extracting triples from {n_ep} episodes...")
        db = kuzu.Database(str(graph_dir))
        conn = kuzu.Connection(db)
        try:
            r = conn.execute("""
                MATCH (ep:Episodic)-[:MENTIONS]->(e1:Entity)
                      -[:RELATES_TO]->(rn:RelatesToNode_)
                      <-[:RELATES_TO]-(e2:Entity)
                RETURN ep.name, e1.name, rn.name, e2.name, rn.fact
            """)
            shard_count = 0
            while r.has_next():
                row = r.get_next()
                case_id = str(row[0])
                triple = {
                    "s": row[1],   # source entity name
                    "p": row[2],   # predicate (relation type)
                    "o": row[3],   # object entity name
                    "f": str(row[4]) if row[4] else "",  # fact sentence
                }
                all_triples.setdefault(case_id, []).append(triple)
                shard_count += 1
            print(f"[shard {shard_id}] {shard_count} triples extracted")
        except Exception as e:
            print(f"[shard {shard_id}] extraction error: {e}")
        finally:
            conn.close()

    TRIPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRIPLES_FILE, "w") as f:
        for case_id, triples in all_triples.items():
            f.write(json.dumps({case_id: triples}) + "\n")

    total = sum(len(v) for v in all_triples.values())
    print(f"\nSaved {total} triples across {len(all_triples)} cases → {TRIPLES_FILE}")


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def print_status(n_shards: int, df: pd.DataFrame) -> None:
    chunk_sizes = [len(df.iloc[i::n_shards]) for i in range(n_shards)]
    total_ingested = 0
    for shard_id in range(n_shards):
        graph_dir = SHARDS_DIR / f"shard_{shard_id}" / "kuzu.db"
        n = _count_episodes(graph_dir)
        total_ingested += n
        target = chunk_sizes[shard_id]
        pct = f"{100*n/target:.0f}%" if target else "n/a"
        print(f"  shard {shard_id}: {n:4d}/{target} ({pct})")
    print(f"  total: {total_ingested}/{len(df)} ({100*total_ingested/len(df):.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Graphiti ingestion into sharded Kuzu DBs")
    parser.add_argument("--shards", type=int, default=4,
                        help="Number of shards (default: 4)")
    parser.add_argument("--concurrent", type=int, default=1,
                        help="Max shards running simultaneously (default: 1 — sequential to avoid Ollama contention)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip ingestion, only extract triples from existing shards")
    parser.add_argument("--status", action="store_true",
                        help="Print shard progress and exit")
    args = parser.parse_args()

    from src.dataset import load_scotus_cases
    df = load_scotus_cases()

    if args.status:
        print(f"Dataset: {len(df)} cases, {args.shards} shards")
        print_status(args.shards, df)
        return

    if not args.extract_only:
        print(f"Dataset: {len(df)} cases, {args.shards} shards")
        chunks = [df.iloc[i::args.shards] for i in range(args.shards)]

        mode = "sequential" if args.concurrent == 1 else f"{args.concurrent} concurrent"
        print(f"Starting sharded ingestion ({mode})...")
        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = {
                executor.submit(_run_shard, i, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    future.result()
                    print(f"[shard {shard_id}] thread finished OK")
                except Exception as exc:
                    print(f"[shard {shard_id}] thread FAILED: {exc}")

        print("\nAll shards complete. Final status:")
        print_status(args.shards, df)

    print("\nExtracting triples from all shards...")
    extract_triples(args.shards)


if __name__ == "__main__":
    main()
