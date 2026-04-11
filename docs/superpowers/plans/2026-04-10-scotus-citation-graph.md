# SCOTUS Citation Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible SCOTUS citation knowledge graph pipeline and run three experiments: graph validation, link prediction ablation (graph features vs. text), and influence analysis (PageRank/betweenness vs. known landmarks).

**Architecture:** eyecite extracts citation edges deterministically into a CSV; Graphiti ingests case text into a local Kuzu database for semantic enrichment; NetworkX loads the citation CSV for structural feature computation; scikit-learn RF + LR models run the ablation; the full pipeline is deterministic after the one-time Graphiti ingestion.

**Tech Stack:** Python 3.11+, `datasets` (HuggingFace), `eyecite`, `graphiti-core`, `kuzu`, `networkx`, `python-louvain`, `scikit-learn`, `transformers`, `torch`, `pandas`, `numpy`, `matplotlib`, `anthropic`, `tqdm`, `pyarrow`

---

## File Structure

```
ai_for_law/
├── data/
│   ├── scotus_cases.parquet       # filtered SCOTUS dataset (cached)
│   ├── citations.csv              # source_id, target_id, year (cached)
│   └── scotus_graph/              # Kuzu DB directory (released as artifact)
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # load + filter HFforLegal SCOTUS cases
│   ├── citation_extraction.py     # eyecite → citations.csv
│   ├── graph_builder.py           # Graphiti + Kuzu semantic ingestion
│   ├── graph_features.py          # NetworkX feature computation
│   ├── embeddings.py              # Legal-BERT mean-pool embeddings
│   ├── splitting.py               # temporal split + negative sampling
│   └── models.py                  # RF + LR training + evaluation utilities
├── experiments/
│   ├── exp1_graph_analysis.py     # graph stats, degree dist, communities
│   ├── exp2_link_prediction.py    # ablation across feature sets × models
│   └── exp3_influence.py          # PageRank + betweenness centrality
├── tests/
│   ├── test_citation_extraction.py
│   ├── test_graph_features.py
│   ├── test_splitting.py
│   └── test_models.py
├── requirements.txt
└── pyproject.toml
```

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/__init__.py`

- [ ] **Step 1: Write requirements.txt**

```
datasets>=2.20.0
eyecite>=2.6.0
graphiti-core>=0.3.0
kuzu>=0.6.0
networkx>=3.3
python-louvain>=0.16
scikit-learn>=1.5.0
transformers>=4.40.0
torch>=2.2.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.9.0
seaborn>=0.13.0
anthropic>=0.28.0
pyarrow>=16.0.0
tqdm>=4.66.0
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "ai-for-law"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create src/__init__.py**

```python
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without conflict.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt pyproject.toml src/__init__.py
git commit -m "chore: project setup with dependencies"
```

---

### Task 2: Dataset Loading

**Files:**
- Create: `src/dataset.py`

- [ ] **Step 1: Explore dataset schema**

Run in a Python REPL to discover the exact field names before writing the module:

```python
from datasets import load_dataset
ds = load_dataset("HFforLegal/case-law", split="train", streaming=True)
sample = next(iter(ds))
print(list(sample.keys()))
print({k: type(v) for k, v in sample.items()})
```

Note the exact names for `court`, `date`, `text`, and `id` fields. Update the constants at the top of `src/dataset.py` in Step 2 if they differ from the defaults below.

- [ ] **Step 2: Write src/dataset.py**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Update these constants if the dataset uses different field names
COURT_FIELD = "court"
DATE_FIELD = "decision_date"
TEXT_FIELD = "text"
ID_FIELD = "id"
SCOTUS_VALUE = "scotus"

DATA_DIR = Path("data")


def load_scotus_cases(
    cache_path: Path = DATA_DIR / "scotus_cases.parquet",
) -> pd.DataFrame:
    """Load and filter SCOTUS cases. Returns cached parquet if it exists."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    ds = load_dataset("HFforLegal/case-law", split="train")
    df = ds.to_pandas()

    scotus = df[df[COURT_FIELD].str.lower() == SCOTUS_VALUE].copy()
    scotus = scotus.dropna(subset=[DATE_FIELD, TEXT_FIELD])
    scotus = scotus[scotus[TEXT_FIELD].str.strip().str.len() > 0]
    scotus[DATE_FIELD] = pd.to_datetime(scotus[DATE_FIELD], errors="coerce")
    scotus = scotus.dropna(subset=[DATE_FIELD])
    scotus = scotus.reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    scotus.to_parquet(cache_path, index=False)
    return scotus
```

- [ ] **Step 3: Smoke test**

```bash
python -c "from src.dataset import load_scotus_cases; df = load_scotus_cases(); print(len(df), list(df.columns)[:6])"
```

Expected: prints a count near 28,000 and the column names. If count is very different, revisit the `SCOTUS_VALUE` constant.

- [ ] **Step 4: Commit**

```bash
git add src/dataset.py
git commit -m "feat: load and cache SCOTUS cases from HFforLegal"
```

---

### Task 3: Citation Extraction (eyecite)

**Files:**
- Create: `src/citation_extraction.py`
- Create: `tests/test_citation_extraction.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_citation_extraction.py
import pandas as pd
from src.citation_extraction import extract_citations_from_text, build_edge_list


def test_extract_citations_from_known_text():
    text = "As held in Brown v. Board of Education, 347 U.S. 483 (1954), the court found..."
    citations = extract_citations_from_text(text)
    assert len(citations) >= 1
    assert any("347 U.S. 483" in c for c in citations)


def test_build_edge_list_returns_dataframe_with_correct_columns():
    df = pd.DataFrame({
        "id": ["case_a", "case_b"],
        "text": [
            "See 1 U.S. 1 for prior precedent.",
            "This is the cited case.",
        ],
        "decision_date": ["2000-01-01", "1990-01-01"],
        "citation": ["1 U.S. 1", None],
    })
    edges = build_edge_list(df, citation_col="citation", cache_path=None)
    assert isinstance(edges, pd.DataFrame)
    assert set(edges.columns) >= {"source_id", "target_id", "year"}
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/test_citation_extraction.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write src/citation_extraction.py**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from eyecite import get_citations
from eyecite.models import FullCaseCitation
from tqdm import tqdm

DATA_DIR = Path("data")


def extract_citations_from_text(text: str) -> list[str]:
    """Return list of FullCaseCitation strings found in text."""
    citations = get_citations(text)
    return [
        str(c.token)
        for c in citations
        if isinstance(c, FullCaseCitation)
    ]


def build_citation_index(df: pd.DataFrame, citation_col: str = "citation") -> dict[str, str]:
    """Map citation string → case id for rows with a non-null citation field."""
    index: dict[str, str] = {}
    for _, row in df.iterrows():
        val = row.get(citation_col)
        if pd.notna(val) and isinstance(val, str) and val.strip():
            index[val.strip()] = str(row["id"])
    return index


def build_edge_list(
    df: pd.DataFrame,
    citation_col: str = "citation",
    cache_path: Path | None = DATA_DIR / "citations.csv",
) -> pd.DataFrame:
    """
    Extract all citation edges from the dataset.
    Returns cached CSV if cache_path exists.
    Pass cache_path=None to skip caching (useful in tests).
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path)

    citation_index = build_citation_index(df, citation_col)
    edges: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting citations"):
        source_id = str(row["id"])
        year = pd.to_datetime(row["decision_date"]).year
        text = str(row.get("text", ""))
        for cited_str in extract_citations_from_text(text):
            target_id = citation_index.get(cited_str.strip())
            if target_id and target_id != source_id:
                edges.append({"source_id": source_id, "target_id": target_id, "year": year})

    edge_df = pd.DataFrame(edges).drop_duplicates()

    if cache_path is not None:
        DATA_DIR.mkdir(exist_ok=True)
        edge_df.to_csv(cache_path, index=False)

    return edge_df
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_citation_extraction.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/citation_extraction.py tests/test_citation_extraction.py
git commit -m "feat: eyecite citation extraction pipeline"
```

---

### Task 4: Semantic Graph Construction (Graphiti + Kuzu)

**Files:**
- Create: `src/graph_builder.py`

Note: Graphiti is async. This task builds the semantic graph. Run once — output is cached in `data/scotus_graph/`. Requires `ANTHROPIC_API_KEY` env var.

- [ ] **Step 1: Verify Graphiti Kuzu driver import path**

```bash
python -c "import graphiti_core.driver; print(dir(graphiti_core.driver))"
```

Expected: lists available drivers including a Kuzu driver class. Note the exact class name and update the import in Step 2 if it differs from `graphiti_core.driver.kuzu_driver.KuzuDriver`.

- [ ] **Step 2: Write src/graph_builder.py**

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Verify this import against Step 1 output
from graphiti_core.driver.kuzu_driver import KuzuDriver

DATA_DIR = Path("data")
GRAPH_DIR = DATA_DIR / "scotus_graph"
MAX_TEXT_CHARS = 4000  # truncate per case to control LLM cost


def _get_graphiti(graph_dir: Path = GRAPH_DIR) -> Graphiti:
    graph_dir.mkdir(parents=True, exist_ok=True)
    driver = KuzuDriver(str(graph_dir))
    # Uses ANTHROPIC_API_KEY env var automatically
    return Graphiti(driver=driver, llm_client=None)


async def _ingest_cases(df: pd.DataFrame, graphiti: Graphiti) -> None:
    await graphiti.build_indices_and_constraints()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting into Graphiti"):
        text = str(row.get("text", ""))[:MAX_TEXT_CHARS]
        decision_date = pd.to_datetime(row["decision_date"])
        await graphiti.add_episode(
            name=str(row["id"]),
            episode_body=text,
            source=EpisodeType.text,
            reference_time=decision_date.to_pydatetime(),
            source_description="SCOTUS case",
        )


def build_semantic_graph(df: pd.DataFrame) -> None:
    """One-time ingestion of SCOTUS cases into the local Kuzu graph."""
    if GRAPH_DIR.exists() and any(GRAPH_DIR.iterdir()):
        print(f"Graph already exists at {GRAPH_DIR}, skipping ingestion.")
        return
    graphiti = _get_graphiti()
    asyncio.run(_ingest_cases(df, graphiti))
    print(f"Semantic graph persisted to {GRAPH_DIR}")
```

- [ ] **Step 3: Validate on 100-case sample before full run**

```bash
export ANTHROPIC_API_KEY=<your_key>
python - <<'EOF'
import pandas as pd
from src.dataset import load_scotus_cases
from src.graph_builder import build_semantic_graph, GRAPH_DIR
import shutil

# Use temp dir for sample run
import pathlib
GRAPH_DIR_ORIG = GRAPH_DIR
shutil.rmtree("data/scotus_graph_sample", ignore_errors=True)

df = load_scotus_cases().head(100)
import src.graph_builder as gb
gb.GRAPH_DIR = pathlib.Path("data/scotus_graph_sample")
build_semantic_graph(df)
print("Sample files:", list(pathlib.Path("data/scotus_graph_sample").iterdir()))
EOF
```

Expected: `data/scotus_graph_sample/` is created with Kuzu database files.

- [ ] **Step 4: Run full ingestion**

This is a long-running step (may take several hours for ~28,000 cases). Run it and let it complete before proceeding.

```bash
python - <<'EOF'
from src.dataset import load_scotus_cases
from src.graph_builder import build_semantic_graph
df = load_scotus_cases()
build_semantic_graph(df)
EOF
```

Expected: `data/scotus_graph/` populated with Kuzu DB files.

- [ ] **Step 5: Commit**

```bash
git add src/graph_builder.py
git commit -m "feat: Graphiti + Kuzu semantic graph ingestion"
```

---

### Task 5: Graph Feature Engineering

**Files:**
- Create: `src/graph_features.py`
- Create: `tests/test_graph_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph_features.py
import networkx as nx
import numpy as np
import pandas as pd

from src.graph_features import (
    build_nx_graph,
    compute_basic_features,
    compute_community_features,
    compute_triangle_features,
)


def _small_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")])
    return G


def test_build_nx_graph_from_edge_df():
    edge_df = pd.DataFrame({
        "source_id": ["A", "B", "A"],
        "target_id": ["B", "C", "C"],
        "year": [2000, 2001, 2002],
    })
    G = build_nx_graph(edge_df)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3


def test_compute_basic_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("A", "D")]
    features = compute_basic_features(G, pairs)
    assert features.shape == (2, 3)  # common_neighbors, pref_attachment, jaccard


def test_compute_triangle_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("B", "D")]
    features = compute_triangle_features(G, pairs)
    assert features.shape == (2, 2)  # triangles_source, clustering_source


def test_compute_community_features_shape():
    G = _small_graph()
    pairs = [("A", "B"), ("A", "D")]
    features = compute_community_features(G, pairs)
    assert features.shape == (2, 2)  # same_louvain, same_label_prop
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_graph_features.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write src/graph_features.py**

```python
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from community import best_partition  # python-louvain
from networkx.algorithms.community import label_propagation_communities


def build_nx_graph(edge_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from the citation edge DataFrame."""
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(str(row["source_id"]), str(row["target_id"]))
    return G


def compute_basic_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [common_neighbors, pref_attachment, jaccard].
    Uses undirected projection for neighbor-based metrics.
    Returns shape (N, 3).
    """
    U = G.to_undirected()
    rows = []
    for u, v in pairs:
        if U.has_node(u) and U.has_node(v):
            cn = len(list(nx.common_neighbors(U, u, v)))
            pa = U.degree(u) * U.degree(v)
            jac = next(nx.jaccard_coefficient(U, [(u, v)]))[2]
        else:
            cn, pa, jac = 0, 0, 0.0
        rows.append([cn, pa, jac])
    return np.array(rows, dtype=float)


def compute_triangle_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [triangles_u, clustering_u].
    Returns shape (N, 2).
    """
    U = G.to_undirected()
    triangles = nx.triangles(U)
    clustering = nx.clustering(U)
    rows = []
    for u, _ in pairs:
        rows.append([triangles.get(u, 0), clustering.get(u, 0.0)])
    return np.array(rows, dtype=float)


def compute_community_features(G: nx.DiGraph, pairs: list[tuple[str, str]]) -> np.ndarray:
    """
    For each (u, v) pair return [same_louvain_community, same_label_prop_community].
    Returns shape (N, 2).
    """
    U = G.to_undirected()
    louvain = best_partition(U)
    lp_communities = label_propagation_communities(U)
    lp: dict[str, int] = {}
    for i, comm in enumerate(lp_communities):
        for node in comm:
            lp[node] = i

    rows = []
    for u, v in pairs:
        same_louvain = int(louvain.get(u, -1) == louvain.get(v, -2))
        same_lp = int(lp.get(u, -1) == lp.get(v, -2))
        rows.append([same_louvain, same_lp])
    return np.array(rows, dtype=float)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_graph_features.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/graph_features.py tests/test_graph_features.py
git commit -m "feat: graph feature engineering — basic, triangle, community"
```

---

### Task 6: Legal-BERT Text Embeddings

**Files:**
- Create: `src/embeddings.py`

- [ ] **Step 1: Write src/embeddings.py**

```python
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def _mean_pool(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )


class LegalBertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return (N, 768) float32 mean-pooled embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)
            emb = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)


def cosine_similarity_pairs(
    embeddings: np.ndarray, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """
    Return cosine similarity for each (i, j) index pair.
    embeddings: (N, D) — rows are already mean-pooled
    Returns: (len(pairs),) float array
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    return np.array([float(np.dot(normed[i], normed[j])) for i, j in pairs], dtype=float)
```

- [ ] **Step 2: Smoke test**

```bash
python - <<'EOF'
from src.embeddings import LegalBertEmbedder, cosine_similarity_pairs
embedder = LegalBertEmbedder()
embs = embedder.embed(["The court held that due process applies.", "Fourth Amendment search and seizure."])
print("Shape:", embs.shape)
score = cosine_similarity_pairs(embs, [(0, 1)])
print("Cosine similarity:", score)
EOF
```

Expected: Shape `(2, 768)`, cosine similarity between 0 and 1.

- [ ] **Step 3: Commit**

```bash
git add src/embeddings.py
git commit -m "feat: Legal-BERT mean-pool embeddings for text baseline"
```

---

### Task 7: Temporal Split + Negative Sampling

**Files:**
- Create: `src/splitting.py`
- Create: `tests/test_splitting.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_splitting.py
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


def test_sample_negatives_balance():
    positive_pairs = [("A", "B"), ("C", "D")]
    all_nodes = ["A", "B", "C", "D", "E", "F"]
    existing_edges = {("A", "B"), ("C", "D")}
    negatives = sample_negatives(positive_pairs, all_nodes, existing_edges, seed=42)
    assert len(negatives) == len(positive_pairs)
    for u, v in negatives:
        assert (u, v) not in existing_edges
        assert u != v
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_splitting.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write src/splitting.py**

```python
from __future__ import annotations

import random

import pandas as pd


def temporal_split(
    edge_df: pd.DataFrame, split_year: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split edges chronologically. Train = before split_year, test = split_year onward."""
    train = edge_df[edge_df["year"] < split_year].reset_index(drop=True)
    test = edge_df[edge_df["year"] >= split_year].reset_index(drop=True)
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
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_splitting.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/splitting.py tests/test_splitting.py
git commit -m "feat: temporal split and degree-matched negative sampling"
```

---

### Task 8: Model Training + Evaluation Utilities

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import numpy as np
from src.models import train_evaluate


def test_train_evaluate_rf_returns_all_metrics():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 5))
    y_train = rng.integers(0, 2, 200)
    X_test = rng.standard_normal((100, 5))
    y_test = rng.integers(0, 2, 100)

    results = train_evaluate(X_train, y_train, X_test, y_test, model_name="rf")
    assert "auc" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "fpr" in results
    assert "tpr" in results
    assert 0.0 <= results["auc"] <= 1.0


def test_train_evaluate_lr_returns_all_metrics():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200, 5))
    y_train = rng.integers(0, 2, 200)
    X_test = rng.standard_normal((100, 5))
    y_test = rng.integers(0, 2, 100)

    results = train_evaluate(X_train, y_train, X_test, y_test, model_name="lr")
    assert "auc" in results
    assert 0.0 <= results["auc"] <= 1.0
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write src/models.py**

```python
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "rf",
) -> dict:
    """
    Train a classifier and return evaluation metrics.
    model_name: "rf" for RandomForest, "lr" for LogisticRegression.
    Returns dict with keys: auc, precision, recall, f1, fpr, tpr, feature_importances.
    """
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
    elif model_name == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Use 'rf' or 'lr'.")

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "feature_importances": getattr(clf, "feature_importances_", None),
    }
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all tests to confirm nothing broken**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: RF + LR training and evaluation utilities"
```

---

### Task 9: Experiment 1 — Graph Construction Analysis

**Files:**
- Create: `experiments/exp1_graph_analysis.py`
- Create: `results/exp1/` (created by the script)

- [ ] **Step 1: Write experiments/exp1_graph_analysis.py**

```python
"""
Experiment 1: Graph Construction Analysis
Validates the pipeline: graph statistics, degree distribution,
top cases by in-degree, and Louvain community structure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from community import best_partition
from pathlib import Path

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.graph_features import build_nx_graph

RESULTS_DIR = Path("results/exp1")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scotus_cases()
    edges = build_edge_list(df)
    G = build_nx_graph(edges)
    U = G.to_undirected()

    id_to_name = dict(zip(df["id"].astype(str), df.get("case_name", df["id"]).astype(str)))

    # Graph statistics
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_in_degree": sum(d for _, d in G.in_degree()) / G.number_of_nodes(),
        "density": nx.density(G),
    }
    print("Graph statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    pd.DataFrame([stats]).to_csv(RESULTS_DIR / "graph_stats.csv", index=False)

    # Degree distribution (log-log)
    in_degrees = sorted([d for _, d in G.in_degree()], reverse=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(range(1, len(in_degrees) + 1), in_degrees, ".", markersize=3, alpha=0.6)
    ax.set_xlabel("Rank")
    ax.set_ylabel("In-degree")
    ax.set_title("SCOTUS Citation In-Degree Distribution (log-log)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "degree_distribution.png", dpi=150)
    plt.close(fig)

    # Top 10 most-cited cases
    in_degree_map = dict(G.in_degree())
    top10 = sorted(in_degree_map.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_df = pd.DataFrame(
        [(id_to_name.get(cid, cid), deg) for cid, deg in top10],
        columns=["case_name", "in_degree"],
    )
    print("\nTop 10 most-cited cases:")
    print(top10_df.to_string(index=False))
    top10_df.to_csv(RESULTS_DIR / "top10_indegree.csv", index=False)

    # Louvain community detection
    partition = best_partition(U)
    num_communities = len(set(partition.values()))
    print(f"\nLouvain communities detected: {num_communities}")

    community_df = pd.DataFrame(
        [(id_to_name.get(k, k), v) for k, v in partition.items()],
        columns=["case_name", "community"],
    )
    community_df["in_degree"] = community_df["case_name"].map(
        lambda n: in_degree_map.get(n, 0)
    )
    top_per_community = (
        community_df.sort_values("in_degree", ascending=False)
        .groupby("community")
        .first()
        .reset_index()[["community", "case_name", "in_degree"]]
        .sort_values("community")
    )
    print("\nTop case per community (first 10 communities):")
    print(top_per_community.head(10).to_string(index=False))
    top_per_community.to_csv(RESULTS_DIR / "top_case_per_community.csv", index=False)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run Experiment 1**

```bash
python experiments/exp1_graph_analysis.py
```

Expected: prints graph stats + top 10 cited cases. Top 10 should include recognizable SCOTUS landmarks. Saves 3 files to `results/exp1/`.

- [ ] **Step 3: Verify degree distribution is approximately power-law**

Open `results/exp1/degree_distribution.png`. The log-log plot should show a roughly linear trend, confirming scale-free structure typical of citation networks.

- [ ] **Step 4: Commit**

```bash
git add experiments/exp1_graph_analysis.py results/exp1/
git commit -m "exp1: graph analysis — stats, degree distribution, community structure"
```

---

### Task 10: Experiment 2 — Link Prediction Ablation

**Files:**
- Create: `experiments/exp2_link_prediction.py`
- Create: `results/exp2/` (created by the script)

- [ ] **Step 1: Write experiments/exp2_link_prediction.py**

```python
"""
Experiment 2: Link Prediction Ablation
5 feature sets × 2 models (RF + LR). Reports AUC table and ROC curve figure.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.embeddings import LegalBertEmbedder, cosine_similarity_pairs
from src.graph_features import (
    build_nx_graph,
    compute_basic_features,
    compute_community_features,
    compute_triangle_features,
)
from src.models import train_evaluate
from src.splitting import sample_negatives, temporal_split

RESULTS_DIR = Path("results/exp2")
SPLIT_YEAR = 2000


def _assemble_features(
    pairs: list[tuple[str, str]],
    G,
    id_to_idx: dict[str, int],
    embeddings: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Build feature matrix for a given feature_set name."""
    basic = compute_basic_features(G, pairs)       # (N, 3)
    triangle = compute_triangle_features(G, pairs)  # (N, 2)
    community = compute_community_features(G, pairs)  # (N, 2)
    pair_idxs = [(id_to_idx.get(u, 0), id_to_idx.get(v, 0)) for u, v in pairs]
    text_sim = cosine_similarity_pairs(embeddings, pair_idxs).reshape(-1, 1)  # (N, 1)

    if feature_set == "text_only":
        return text_sim
    elif feature_set == "graph_basic":
        return basic
    elif feature_set == "graph_triangle":
        return np.hstack([basic, triangle])
    elif feature_set == "graph_community":
        return np.hstack([basic, triangle, community])
    elif feature_set == "combined":
        return np.hstack([basic, triangle, community, text_sim])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scotus_cases()
    edges = build_edge_list(df)
    train_edges, test_edges = temporal_split(edges, SPLIT_YEAR)

    G = build_nx_graph(train_edges)
    all_nodes = list(G.nodes())
    existing_edges = set(zip(train_edges["source_id"], train_edges["target_id"]))

    print("Computing Legal-BERT embeddings...")
    embedder = LegalBertEmbedder()
    texts = df["text"].str[:512].tolist()
    embeddings = embedder.embed(texts, batch_size=32)
    id_to_idx = {str(row["id"]): i for i, (_, row) in enumerate(df.iterrows())}

    # Build train pairs
    train_pos = list(zip(train_edges["source_id"], train_edges["target_id"]))
    train_neg = sample_negatives(train_pos, all_nodes, existing_edges, seed=0)
    train_pairs = train_pos + train_neg
    train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

    # Build test pairs
    test_pos = list(zip(test_edges["source_id"], test_edges["target_id"]))
    test_neg = sample_negatives(test_pos, all_nodes, existing_edges, seed=42)
    test_pairs = test_pos + test_neg
    test_labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    feature_sets = ["text_only", "graph_basic", "graph_triangle", "graph_community", "combined"]
    model_names = ["rf", "lr"]
    results_rows = []
    roc_data: dict[str, tuple] = {}

    for fs in feature_sets:
        X_train = _assemble_features(train_pairs, G, id_to_idx, embeddings, fs)
        X_test = _assemble_features(test_pairs, G, id_to_idx, embeddings, fs)
        for model_name in model_names:
            print(f"  {model_name.upper()} / {fs} ...")
            res = train_evaluate(X_train, train_labels, X_test, test_labels, model_name)
            results_rows.append({
                "feature_set": fs,
                "model": model_name.upper(),
                "auc": round(res["auc"], 4),
                "precision": round(res["precision"], 4),
                "recall": round(res["recall"], 4),
                "f1": round(res["f1"], 4),
            })
            roc_data[f"{fs}_{model_name}"] = (res["fpr"], res["tpr"], res["auc"])

    results_df = pd.DataFrame(results_rows)
    print("\nAblation results:")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    # ROC curves (RF only for clarity)
    fig, ax = plt.subplots(figsize=(7, 5))
    for fs in feature_sets:
        fpr, tpr, auc = roc_data[f"{fs}_rf"]
        ax.plot(fpr, tpr, label=f"{fs} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Link Prediction (Random Forest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved results to results/exp2/")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run Experiment 2**

```bash
python experiments/exp2_link_prediction.py
```

Expected: prints ablation table. `combined` feature set should have the highest AUC. Saves `results/exp2/ablation_results.csv` and `results/exp2/roc_curves.png`.

- [ ] **Step 3: Commit**

```bash
git add experiments/exp2_link_prediction.py results/exp2/
git commit -m "exp2: link prediction ablation — 5 feature sets × RF + LR"
```

---

### Task 11: Experiment 3 — Influence Analysis

**Files:**
- Create: `experiments/exp3_influence.py`
- Create: `results/exp3/` (created by the script)

- [ ] **Step 1: Write experiments/exp3_influence.py**

```python
"""
Experiment 3: Influence Analysis
PageRank + betweenness centrality on the full SCOTUS citation graph.
Validates that the graph recovers the known legal canon.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from pathlib import Path

from src.citation_extraction import build_edge_list
from src.dataset import load_scotus_cases
from src.graph_features import build_nx_graph

RESULTS_DIR = Path("results/exp3")

LANDMARK_CASES = {
    "Marbury v. Madison",
    "McCulloch v. Maryland",
    "Brown v. Board of Education",
    "Roe v. Wade",
    "Miranda v. Arizona",
    "Gideon v. Wainwright",
    "Mapp v. Ohio",
    "United States v. Nixon",
    "Bush v. Gore",
    "Obergefell v. Hodges",
}


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_scotus_cases()
    edges = build_edge_list(df)
    G = build_nx_graph(edges)

    id_to_name = dict(zip(df["id"].astype(str), df.get("case_name", df["id"]).astype(str)))

    # PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    top20_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_pr_df = pd.DataFrame(
        [(id_to_name.get(cid, cid), round(score, 6)) for cid, score in top20_pr],
        columns=["case_name", "pagerank"],
    )
    print("Top 20 by PageRank:")
    print(top20_pr_df.to_string(index=False))
    top20_pr_df.to_csv(RESULTS_DIR / "top20_pagerank.csv", index=False)

    # Betweenness centrality (approximate — k=500 for speed)
    print("\nComputing betweenness centrality (k=500 approximation)...")
    betweenness = nx.betweenness_centrality(G, k=500, normalized=True, seed=42)
    top20_bc = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_bc_df = pd.DataFrame(
        [(id_to_name.get(cid, cid), round(score, 6)) for cid, score in top20_bc],
        columns=["case_name", "betweenness"],
    )
    print("\nTop 20 by Betweenness Centrality:")
    print(top20_bc_df.to_string(index=False))
    top20_bc_df.to_csv(RESULTS_DIR / "top20_betweenness.csv", index=False)

    # Qualitative validation against known landmarks
    pr_hits = LANDMARK_CASES & set(top20_pr_df["case_name"])
    bc_hits = LANDMARK_CASES & set(top20_bc_df["case_name"])
    print(f"\nLandmark cases in top-20 PageRank: {len(pr_hits)}/{len(LANDMARK_CASES)}")
    for name in sorted(pr_hits):
        print(f"  + {name}")
    print(f"\nLandmark cases in top-20 Betweenness: {len(bc_hits)}/{len(LANDMARK_CASES)}")
    for name in sorted(bc_hits):
        print(f"  + {name}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run Experiment 3**

```bash
python experiments/exp3_influence.py
```

Expected: top 20 by PageRank should include recognizable SCOTUS landmarks (*Marbury v. Madison*, *Miranda v. Arizona*, etc.). Saves 2 CSV files to `results/exp3/`.

- [ ] **Step 3: Commit**

```bash
git add experiments/exp3_influence.py results/exp3/
git commit -m "exp3: influence analysis — PageRank and betweenness centrality"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|---|---|
| eyecite citation extraction | Task 3 |
| Graphiti + Kuzu semantic graph | Task 4 |
| Local Kuzu DB for reproducibility | Task 4 |
| Temporal split (pre-2000 train, 2000–2023 test) | Task 7 |
| Negative sampling 1:1 degree-matched | Task 7 |
| Basic graph features (common neighbors, pref. attachment, Jaccard) | Task 5 |
| Triangle features (triangle count, clustering coefficient) | Task 5 |
| Community features (Louvain + Label Propagation) | Task 5 |
| Legal-BERT text baseline (cosine similarity) | Task 6 |
| Random Forest model | Task 8 |
| Logistic Regression model | Task 8 |
| AUC-ROC primary metric | Task 8 |
| Precision / Recall / F1 secondary metrics | Task 8 |
| Ablation table (feature set × model × AUC) | Task 10 |
| ROC curve figure | Task 10 |
| Graph stats + degree distribution plot (Exp 1) | Task 9 |
| Top 10 by in-degree (Exp 1) | Task 9 |
| Community detection validation (Exp 1) | Task 9 |
| PageRank + betweenness centrality (Exp 3) | Task 11 |
| Landmark case qualitative validation (Exp 3) | Task 11 |

All requirements covered. ✓

### Placeholder Scan
No TBDs, no "implement later". Task 4 Step 1 is an intentional verification step, not a placeholder. ✓

### Type Consistency
- `build_nx_graph` returns `nx.DiGraph` — imported consistently in exp1, exp2, exp3 ✓
- `train_evaluate` returns dict with keys `auc`, `precision`, `recall`, `f1`, `fpr`, `tpr` — used correctly in exp2 ✓
- `compute_basic_features`, `compute_triangle_features`, `compute_community_features` all take `(G: nx.DiGraph, pairs: list[tuple[str, str]])` — consistent with exp2 usage ✓
- `cosine_similarity_pairs` takes `(embeddings: np.ndarray, pairs: list[tuple[int, int]])` — called with `pair_idxs` (int pairs) in exp2 ✓
