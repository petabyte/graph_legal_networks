# SCOTUS Citation Graph — Experiment Design

**Target venue:** ICML Workshop on AI for Law  
**Format:** 8-page workshop paper  
**Date:** 2026-04-10

---

## Hypothesis

By processing the unstructured text of the HFforLegal/case-law SCOTUS subset through a two-layer graph construction pipeline (deterministic citation extraction + LLM-powered semantic enrichment), and subsequently extracting connected graph features (preferential attachment, triangle counts, community partitions), machine learning models can more accurately predict missing legal citations (link prediction) and identify influential case law than models relying on text embeddings alone.

---

## Pipeline Architecture

```
HFforLegal/case-law (SCOTUS subset)
        │
        ▼
[Stage 1] eyecite extraction
        → directed citation edges: (source_id, target_id, year) as CSV
        │
        ▼
[Stage 2] Graphiti + Kuzu ingestion
        → semantic nodes: doctrines, statutes, constitutional provisions, key parties
        → semantic edges enriching the citation backbone
        → persisted to local Kuzu DB at data/scotus_graph/
        │
        ▼
[Stage 3] Feature engineering
        → graph features queried from Kuzu
        → text features from Legal-BERT embeddings
        │
        ▼
[Stage 4] Temporal split + negative sampling
        → train: decisions before 2000
        → test: decisions 2000–2023
        → 1:1 negative sampling matched on degree
        │
        ▼
[Stage 5] Models: Random Forest + Logistic Regression
        │
        ▼
[Stage 6] Evaluation: AUC, Precision, Recall, F1
        + PageRank / betweenness for influence analysis
```

**Reproducibility:** The Kuzu DB is a local directory released as a dataset artifact. All stages after Stage 2 are deterministic and require no API calls.

---

## Dataset

- **Source:** `HFforLegal/case-law`, SCOTUS subset
- **Filter:** Cases with a known decision date and at least one outbound citation
- **Expected size:** ~28,000 cases after filtering
- **Temporal split:** Train = pre-2000 (~80%), Test = 2000–2023 (~20%)
- **Negative sampling:** For each positive citation pair in the test set, sample one non-citing pair from the same era, matched on degree to avoid trivial negatives. Final class balance: 1:1.

---

## Graph Construction

### Layer 1 — Citation edges (eyecite)
- Run eyecite over each case's full text
- Each resolved citation → directed edge `(citing_case) → (cited_case)`
- Discard unresolved citations (eyecite flags these automatically)
- Output: edge list CSV `(source_id, target_id, year)`

### Layer 2 — Semantic enrichment (Graphiti + Kuzu)
- Feed each case's text into Graphiti as a timestamped episode (decision date)
- Graphiti extracts semantic nodes: legal doctrines, constitutional provisions, statutes, parties
- Semantic edges connect cases sharing extracted concepts
- Persisted to `data/scotus_graph/` (local Kuzu database)
- LLM cost is one-time and bounded; graph is cached after initial build

---

## Experiment 1: Graph Construction Analysis

**Purpose:** Validate that the pipeline produces a legally meaningful graph.

**Metrics to compute:**
- Node count, edge count, average degree, graph density
- Degree distribution (expected power-law)
- Top 10 cases by in-degree (expected: landmark cases)
- Leiden community detection: number of communities, top cases per community (expected: communities align with legal domains — First Amendment, Commerce Clause, etc.)

**Paper output:**
- 1 table of graph statistics
- 1 figure: degree distribution or community visualization
- ~1 page

---

## Experiment 2: Link Prediction with Ablation

**Purpose:** Test whether graph features predict citations better than text embeddings alone.

### Feature Sets

| Feature Set | Contents |
|---|---|
| Text-only | Cosine similarity of Legal-BERT sentence embeddings |
| Graph-basic | Common neighbors, preferential attachment, Jaccard coefficient |
| Graph-triangle | Graph-basic + triangle count, local clustering coefficient |
| Graph-community | Graph-triangle + community membership (Louvain + Label Propagation) |
| Combined | All graph features + text similarity score |

### Models
- Random Forest (ensemble, non-linear, interpretable via feature importance)
- Logistic Regression (linear, shows features generalize across model classes)

### Metrics
- **Primary:** AUC-ROC
- **Secondary:** Precision, Recall, F1

### Paper output
- 1 results table: feature set × model × AUC/F1
- 1 ROC curve figure: comparing feature sets
- ~1.5 pages

---

## Experiment 3: Influence Analysis

**Purpose:** Demonstrate the graph recovers the legal canon, grounding the paper for the law-facing workshop audience.

**Metrics to compute:**
- PageRank on the full citation graph
- Betweenness centrality (identifies "bridge" cases between legal communities)
- Top 20 cases by each metric
- Qualitative validation against known SCOTUS landmark cases

**Paper output:**
- 1 table: top 20 cases by PageRank + betweenness centrality
- Short qualitative discussion validating alignment with legal canon
- ~0.75 pages

---

## Positioning and Contribution Framing

**Contribution angle:** Interpretable, low-resource pipeline for legal citation prediction.  
- No GPU required after graph construction
- Practitioners can inspect feature importances (e.g., "triangle count was the strongest predictor")
- Fully reproducible via released Kuzu graph artifact

**Baselines:** Text-only (Legal-BERT cosine similarity). GNNs are acknowledged in limitations/future work — not included to maintain the interpretability framing.

**Narrative:** The graph not only predicts future citations (Experiment 2) but recovers the known legal canon (Experiment 3), demonstrating that structural graph features capture legally meaningful information beyond semantic similarity.

---

## Estimated Page Budget

| Section | Pages |
|---|---|
| Introduction + Related Work | 2.0 |
| Pipeline & Graph Construction | 1.5 |
| Experiment 1: Graph Analysis | 1.0 |
| Experiment 2: Link Prediction | 1.5 |
| Experiment 3: Influence Analysis | 0.75 |
| Discussion + Conclusion | 0.75 |
| **Total** | **7.5** |
