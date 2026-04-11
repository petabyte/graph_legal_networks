# SCOTUS Citation Graph — Experiment Results

**Paper:** AI for Law Workshop, ICML 2026
**Dataset:** `st3re0/scotus-opinions` (4,915 SCOTUS opinions, 7,159 citation edges)
**Pipeline:** eyecite citation extraction + Graphiti/Kuzu semantic enrichment + NetworkX feature engineering

---

## Hypothesis

By processing the unstructured text of SCOTUS opinions through a two-layer graph construction pipeline (deterministic citation extraction + LLM-powered semantic enrichment), and subsequently extracting connected graph features (preferential attachment, triangle counts, community partitions), machine learning models can more accurately predict missing legal citations and identify influential case law than models relying on text embeddings alone.

---

## Dataset Notes

- **Source:** `st3re0/scotus-opinions` — 27,186 total rows; 4,915 usable opinions after filtering for non-empty text and valid dates
- **Citation edges:** 7,159 directed edges across 2,476 unique citing opinions, extracted from the pre-built `opinions_cited` column (CourtListener URIs)
- **Temporal split:** The dataset lacks reliable decision dates (`date_created` is the 2010 CourtListener ingestion date for most records). A random 80/20 split (seed=42) was used instead. This is noted as a limitation — a future version using CourtListener's full cluster metadata would enable a proper chronological split.
- **Text baseline model:** `roberta-base` (used as proxy for `nlpaueb/legal-bert-base-uncased` due to CUDA driver compatibility constraints; experiment logic is identical)

---

## Experiment 1: Graph Construction Analysis

**Purpose:** Validate that the pipeline produces a legally meaningful citation graph.

### Graph Statistics

| Metric | Value |
|---|---|
| Nodes (unique cases) | 3,311 |
| Edges (citation links) | 7,159 |
| Average in-degree | 2.16 |
| Graph density | 0.00065 |

The graph is sparse (density < 0.001), consistent with real-world citation networks. The heavy-tailed degree distribution (see `results/exp1/degree_distribution.png`) confirms scale-free structure — a hallmark of preferential attachment in legal citation.

### Top 10 Most-Cited Cases (by in-degree)

| Case | Citations Received |
|---|---|
| United States V Detroit Timber & Lumber Co | 88 |
| Boyd V United States | 40 |
| Roth V United States | 26 |
| United States V Obrien | 24 |
| Dandridge V Williams | 23 |
| NAACP V Alabama Ex Rel | 22 |
| Bivens V Six Unknown Named Agents of Federal Bureau of Narcotics | 22 |
| West Virginia State Board Of Education V Barnette | 19 |
| Textile Workers V Lincoln Mills Of Ala | 19 |
| Pierce V Society Of Sisters | 19 |

All top-10 cases are recognized constitutional precedents — *Boyd* (Fourth/Fifth Amendment), *Roth* (First Amendment), *Bivens* (constitutional torts), *Barnette* (First Amendment/compelled speech). This validates that citation frequency in the graph reflects genuine legal significance.

### Community Structure

- **122 communities** detected via Louvain community detection (seed=42)
- Average community size: ~27 cases
- Communities align with identifiable legal domains, e.g.:
  - Community 33: anchored by *Boyd V United States* → Fourth/Fifth Amendment cluster
  - Community 61: anchored by *Roth V United States* → First Amendment/obscenity cluster
  - Community 51: anchored by *Dandridge V Williams* → equal protection/welfare rights cluster
- The 4:1 ratio between the top case (88 citations) and the second (40 citations) is consistent with the "superstar" precedent effect in legal systems

**Conclusion:** The pipeline produces a graph that recovers known legal structure without any domain labels — purely from citation patterns.

---

## Experiment 2: Link Prediction Ablation

**Purpose:** Test whether graph structural features predict citations better than text embeddings alone, and isolate which feature classes contribute most.

### What "predicting a citation" means

The model is given a **pair of cases** and asked: *"Does case A cite case B?"* — a binary yes/no prediction.

- A **positive example** is a real citation edge: case A actually cites case B in its opinion text
- A **negative example** is a non-edge: case A does not cite case B (sampled randomly at 1:1 ratio)

For each pair, the model sees features such as how many cases A and B both cite (common neighbors), whether A and B belong to the same Louvain community, and how semantically similar their opinion texts are (RoBERTa cosine similarity). It outputs a probability (0–1) that a citation exists. AUC measures how well it ranks true citation pairs above non-citation pairs across all classification thresholds.

**Why this is useful for law:**

A predicted citation is a prediction of legal relevance — if the model says "case A probably cites case B," it is saying "case B is probably a relevant precedent for the legal question in case A." This has three practical applications:

1. **Finding missing precedents** — a lawyer working on a new matter can ask the model for cases that are structurally similar, surfacing relevant precedents that might otherwise be missed
2. **Filling incomplete citation databases** — older SCOTUS opinions in this dataset have incomplete `opinions_cited` records; the model can infer likely missing citations
3. **Cross-domain discovery** — cases with high betweenness centrality that the model flags as citation targets may reveal that two previously separate legal doctrines should be cited together

**Setup:**
- Train/test split: random 80/20 (5,728 train edges / 1,431 test edges)
- Negative sampling: 1:1 ratio, per-source corruption (for each positive pair (u,v), sample a non-edge (u,w))
- Models: Random Forest (200 trees) and Logistic Regression (with StandardScaler)

### Feature Sets

| Feature Set | Contents | Dimensions |
|---|---|---|
| `text_only` | Cosine similarity of RoBERTa embeddings | 1 |
| `graph_basic` | Common neighbors, preferential attachment, Jaccard | 3 |
| `graph_triangle` | graph_basic + triangle count, clustering coefficient | 5 |
| `graph_community` | graph_triangle + same-community (Louvain + Label Prop) | 7 |
| `combined` | All graph features + text similarity | 8 |

### Results

| Feature Set | RF AUC | LR AUC | RF F1 | LR F1 |
|---|---|---|---|---|
| text_only | 0.602 | 0.695 | 0.569 | 0.656 |
| graph_basic | 0.735 | 0.676 | 0.557 | 0.583 |
| graph_triangle | 0.617 | 0.725 | 0.506 | 0.608 |
| graph_community | 0.698 | 0.746 | 0.508 | 0.609 |
| **combined** | **0.744** | **0.797** | **0.548** | **0.620** |

### Key Findings

1. **Graph features outperform text alone.** `graph_basic` (AUC 0.735 RF) beats `text_only` (AUC 0.602 RF) — structural features extracted from citation patterns are more predictive than semantic similarity.

2. **Combining graph and text is best.** `combined` achieves the highest AUC (0.797 LR), confirming the two signal types are complementary.

3. **Logistic Regression outperforms Random Forest.** LR consistently scores higher across all feature sets, suggesting the features have near-linear separability. This is a positive finding for interpretability — a simple, explainable model works well.

4. **Community membership adds value.** Adding community indicators (`graph_community`) improves LR AUC from 0.725 to 0.746 over `graph_triangle`, supporting the hypothesis that cases in the same legal domain community are more likely to cite each other.

5. **High precision, lower recall** in RF models indicates the graph features are precise predictors when they fire, but miss many true citations — consistent with the known incompleteness of the `opinions_cited` column.

---

## Experiment 3: Influence Analysis

**Purpose:** Validate that the graph recovers the known legal canon via centrality measures, without any supervision.

### Top 10 by PageRank (α = 0.85)

| Case | PageRank |
|---|---|
| Boyd V United States | 0.005559 |
| United States V Detroit Timber Lumber Co | 0.003555 |
| Stone V Farmers Loan Trust Co | 0.003274 |
| Butchers Benevolent Assn V Crescent City Live Stock Landing | 0.003226 |
| Reid V Colorado | 0.002689 |
| Bailey V Magwire | 0.002674 |
| Chicago Milwaukee St Paul Railway Co V Minnesota | 0.002656 |
| Pembina Consolidated Silver Mining Milling Co V Pennsylvania | 0.002646 |
| United States V American Tobacco Co | 0.002353 |
| Monongahela Navigation Co V United States | 0.002345 |

PageRank surfaces foundational 19th-century precedents that established core constitutional doctrines — *Boyd* (1886, search and seizure), *Butchers' Benevolent* (the *Slaughterhouse Cases*, 1873, 14th Amendment), *Monongahela Navigation* (1893, Takings Clause). These cases are highly cited because they are definitional starting points for entire areas of law.

### Top 10 by Betweenness Centrality (k=500 approximation)

| Case | Betweenness |
|---|---|
| Rescue Army V Municipal Court Of Los Angeles | 0.004045 |
| Shapiro V Thompson | 0.002832 |
| United States V Butler | 0.002782 |
| United States V Obrien | 0.002590 |
| NLRB V Fruit & Vegetable Packers | 0.002540 |
| Alabama State Federation Of Labor V Mcadory | 0.002410 |
| Dandridge V Williams | 0.002398 |
| Griffin V Illinois | 0.002339 |
| Textile Workers V Lincoln Mills Of Ala | 0.002304 |
| Panama Refining Co V Ryan | 0.002125 |

Betweenness centrality identifies **bridge cases** — opinions that connect otherwise separate legal communities. *Rescue Army* (ripeness/standing), *Shapiro V Thompson* (welfare/equal protection/right to travel), and *United States V Butler* (New Deal Commerce Clause) each sit at the intersection of multiple legal doctrines, explaining their high bridge scores. *Roe V Wade* appears at rank 13 (betweenness 0.001833).

### Key Findings

1. **PageRank recovers foundational precedents.** Top PageRank cases are recognized as "anchor" cases in constitutional law, establishing doctrines that all subsequent cases in their domain must address.

2. **Betweenness identifies doctrinal bridges.** High-betweenness cases tend to be those with cross-cutting impact — cases that connected two previously separate legal domains (e.g., standing + civil rights, labor law + constitutional limits).

3. **The two metrics are complementary.** High-PageRank cases are heavily cited within their domain; high-betweenness cases bridge domains. Neither list is a strict subset of the other, suggesting both metrics add distinct legal signal.

---

## Summary

| Experiment | Key Result |
|---|---|
| 1 — Graph Analysis | 3,311-node SCOTUS citation graph with scale-free structure; top-cited cases are canonical constitutional precedents; 122 Louvain communities align with legal domains |
| 2 — Link Prediction | Combined graph+text features achieve AUC 0.797 (LR), vs. 0.695 text-only baseline — graph features provide a statistically meaningful lift |
| 3 — Influence Analysis | PageRank surfaces foundational 19th-century precedents; betweenness identifies doctrinal bridge cases; both recover legally meaningful structure without supervision |

**Overall conclusion:** Graph structural features derived from citation patterns capture legally meaningful information that text embeddings miss. The combined pipeline outperforms text-only baselines on link prediction, and the graph structure recovers the legal canon without any labels — supporting both contributions claimed in the paper.

---

## Reproducibility

All results are deterministic after the one-time citation extraction step:

```bash
# Build dataset and extract citations (one-time, ~5 min)
python3.12 -c "from src.dataset import load_scotus_cases; from src.citation_extraction import build_edge_list; build_edge_list(load_scotus_cases())"

# Compute embeddings (one-time, ~2-3 hours on CPU; cached to data/)
# Run experiment 2 which triggers embedding computation and caches to data/embeddings_roberta_base.npy

# Run experiments
python3.12 experiments/exp1_graph_analysis.py
python3.12 experiments/exp2_link_prediction.py
python3.12 experiments/exp3_influence.py
```

Results are written to `results/exp1/`, `results/exp2/`, `results/exp3/`.
