from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Field names for st3re0/scotus-opinions dataset.
# NOTE: The originally specified dataset (HFforLegal/case-law, split="train")
# does not exist and its only split ("us") contains ~529k state supreme court
# cases with no U.S. Supreme Court (SCOTUS) records at all.  The replacement
# dataset st3re0/scotus-opinions (27 186 rows) is a direct match for the
# SCOTUS Citation Graph project: it contains every published SCOTUS opinion
# together with a pre-extracted citations list (opinions_cited).
#
# Field mapping vs. original spec:
#   COURT_FIELD  → not needed (all rows are SCOTUS by construction)
#   DATE_FIELD   → "date_created"   (ISO-8601 timestamp)
#   TEXT_FIELD   → "html_with_citations"  (100 % coverage; plain_text < 5 %)
#   ID_FIELD     → "id"             (numeric opinion ID from CourtListener)
#   CITATIONS_FIELD → "opinions_cited"  (list of cited opinion URIs)

DATE_FIELD = "date_created"
TEXT_FIELD = "html_with_citations"
ID_FIELD = "id"
CITATIONS_FIELD = "opinions_cited"

HF_DATASET = "st3re0/scotus-opinions"
HF_SPLIT = "train"

DATA_DIR = Path("data")


def load_scotus_cases(
    cache_path: Path = DATA_DIR / "scotus_cases.parquet",
) -> pd.DataFrame:
    """Load and filter SCOTUS opinions.  Returns cached parquet if it exists.

    The returned DataFrame contains every SCOTUS opinion from the
    st3re0/scotus-opinions dataset that has a parseable date and non-empty
    opinion text.  The ``opinions_cited`` column (list of CourtListener URIs)
    is preserved for downstream citation-graph construction.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    df = ds.to_pandas()

    # Drop rows that lack usable text
    scotus = df.dropna(subset=[TEXT_FIELD]).copy()
    scotus = scotus[scotus[TEXT_FIELD].str.strip().str.len() > 0]

    # Parse and validate the decision date.
    # format="ISO8601" is required because the column mixes timestamps with and
    # without sub-second precision (e.g. "2010-04-28T09:00:12-07:00" vs
    # "2014-04-02T12:57:22.894687-07:00").  Without it, pandas silently returns
    # NaT for the majority of rows when utc=True is combined with mixed
    # fractional-second precision strings.
    scotus[DATE_FIELD] = pd.to_datetime(
        scotus[DATE_FIELD], format="ISO8601", errors="coerce", utc=True
    )
    scotus = scotus.dropna(subset=[DATE_FIELD])

    scotus = scotus.reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    scotus.to_parquet(cache_path, index=False)
    return scotus
