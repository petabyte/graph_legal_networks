import pandas as pd
from src.citation_extraction import extract_citations_from_text, build_edge_list


def test_extract_citations_from_known_text():
    text = "As held in Brown v. Board of Education, 347 U.S. 483 (1954), the court found..."
    citations = extract_citations_from_text(text)
    assert len(citations) >= 1
    assert any("347 U.S. 483" in c for c in citations)


def test_build_edge_list_returns_dataframe_with_correct_columns():
    df = pd.DataFrame({
        "id": [111, 222, 333],
        "date_created": ["2000-01-01", "1990-01-01", "1995-01-01"],
        "opinions_cited": [
            ["/api/rest/v3/opinions/222/", "/api/rest/v3/opinions/333/"],
            [],
            ["/api/rest/v3/opinions/222/"],
        ],
    })
    edges = build_edge_list(df, cache_path=None)
    assert isinstance(edges, pd.DataFrame)
    assert set(edges.columns) >= {"source_id", "target_id", "year"}
    # case 111 cites 222 and 333
    assert len(edges) == 3
    assert set(edges["source_id"].unique()) == {"111", "333"}


def test_build_edge_list_excludes_self_citations():
    df = pd.DataFrame({
        "id": [111],
        "date_created": ["2000-01-01"],
        "opinions_cited": [["/api/rest/v3/opinions/111/"]],
    })
    edges = build_edge_list(df, cache_path=None)
    assert len(edges) == 0


def test_build_edge_list_excludes_unresolvable_citations():
    df = pd.DataFrame({
        "id": [111],
        "date_created": ["2000-01-01"],
        "opinions_cited": [["/api/rest/v3/opinions/99999/"]],
    })
    edges = build_edge_list(df, cache_path=None)
    # 99999 is not in the dataset, so no edge should be created
    assert len(edges) == 0
