import json
from pathlib import Path


def test_mini_queries_are_valid_json():
    root = Path(__file__).resolve().parents[1]
    query_file = root / "examples" / "mini_queries.json"

    with open(query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    assert isinstance(queries, list)
    assert len(queries) > 0

    for q in queries:
        assert "query_id" in q
        assert "sql" in q
        assert "template" in q