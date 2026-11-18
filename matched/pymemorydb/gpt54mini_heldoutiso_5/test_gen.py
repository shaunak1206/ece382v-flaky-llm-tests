import pytest

from pymemorydb.memorydb import MemoryDb
from pymemorydb.text_analysis import FullTextAnalysis
import pymemorydb.memorydb as memorydb_module


def test_text_index_creates_partition_scope_and_merges_fields():
    db = MemoryDb()

    db.textIndex("p1", {"title": {"weight": 2}, "body": 1})
    db.textIndex("p1", {"tags": {"weight": 3}})

    assert "p1" in db.all_indices
    assert db.all_indices["p1"]["title"] == {"weight": 2}
    assert db.all_indices["p1"]["body"] == 1
    assert db.all_indices["p1"]["tags"] == {"weight": 3}


def test_insert_adds_partition_row_timestamp_and_copies_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 123.456)

    item = {"name": "alpha"}
    db.insert("p1", "r1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "p1"
    assert stored["row_key"] == "r1"
    assert stored["timestamp"] == 123.456
    assert stored["name"] == "alpha"

    # The source item is mutated by insert, but the stored row is a copy.
    assert item["partition_key"] == "p1"
    assert item["row_key"] == "r1"
    assert item["timestamp"] == 123.456
    assert stored is not item


def test_insert_adds_index_data_when_partition_is_indexed(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 2}, "body": 1})
    db.insert("p1", "r1", {"title": "foo foo bar", "body": "bar baz"})

    stored = db.table[0]
    assert "index_data" in stored
    assert stored["index_data"] == [("foo", 4), ("bar", 2), ("baz", 1)]


def test_insert_does_not_add_index_data_for_unindexed_partition(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.insert("p1", "r1", {"title": "foo"})
    stored = db.table[0]
    assert "index_data" not in stored


def test_text_analysis_returns_none_when_no_index_data_exists(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.insert("p1", "r1", {"title": "foo"})
    assert db.textAnalysis("p1") is None


def test_text_analysis_builds_full_text_analysis_for_simple_index(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 1}})
    db.insert("p1", "r1", {"title": "foo foo bar"})
    db.insert("p1", "r2", {"title": "foo baz"})

    analysis = db.textAnalysis("p1")
    assert isinstance(analysis, FullTextAnalysis)

    data = analysis.dump()
    assert set(data.keys()) == {"foo", "bar", "baz"}
    assert data["foo"]["sum"] == 2
    assert data["bar"]["sum"] == 1
    assert data["baz"]["sum"] == 1
    assert data["foo"]["spread"][2]["value"] == "r1"
    assert data["foo"]["spread"][1]["value"] == "r2"


def test_full_text_analysis_terms_scores_top_and_all():
    data = {
        "foo": {
            "sum": 3,
            "spread": {
                2: {"value": "r1", "next": {"value": "r2"}},
                1: {"value": "r3"},
            },
        },
        "bar": {"sum": 1, "spread": {1: {"value": "r4"}}},
    }
    analysis = FullTextAnalysis(data)

    assert set(analysis.terms()) == {"foo", "bar"}
    assert analysis.scores("foo") == [2, 1]
    assert analysis.top("foo") == "r1"
    assert analysis.all("foo") == ["r1", "r2", "r3"]
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_full_text_analysis_filter_returns_matching_terms_sorted_by_sum():
    analysis = FullTextAnalysis(
        {
            "foo": {"sum": 3, "spread": {1: {"value": "a"}}},
            "bar": {"sum": 1, "spread": {1: {"value": "b"}}},
            "baz": {"sum": 5, "spread": {1: {"value": "c"}}},
        }
    )

    filtered = analysis.filter(lambda term: term != "bar")
    assert list(filtered.keys()) == ["baz", "foo"]
    assert set(filtered.keys()) == {"foo", "baz"}


def test_full_text_analysis_filter_with_non_callable_returns_empty_list():
    analysis = FullTextAnalysis({"foo": {"sum": 1, "spread": {1: {"value": "r1"}}}})
    assert analysis.filter("not callable") == []


def test_full_text_analysis_serialize_returns_unique_ids_in_expected_order():
    analysis = FullTextAnalysis(
        {
            "foo": {
                "sum": 4,
                "spread": {
                    3: {"value": "r3", "next": {"value": "r2"}},
                    1: {"value": "r1"},
                },
            },
            "bar": {
                "sum": 2,
                "spread": {2: {"value": "r2b"}, 1: {"value": "r4"}},
            },
        }
    )

    result = analysis.serialize(analysis.dump())
    assert result == ["r3", "r2", "r1", "r2b", "r4"]


def test_full_text_analysis_highlight_replaces_word_boundaries_only():
    analysis = FullTextAnalysis({})
    text = "foo and foobar and foo."
    rendered = analysis.highlight("foo", "<mark>{{_}}</mark>", text)
    assert rendered == "<mark>foo</mark> and foobar and <mark>foo</mark>."


def test_get_by_row_key_returns_inserted_row(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.insert("p1", "r1", {"name": "alpha"})
    row = db.get("p1", "r1")

    assert row["row_key"] == "r1"
    assert row["partition_key"] == "p1"
    assert row["name"] == "alpha"


def test_get_by_row_key_raises_404_for_missing_row():
    db = MemoryDb()
    with pytest.raises(ValueError, match="404"):
        db.get("p1", "missing")


def test_get_with_list_returns_rows_in_requested_order(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.insert("p1", "r1", {"name": "one"})
    db.insert("p1", "r2", {"name": "two"})
    db.insert("p1", "r3", {"name": "three"})

    rows = db.get("p1", ["r3", "r1"])
    assert [r["row_key"] for r in rows] == ["r3", "r1"]


def test_query_sorts_by_ts_descending_before_filtering():
    db = MemoryDb()
    db.table = [
        {"partition_key": "p1", "row_key": "old", "ts": 1, "keep": True},
        {"partition_key": "p1", "row_key": "new", "ts": 2, "keep": True},
        {"partition_key": "p1", "row_key": "mid", "ts": 1.5, "keep": False},
    ]

    result = db.query("p1", lambda item: item["keep"])
    assert [r["row_key"] for r in result] == ["new", "old"]


def test_query_requires_callable_expression():
    db = MemoryDb()
    with pytest.raises(ValueError, match="expression function is required for query"):
        db.query("p1", "not callable")


def test_sample_returns_empty_list_for_empty_partition():
    db = MemoryDb()
    assert db.sample("p1") == []


def test_text_analysis_handles_chain_of_duplicate_terms(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb_module, "time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 1}})
    db.insert("p1", "r1", {"title": "foo foo foo"})
    db.insert("p1", "r2", {"title": "foo"})

    analysis = db.textAnalysis("p1")
    assert analysis.dump()["foo"]["sum"] == 4
    assert analysis.dump()["foo"]["spread"][3]["value"] == "r1"
    assert analysis.dump()["foo"]["spread"][1]["value"] == "r2"
