import copy
import pytest

from memorydb import MemoryDb
from text_analysis import FullTextAnalysis


def test_text_index_registers_descriptors_and_merges_fields():
    db = MemoryDb()

    db.textIndex("p1", {"title": 1})
    db.textIndex("p1", {"body": {"weight": 2}, "extra": 3})

    assert db.all_indices["p1"]["title"] == 1
    assert db.all_indices["p1"]["body"] == {"weight": 2}
    assert db.all_indices["p1"]["extra"] == 3


def test_insert_sets_keys_timestamp_and_stores_copy(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 123.456)

    item = {"name": "alice"}
    db.insert("pk", "rk", item)

    assert item["partition_key"] == "pk"
    assert item["row_key"] == "rk"
    assert item["timestamp"] == 123.456
    assert len(db.table) == 1
    assert db.table[0] is not item
    assert db.table[0]["name"] == "alice"
    assert db.table[0]["timestamp"] == 123.456


def test_insert_adds_index_data_when_index_registered(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("p1", {"title": 1})
    db.insert("p1", "r1", {"title": "alpha alpha beta"})

    assert "index_data" in db.table[0]
    assert db.table[0]["index_data"] == [("alpha", 2), ("beta", 1)]


def test_insert_without_index_registration_does_not_add_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.insert("p1", "r1", {"title": "alpha beta"})

    assert "index_data" not in db.table[0]


def test_insert_shallow_copies_top_level_but_shares_nested_objects(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    nested = {"tags": ["x"]}
    item = {"meta": nested}
    db.insert("p1", "r1", item)

    item["meta"]["tags"].append("y")
    assert db.table[0]["meta"]["tags"] == ["x", "y"]


def test_text_analysis_returns_full_text_analysis_with_expected_terms(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("p1", {"title": 1})
    db.insert("p1", "r1", {"title": "alpha alpha beta"})
    db.insert("p1", "r2", {"title": "alpha gamma"})

    analysis = db.textAnalysis("p1")

    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ["alpha", "beta", "gamma"]


def test_text_analysis_summarizes_scores_top_and_all(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("p1", {"title": 1})
    db.insert("p1", "r1", {"title": "alpha alpha beta"})
    db.insert("p1", "r2", {"title": "alpha gamma"})

    analysis = db.textAnalysis("p1")

    assert analysis.scores("alpha") == [2, 1]
    assert analysis.top("alpha") == "r1"
    assert analysis.all("alpha") == ["r1", "r2"]
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None
    assert analysis.all("missing") == []


def test_text_analysis_uses_weighted_fields(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 2}})
    db.insert("p1", "r1", {"title": "alpha alpha beta"})

    analysis = db.textAnalysis("p1")

    assert analysis.scores("alpha") == [4]
    assert analysis.scores("beta") == [2]
    assert analysis.top("alpha") == "r1"


def test_text_analysis_ignores_items_without_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.insert("p1", "r1", {"name": "no indexing configured"})

    assert db.textAnalysis("p1") is None


def test_full_text_analysis_dump_and_terms_reflect_input():
    data = {
        "alpha": {"sum": 3, "spread": {2: {"value": "r1"}, 1: {"value": "r2"}}},
        "beta": {"sum": 1, "spread": {1: {"value": "r3"}}},
    }
    analysis = FullTextAnalysis(data)

    assert analysis.dump() == data
    assert sorted(analysis.terms()) == ["alpha", "beta"]


def test_full_text_analysis_filter_with_callable_returns_matching_terms():
    data = {
        "alpha": {"sum": 3, "spread": {2: {"value": "r1"}}},
        "beta": {"sum": 1, "spread": {1: {"value": "r2"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term.startswith("a"))

    assert list(filtered.keys()) == ["alpha"]
    assert filtered["alpha"]["sum"] == 3


def test_full_text_analysis_filter_non_callable_returns_empty_list():
    analysis = FullTextAnalysis({"alpha": {"sum": 3, "spread": {2: {"value": "r1"}}}})

    assert analysis.filter("not callable") == []


def test_full_text_analysis_serialize_deduplicates_and_orders_ids():
    data = {
        "alpha": {
            "sum": 3,
            "spread": {
                2: {"value": "r1", "next": {"value": "r2"}},
                1: {"value": "r3"},
            },
        }
    }
    analysis = FullTextAnalysis(data)

    assert analysis.serialize(data) == ["r1", "r2", "r3"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})

    result = analysis.highlight("cat", "cat scatter catalog cat.", "X {{_}} Y")

    assert result == "X cat Y scatter catalog X cat Y."
