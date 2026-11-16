import random
import sys
import os

import pytest

try:
    import memorydb
    import text_analysis
except ImportError:  # pragma: no cover - fallback when root isn't on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import memorydb
    import text_analysis

from memorydb import MemoryDb
from text_analysis import FullTextAnalysis


def _fixed_time(monkeypatch, value=1234.5):
    monkeypatch.setattr(memorydb, "time", lambda: value)


def _build_db():
    db = MemoryDb()
    db.textIndex("p", {"body": 1})
    db.insert("p", "r1", {"body": "apple apple banana"})
    db.insert("p", "r2", {"body": "apple cherry"})
    return db


# ---------------------------------------------------------------- textIndex

def test_textIndex_creates_partition_index():
    random.seed(0)
    db = MemoryDb()
    desc = {"title": {"weight": 3}, "body": 1}
    db.textIndex("p", desc)
    assert "p" in db.all_indices
    assert db.all_indices["p"] == {"title": {"weight": 3}, "body": 1}
    assert db.dumpindices("p") == desc


def test_textIndex_merges_and_overrides_existing_keys():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p", {"title": 1, "body": 1})
    db.textIndex("p", {"body": {"weight": 5}, "tags": 2})
    assert db.all_indices["p"] == {"title": 1, "body": {"weight": 5}, "tags": 2}


def test_textIndex_partitions_are_independent():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("a", {"x": 1})
    db.textIndex("b", {"y": 1})
    assert db.all_indices["a"] == {"x": 1}
    assert db.all_indices["b"] == {"y": 1}
    assert set(db.dumpindices().keys()) == {"a", "b"}


# ------------------------------------------------------------------- insert

def test_insert_sets_metadata_and_stores_row(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch, 1234.5)
    db = MemoryDb()
    item = {"name": "x"}
    db.insert("p", "r1", item)
    assert item["partition_key"] == "p"
    assert item["row_key"] == "r1"
    assert item["timestamp"] == pytest.approx(1234.5)
    table = db.dumpdb()
    assert len(table) == 1
    assert table[0]["name"] == "x"
    assert table[0]["partition_key"] == "p"
    assert table[0]["row_key"] == "r1"
    assert table[0]["timestamp"] == pytest.approx(1234.5)


def test_insert_stores_a_copy_not_the_same_object(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    item = {"name": "x"}
    db.insert("p", "r1", item)
    assert db.table[0] is not item
    assert db.table[0] == item


def test_insert_without_index_adds_no_index_data(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.insert("p", "r1", {"body": "hello world"})
    assert "index_data" not in db.table[0]


def test_insert_with_index_computes_weighted_counts(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"title": {"weight": 3}, "body": 1})
    db.insert("p", "r1", {"title": "foo bar", "body": "foo foo baz"})
    # title: foo=3, bar=3 ; body: foo=2, baz=1  -> foo=5, bar=3, baz=1
    assert db.table[0]["index_data"] == [("foo", 5), ("bar", 3), ("baz", 1)]


def test_insert_dict_index_without_weight_defaults_to_one(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"body": {}})
    db.insert("p", "r1", {"body": "a a a b"})
    assert db.table[0]["index_data"] == [("a", 3), ("b", 1)]


def test_insert_only_indexes_configured_fields(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"title": 1})
    db.insert("p", "r1", {"title": "alpha", "body": "beta gamma"})
    assert dict(db.table[0]["index_data"]) == {"alpha": 1}


def test_insert_counts_only_consecutive_repeats_per_run(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"body": 1})
    db.insert("p", "r1", {"body": "x y x x"})
    # groups: x:1, y:1, x:2 -> x total 3, y total 1
    assert db.table[0]["index_data"] == [("x", 3), ("y", 1)]


def test_insert_other_partition_not_indexed(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("indexed", {"body": 1})
    db.insert("other", "r1", {"body": "hello"})
    assert "index_data" not in db.table[0]


def test_insert_non_string_indexed_field_raises(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"body": 1})
    with pytest.raises(AttributeError):
        db.insert("p", "r1", {"body": 123})
    assert db.table == []


# ------------------------------------------------------------- textAnalysis

def test_textAnalysis_returns_none_without_index_data(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.insert("p", "r1", {"body": "hello"})
    assert db.textAnalysis("p") is None
    assert db.textAnalysis("unknown") is None


def test_textAnalysis_sums_and_spread(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = _build_db()
    analysis = db.textAnalysis("p")
    assert isinstance(analysis, FullTextAnalysis)
    data = analysis.dump()
    assert set(data.keys()) == {"apple", "banana", "cherry"}
    assert data["apple"]["sum"] == 3
    assert data["banana"]["sum"] == 1
    assert data["cherry"]["sum"] == 1
    assert data["apple"]["spread"] == {2: {"value": "r1"}, 1: {"value": "r2"}}
    assert data["banana"]["spread"] == {1: {"value": "r1"}}
    assert data["cherry"]["spread"] == {1: {"value": "r2"}}


def test_textAnalysis_chains_rows_with_same_score(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"body": 1})
    for rk in ("r1", "r2", "r3"):
        db.insert("p", rk, {"body": "apple"})
    data = db.textAnalysis("p").dump()
    assert data["apple"]["sum"] == 3
    assert data["apple"]["spread"] == {
        1: {"value": "r1", "next": {"value": "r2", "next": {"value": "r3"}}}
    }


def test_textAnalysis_ignores_other_partitions(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    db = MemoryDb()
    db.textIndex("p", {"body": 1})
    db.textIndex("q", {"body": 1})
    db.insert("p", "r1", {"body": "alpha"})
    db.insert("q", "r2", {"body": "beta"})
    assert set(db.textAnalysis("p").terms()) == {"alpha"}
    assert set(db.textAnalysis("q").terms()) == {"beta"}


# --------------------------------------------------------- FullTextAnalysis

def test_analysis_terms_scores_top(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    analysis = _build_db().textAnalysis("p")
    assert set(analysis.terms()) == {"apple", "banana", "cherry"}
    assert analysis.scores("apple") == [2, 1]
    assert analysis.top("apple") == "r1"
    assert analysis.top("cherry") == "r2"
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_analysis_all_returns_every_row(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    analysis = _build_db().textAnalysis("p")
    assert sorted(analysis.all("apple")) == ["r1", "r2"]
    assert analysis.all("banana") == ["r1"]

    chained = FullTextAnalysis({
        "t": {"sum": 3, "spread": {1: {"value": "a", "next": {"value": "b", "next": {"value": "c"}}}}}
    })
    assert chained.all("t") == ["a", "b", "c"]


def test_analysis_filter_selects_terms(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    analysis = _build_db().textAnalysis("p")
    result = analysis.filter(lambda t: t != "banana")
    assert set(result.keys()) == {"apple", "cherry"}
    assert result["apple"] is analysis.dump()["apple"]
    assert analysis.filter(lambda t: False) == {}


def test_analysis_filter_non_callable_returns_empty_list():
    random.seed(0)
    analysis = FullTextAnalysis({"a": {"sum": 1, "spread": {1: {"value": "r1"}}}})
    assert analysis.filter("not callable") == []


def test_analysis_serialize_orders_by_score_and_dedupes(monkeypatch):
    random.seed(0)
    _fixed_time(monkeypatch)
    assert FullTextAnalysis({}).serialize(None) is None

    single = FullTextAnalysis({
        "apple": {"sum": 3, "spread": {1: {"value": "r2"}, 2: {"value": "r1"}}}
    })
    assert single.serialize(single.filter(lambda t: True)) == ["r1", "r2"]

    chained = FullTextAnalysis({
        "t": {"sum": 3, "spread": {1: {"value": "a", "next": {"value": "b"}}}}
    })
    assert chained.serialize(chained.filter(lambda t: True)) == ["a", "b"]

    analysis = _build_db().textAnalysis("p")
    ids = analysis.serialize(analysis.filter(lambda t: True))
    assert sorted(ids) == ["r1", "r2"]
    assert len(ids) == len(set(ids))


def test_analysis_highlight_wraps_whole_words_only():
    random.seed(0)
    analysis = FullTextAnalysis({})
    out = analysis.highlight("foo", "foo bar foo", "<b>{{_}}</b>")
    assert out == "<b>foo</b> bar <b>foo</b>"
    out = analysis.highlight("foo", "foobar foo", "[{{_}}]")
    assert out == "foobar [foo]"
    assert analysis.highlight("zzz", "foo bar", "[{{_}}]") == "foo bar"
