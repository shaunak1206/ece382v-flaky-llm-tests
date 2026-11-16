import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import memorydb as memorydb_module
except ImportError:
    from pymemorydb import memorydb as memorydb_module

MemoryDb = memorydb_module.MemoryDb
FullTextAnalysis = memorydb_module.text_analysis.FullTextAnalysis


def make_db(partition="p", desc=None):
    db = MemoryDb()
    db.textIndex(partition, desc if desc is not None else {"title": True})
    return db


# ---------------------------------------------------------------- textIndex

def test_textindex_creates_partition_index_and_registers_fields():
    db = MemoryDb()
    db.textIndex("p", {"title": {"weight": 3}, "body": True})
    assert db.all_indices == {"p": {"title": {"weight": 3}, "body": True}}


def test_textindex_merges_into_existing_partition_and_overrides_field():
    db = MemoryDb()
    db.textIndex("p", {"title": {"weight": 1}, "body": True})
    db.textIndex("p", {"title": {"weight": 5}, "tags": True})
    assert db.all_indices["p"] == {
        "title": {"weight": 5},
        "body": True,
        "tags": True,
    }


def test_textindex_keeps_partitions_separate():
    db = MemoryDb()
    db.textIndex("a", {"title": True})
    db.textIndex("b", {"body": True})
    assert db.all_indices == {"a": {"title": True}, "b": {"body": True}}


# ------------------------------------------------------------------- insert

def test_insert_sets_key_fields_and_timestamp(monkeypatch):
    monkeypatch.setattr(memorydb_module, "time", lambda: 1000.0)
    db = MemoryDb()
    item = {"name": "x"}
    db.insert("p", "r1", item)
    row = db.table[0]
    assert row["partition_key"] == "p"
    assert row["row_key"] == "r1"
    assert row["timestamp"] == 1000.0
    assert row["name"] == "x"
    assert item["timestamp"] == 1000.0


def test_insert_without_index_definition_stores_no_index_data():
    db = MemoryDb()
    db.insert("p", "r1", {"title": "hello world"})
    assert "index_data" not in db.table[0]


def test_insert_orders_index_tokens_by_descending_score():
    db = make_db("p", {"title": True})
    db.insert("p", "r1", {"title": "a b b"})
    assert db.table[0]["index_data"] == [("b", 2), ("a", 1)]


def test_insert_applies_field_weight_to_scores():
    db = make_db("p", {"title": {"weight": 3}})
    db.insert("p", "r1", {"title": "cat dog"})
    assert dict(db.table[0]["index_data"]) == {"cat": 3, "dog": 3}


def test_insert_sums_scores_across_indexed_fields_only():
    db = make_db("p", {"title": {"weight": 2}, "body": True})
    db.insert("p", "r1", {"title": "fox", "body": "fox fox", "notes": "fox"})
    # title: fox -> 1 * 2 = 2 ; body: fox fox -> 2 * 1 = 2 ; notes not indexed
    assert dict(db.table[0]["index_data"]) == {"fox": 4}


def test_insert_with_empty_partition_index_sets_empty_index_data():
    db = make_db("p", {})
    db.insert("p", "r1", {"title": "hello"})
    assert db.table[0]["index_data"] == []


def test_insert_stores_shallow_copy_of_item():
    db = make_db("p", {"title": True})
    item = {"title": "alpha"}
    db.insert("p", "r1", item)
    item["title"] = "changed"
    assert db.table[0]["title"] == "alpha"


# ------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_when_no_rows_have_index_data():
    db = MemoryDb()
    db.insert("p", "r1", {"title": "plain"})
    assert db.textAnalysis("p") is None
    assert db.textAnalysis("missing") is None


def test_textanalysis_groups_row_keys_under_terms_with_summed_scores():
    db = make_db("p", {"title": True})
    db.insert("p", "r1", {"title": "red blue"})
    db.insert("p", "r2", {"title": "red"})
    analysis = db.textAnalysis("p")
    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ["blue", "red"]
    assert analysis.dump()["red"]["sum"] == 2
    assert analysis.dump()["blue"]["sum"] == 1
    assert analysis.all("red") == ["r1", "r2"]


def test_textanalysis_sum_and_ranking_use_score_levels():
    db = make_db("p", {"title": True})
    db.insert("p", "r1", {"title": "go go"})  # go scores 2
    db.insert("p", "r2", {"title": "go"})     # go scores 1
    analysis = db.textAnalysis("p")
    assert analysis.scores("go") == [2, 1]
    assert analysis.top("go") == "r1"
    assert analysis.dump()["go"]["sum"] == 3


def test_textanalysis_chains_rows_sharing_same_score_in_insertion_order():
    db = make_db("p", {"title": True})
    for key in ("a", "b", "c"):
        db.insert("p", key, {"title": "x"})
    analysis = db.textAnalysis("p")
    assert analysis.all("x") == ["a", "b", "c"]
    assert analysis.dump()["x"]["sum"] == 3
    assert analysis.top("x") == "a"


def test_textanalysis_limits_to_partition_and_skips_rows_without_index_data():
    db = MemoryDb()
    db.insert("p", "old", {"title": "ancient"})  # no index yet -> no index_data
    db.textIndex("p", {"title": True})
    db.textIndex("q", {"title": True})
    db.insert("p", "r1", {"title": "alpha"})
    db.insert("q", "r2", {"title": "beta"})
    analysis = db.textAnalysis("p")
    assert analysis.terms() == ["alpha"]


# ----------------------------------------------------------- FullTextAnalysis

def test_full_text_analysis_terms_and_dump_return_wrapped_data():
    data = {
        "apple": {"sum": 1, "spread": {1: {"value": "r1"}}},
        "banana": {"sum": 1, "spread": {1: {"value": "r2"}}},
    }
    fta = FullTextAnalysis(data)
    assert sorted(fta.terms()) == ["apple", "banana"]
    assert fta.dump() is data


def test_scores_are_descending_and_none_for_unknown_term():
    data = {
        "t": {
            "sum": 6,
            "spread": {1: {"value": "a"}, 3: {"value": "b"}, 2: {"value": "c"}},
        }
    }
    fta = FullTextAnalysis(data)
    assert fta.scores("t") == [3, 2, 1]
    assert fta.scores("missing") is None


def test_top_returns_head_row_of_highest_scored_level():
    data = {
        "t": {
            "sum": 6,
            "spread": {
                1: {"value": "low"},
                4: {"value": "high", "next": {"value": "high2"}},
            },
        }
    }
    fta = FullTextAnalysis(data)
    assert fta.top("t") == "high"
    assert fta.top("missing") is None


def test_all_flattens_chains_in_score_order():
    spread = {
        5: {"value": "A", "next": {"value": "B"}},
        2: {"value": "C"},
    }
    fta = FullTextAnalysis({"t": {"sum": 9, "spread": spread}})
    assert fta.all("t") == ["A", "B", "C"]


def test_filter_keeps_matching_terms_and_rejects_non_callable():
    data = {
        "apple": {"sum": 1, "spread": {1: {"value": "r1"}}},
        "banana": {"sum": 1, "spread": {1: {"value": "r2"}}},
        "avocado": {"sum": 1, "spread": {1: {"value": "r3"}}},
    }
    fta = FullTextAnalysis(data)
    result = fta.filter(lambda term: term.startswith("a"))
    assert list(result.keys()) == ["apple", "avocado"]
    assert result["apple"] is data["apple"]
    assert fta.filter("a") == []


def test_serialize_none_returns_none():
    fta = FullTextAnalysis({})
    assert fta.serialize(None) is None


def test_serialize_lists_unique_row_keys_by_descending_score():
    data = {
        "x": {
            "sum": 9,
            "spread": {
                2: {"value": "A", "next": {"value": "B"}},
                5: {"value": "C"},
            },
        },
        "y": {"sum": 1, "spread": {1: {"value": "B"}}},
    }
    fta = FullTextAnalysis(data)
    assert fta.serialize(fta.filter(lambda t: True)) == ["C", "A", "B"]


def test_serialize_round_trip_from_text_analysis_returns_all_row_keys():
    db = make_db("p", {"title": True})
    db.insert("p", "r1", {"title": "red blue"})
    db.insert("p", "r2", {"title": "red"})
    analysis = db.textAnalysis("p")
    ids = analysis.serialize(analysis.filter(lambda t: True))
    assert len(ids) == 2
    assert sorted(ids) == ["r1", "r2"]


def test_highlight_wraps_only_whole_word_matches():
    fta = FullTextAnalysis({})
    assert fta.highlight("fox", "the fox and foxes", "[{{_}}]") == "the [fox] and foxes"
