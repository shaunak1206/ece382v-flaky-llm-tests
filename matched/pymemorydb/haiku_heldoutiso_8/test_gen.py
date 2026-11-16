import os
import sys
import time as _time_module

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _add_search_paths():
    """Make memorydb.py and text_analysis.py importable from common project layouts."""
    bases = [os.getcwd(), _HERE, os.path.dirname(_HERE)]
    for base in bases:
        candidates = [
            base,
            os.path.join(base, "pymemorydb"),
            os.path.join(base, "src"),
            os.path.join(base, "src", "pymemorydb"),
        ]
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, "memorydb.py")) and candidate not in sys.path:
                sys.path.insert(0, candidate)


_add_search_paths()

try:
    import memorydb as memorydb_module
except ImportError:
    import pymemorydb.memorydb as memorydb_module


@pytest.fixture
def db():
    return memorydb_module.MemoryDb()


def _make_analysis():
    data = {
        "apple": {"sum": 3, "spread": {2: {"value": "a"}, 1: {"value": "b"}}},
        "avocado": {"sum": 4, "spread": {3: {"value": "c"}, 1: {"value": "a"}}},
        "kiwi": {"sum": 1, "spread": {1: {"value": "d"}}},
    }
    return memorydb_module.text_analysis.FullTextAnalysis(data)


# ---------------------------------------------------------------- textIndex

def test_textindex_creates_partition_entry(db):
    db.textIndex("posts", {"title": {"weight": 2}})
    assert db.all_indices == {"posts": {"title": {"weight": 2}}}


def test_textindex_merges_and_overwrites_existing_fields(db):
    db.textIndex("posts", {"title": 1, "body": 1})
    db.textIndex("posts", {"title": {"weight": 5}, "tags": {"weight": 3}})
    assert db.all_indices["posts"] == {
        "title": {"weight": 5},
        "body": 1,
        "tags": {"weight": 3},
    }


# ---------------------------------------------------------------- insert

def test_insert_sets_key_fields_and_timestamp(db):
    item = {"name": "x"}
    before = _time_module.time()
    db.insert("p", "r1", item)
    after = _time_module.time()

    row = db.table[0]
    assert row["partition_key"] == "p"
    assert row["row_key"] == "r1"
    assert before <= row["timestamp"] <= after
    assert item["partition_key"] == "p"  # input is mutated in place
    assert "index_data" not in row


def test_insert_without_partition_index_omits_index_data(db):
    db.insert("p", "r1", {"body": "apple pie"})
    assert "index_data" not in db.table[0]


def test_insert_with_index_records_scored_tokens(db):
    db.textIndex("p", {"body": {}})
    db.insert("p", "r1", {"body": "apple banana apple"})
    assert db.table[0]["index_data"] == [("apple", 2), ("banana", 1)]


def test_insert_applies_field_weight_to_consecutive_runs(db):
    db.textIndex("p", {"title": {"weight": 3}})
    db.insert("p", "r1", {"title": "go go stop"})
    assert db.table[0]["index_data"] == [("go", 6), ("stop", 3)]


def test_insert_ignores_fields_missing_from_index(db):
    db.textIndex("p", {"title": {}})
    db.insert("p", "r1", {"title": "banana", "body": "apple apple"})
    assert db.table[0]["index_data"] == [("banana", 1)]


def test_insert_with_no_matching_fields_stores_empty_index_data(db):
    db.textIndex("p", {"title": {}})
    db.insert("p", "r1", {"body": "apple"})
    assert db.table[0]["index_data"] == []


def test_insert_accumulates_repeated_tokens_across_runs(db):
    db.textIndex("p", {"body": {}})
    db.insert("p", "r1", {"body": "a b a"})
    assert db.table[0]["index_data"] == [("a", 2), ("b", 1)]


def test_insert_combines_scores_from_multiple_fields(db):
    db.textIndex("p", {"title": {"weight": 2}, "body": {}})
    db.insert("p", "r1", {"title": "cat", "body": "cat dog"})
    assert db.table[0]["index_data"] == [("cat", 3), ("dog", 1)]


# ---------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_without_indexed_rows(db):
    db.insert("p", "r1", {"body": "apple"})
    assert db.textAnalysis("p") is None


def test_textanalysis_returns_none_for_unknown_partition(db):
    assert db.textAnalysis("missing") is None


def test_textanalysis_sums_scores_per_term(db):
    db.textIndex("p", {"body": {}})
    db.insert("p", "a", {"body": "apple apple"})
    db.insert("p", "b", {"body": "apple"})

    ta = db.textAnalysis("p")
    assert isinstance(ta, memorydb_module.text_analysis.FullTextAnalysis)
    assert ta.dump()["apple"]["sum"] == 3
    assert ta.scores("apple") == [2, 1]


def test_textanalysis_only_includes_requested_partition(db):
    db.textIndex("p", {"body": {}})
    db.textIndex("q", {"body": {}})
    db.insert("p", "a", {"body": "apple"})
    db.insert("q", "b", {"body": "kiwi"})

    ta = db.textAnalysis("p")
    assert ta.terms() == ["apple"]


def test_textanalysis_links_rows_sharing_a_term_and_score(db):
    db.textIndex("p", {"body": {}})
    db.insert("p", "a", {"body": "apple"})
    db.insert("p", "b", {"body": "apple"})

    ta = db.textAnalysis("p")
    assert ta.dump()["apple"]["sum"] == 2
    assert sorted(ta.all("apple")) == ["a", "b"]


def test_top_returns_row_key_with_highest_score(db):
    db.textIndex("p", {"body": {}})
    db.insert("p", "a", {"body": "apple apple"})
    db.insert("p", "b", {"body": "apple"})

    ta = db.textAnalysis("p")
    assert ta.top("apple") == "a"
    assert ta.top("missing") is None


# ---------------------------------------------------------------- FullTextAnalysis

def test_scores_returns_none_for_unknown_term():
    ta = _make_analysis()
    assert ta.scores("missing") is None
    assert ta.scores("apple") == [2, 1]


def test_filter_returns_entries_for_matching_terms():
    ta = _make_analysis()
    result = ta.filter(lambda t: t.startswith("a"))
    assert set(result.keys()) == {"apple", "avocado"}
    assert result["apple"] is ta.dump()["apple"]


def test_filter_with_non_callable_returns_empty_list():
    assert _make_analysis().filter("apple") == []


def test_serialize_orders_row_keys_by_score_and_removes_duplicates():
    ta = _make_analysis()
    filtered = ta.filter(lambda t: t.startswith("a"))
    assert ta.serialize(filtered) == ["a", "b", "c"]


def test_highlight_wraps_whole_word_matches_only():
    ta = _make_analysis()
    assert ta.highlight("apple", "pineapple apple", "[{{_}}]") == "pineapple [apple]"


def test_highlight_is_case_sensitive():
    ta = _make_analysis()
    assert ta.highlight("apple", "Apple apple", "[{{_}}]") == "Apple [apple]"


def test_highlight_replaces_every_occurrence():
    ta = _make_analysis()
    assert ta.highlight("apple", "apple pie apple", "<b>{{_}}</b>") == "<b>apple</b> pie <b>apple</b>"
