import random
import sys

import pytest

try:
    from memorydb import MemoryDb
except ImportError:
    from pymemorydb.memorydb import MemoryDb

memorydb_module = sys.modules[MemoryDb.__module__]
FullTextAnalysis = memorydb_module.text_analysis.FullTextAnalysis


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    # Replace the wall-clock reader used by memorydb.insert with a constant.
    # monkeypatch restores the original attribute after each test, even on failure.
    monkeypatch.setattr(memorydb_module, "time", lambda: 1234.5)


def _fresh_fta_data():
    """Hand-built text analysis result, identical to what textAnalysis would produce
    for rows r1 "apple banana", r2 "apple apple", r3 "banana"."""
    return {
        "apple": {
            "sum": 3,
            "spread": {1: {"value": "r1"}, 2: {"value": "r2"}},
        },
        "banana": {
            "sum": 2,
            "spread": {1: {"value": "r1", "next": {"value": "r3"}}},
        },
    }


# ---------------------------------------------------------------- MemoryDb.insert / textIndex

def test_insert_without_descriptor_sets_keys_and_timestamp_only():
    random.seed(0)
    db = MemoryDb()
    item = {"title": "hello"}

    db.insert("p1", "r1", item)

    assert len(db.table) == 1
    row = db.table[0]
    assert row["partition_key"] == "p1"
    assert row["row_key"] == "r1"
    assert row["timestamp"] == 1234.5
    assert row["title"] == "hello"
    assert "index_data" not in row
    # insert mutates the caller's dict as well
    assert item["partition_key"] == "p1"
    assert item["row_key"] == "r1"


def test_insert_stores_copy_not_the_original_item():
    random.seed(0)
    db = MemoryDb()
    item = {"title": "hello"}

    db.insert("p1", "r1", item)
    item["title"] = "changed"

    assert db.table[0]["title"] == "hello"
    assert db.table[0] is not item


def test_textindex_registers_descriptor_for_new_partition():
    random.seed(0)
    db = MemoryDb()

    db.textIndex("p1", {"body": 1})

    assert db.all_indices == {"p1": {"body": 1}}


def test_textindex_merges_into_existing_partition_and_overwrites_same_fields():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": {"weight": 1}})

    db.textIndex("p1", {"title": {"weight": 2}, "body": {"weight": 5}})

    assert db.all_indices["p1"] == {"body": {"weight": 5}, "title": {"weight": 2}}
    assert "p2" not in db.all_indices


def test_insert_builds_scored_index_data_sorted_descending():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    db.insert("p1", "r1", {"body": "a b a c"})

    # a appears twice, b and c once; ties keep first-seen order
    assert db.table[0]["index_data"] == [("a", 2), ("b", 1), ("c", 1)]


def test_insert_collapses_consecutive_runs_and_strips_whitespace():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    db.insert("p1", "r1", {"body": "  a a a b a  "})

    # runs: (a,3), (b,1), (a,1) -> a total 4, b total 1
    assert db.table[0]["index_data"] == [("a", 4), ("b", 1)]


def test_insert_applies_weight_from_descriptor_and_defaults_to_one():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"title": {"weight": 3}, "body": {}})

    db.insert("p1", "r1", {"title": "x y", "body": "x"})

    index = db.table[0]["index_data"]
    assert dict(index) == {"x": 4, "y": 3}  # title x,y weighted 3; body x weight 1
    assert index[0] == ("x", 4)


def test_insert_ignores_fields_not_in_descriptor():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    db.insert("p1", "r1", {"body": "alpha", "secret": "beta gamma"})

    assert db.table[0]["index_data"] == [("alpha", 1)]


def test_insert_missing_indexed_field_gives_empty_index_data():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    db.insert("p1", "r1", {"title": "x"})

    assert db.table[0]["index_data"] == []


def test_insert_non_string_indexed_value_raises_and_stores_nothing():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    with pytest.raises(AttributeError):
        db.insert("p1", "r1", {"body": 42})

    assert db.table == []


def test_insert_into_partition_without_descriptor_has_no_index_data():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    db.insert("p2", "r1", {"body": "a b"})

    assert "index_data" not in db.table[0]


# ---------------------------------------------------------------- MemoryDb.textAnalysis

def test_textanalysis_empty_partition_returns_none():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})

    assert db.textAnalysis("p1") is None


def test_textanalysis_without_index_data_returns_none():
    random.seed(0)
    db = MemoryDb()
    db.insert("p1", "r1", {"title": "x"})  # no descriptor, so no index_data

    assert db.textAnalysis("p1") is None


def test_textanalysis_builds_sum_and_spread_for_each_term():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})
    db.insert("p1", "r1", {"body": "apple banana"})
    db.insert("p1", "r2", {"body": "apple apple"})
    db.insert("p1", "r3", {"body": "banana"})

    fta = db.textAnalysis("p1")

    assert isinstance(fta, FullTextAnalysis)
    data = fta.dump()
    assert set(data.keys()) == {"apple", "banana"}
    # apple: score 1 once (r1) + score 2 once (r2) = 3
    assert data["apple"]["sum"] == 3
    # banana: score 1 twice (r1 then r3 chained) = 2
    assert data["banana"]["sum"] == 2
    assert sorted(data["apple"]["spread"].keys()) == [1, 2]
    assert data["apple"]["spread"][1]["value"] == "r1"
    assert data["apple"]["spread"][2]["value"] == "r2"
    assert data["banana"]["spread"][1]["value"] == "r1"
    assert data["banana"]["spread"][1]["next"]["value"] == "r3"


def test_textanalysis_sum_counts_every_chained_row():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})
    for rk in ("r1", "r2", "r3"):
        db.insert("p1", rk, {"body": "x x"})  # each row: x with score 2

    data = db.textAnalysis("p1").dump()

    assert data["x"]["sum"] == 6  # score 2 times 3 chained rows
    node = data["x"]["spread"][2]
    walked = []
    while node is not None:
        walked.append(node["value"])
        node = node.get("next")
    assert walked == ["r1", "r2", "r3"]


def test_textanalysis_excludes_rows_from_other_partitions():
    random.seed(0)
    db = MemoryDb()
    db.textIndex("p1", {"body": 1})
    db.textIndex("p2", {"body": 1})
    db.insert("p1", "r1", {"body": "apple"})
    db.insert("p2", "r2", {"body": "kiwi"})

    fta = db.textAnalysis("p1")

    assert sorted(fta.terms()) == ["apple"]


# ---------------------------------------------------------------- FullTextAnalysis

def test_terms_returns_all_keys_and_dump_returns_same_data():
    data = _fresh_fta_data()
    fta = FullTextAnalysis(data)

    assert sorted(fta.terms()) == ["apple", "banana"]
    assert fta.dump() is data


def test_scores_returns_descending_scores_or_none_for_unknown_term():
    fta = FullTextAnalysis(_fresh_fta_data())

    assert fta.scores("apple") == [2, 1]
    assert fta.scores("banana") == [1]
    assert fta.scores("missing") is None


def test_top_returns_row_key_with_highest_score():
    fta = FullTextAnalysis(_fresh_fta_data())

    assert fta.top("apple") == "r2"
    assert fta.top("banana") == "r1"
    assert fta.top("missing") is None


def test_all_returns_every_row_including_chained_ones():
    fta = FullTextAnalysis(_fresh_fta_data())

    assert sorted(fta.all("apple")) == ["r1", "r2"]
    assert sorted(fta.all("banana")) == ["r1", "r3"]


def test_filter_keeps_matching_terms_and_handles_bad_expression():
    data = _fresh_fta_data()
    fta = FullTextAnalysis(data)

    result = fta.filter(lambda term: term.startswith("a"))

    assert set(result.keys()) == {"apple"}
    assert result["apple"] is data["apple"]
    assert fta.filter(lambda term: False) == {}
    assert fta.filter("apple") == []


def test_serialize_returns_unique_row_keys_in_expected_order():
    fta = FullTextAnalysis(_fresh_fta_data())

    # apple: score 2 (r2) then score 1 (r1)
    assert fta.serialize(fta.filter(lambda term: term == "apple")) == ["r2", "r1"]

    everything = fta.serialize(fta.filter(lambda term: True))
    assert len(everything) == 3
    assert sorted(everything) == ["r1", "r2", "r3"]

    assert fta.serialize(None) is None


def test_highlight_wraps_whole_word_matches_only():
    fta = FullTextAnalysis({})

    result = fta.highlight("cat", "concatenate cat and cat", "[{{_}}]")

    assert result == "concatenate [cat] and [cat]"
