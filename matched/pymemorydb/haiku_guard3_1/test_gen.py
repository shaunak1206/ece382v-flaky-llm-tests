import random

import memorydb
import text_analysis


# ---------------------------------------------------------------------------
# MemoryDb.textIndex
# ---------------------------------------------------------------------------

def test_text_index_stores_description_for_partition():
    random.seed(0)
    db = memorydb.MemoryDb()

    result = db.textIndex("docs", {"body": {"weight": 2}, "title": {}})

    assert result is None
    assert db.dumpindices("docs") == {"body": {"weight": 2}, "title": {}}
    assert db.dumpindices("missing") is None


def test_text_index_merges_new_fields_and_overwrites_existing():
    random.seed(0)
    db = memorydb.MemoryDb()

    db.textIndex("docs", {"body": {"weight": 1}})
    db.textIndex("docs", {"title": {"weight": 5}})
    assert db.dumpindices("docs") == {"body": {"weight": 1}, "title": {"weight": 5}}

    db.textIndex("docs", {"body": {"weight": 9}})
    assert db.dumpindices("docs") == {"body": {"weight": 9}, "title": {"weight": 5}}


def test_text_index_keeps_partitions_separate():
    random.seed(0)
    db = memorydb.MemoryDb()

    db.textIndex("p1", {"body": {}})
    db.textIndex("p2", {"title": {}})

    assert db.dumpindices("p1") == {"body": {}}
    assert db.dumpindices("p2") == {"title": {}}
    assert sorted(db.dumpindices().keys()) == ["p1", "p2"]


def test_text_index_with_empty_description_still_marks_partition():
    random.seed(0)
    db = memorydb.MemoryDb()

    db.textIndex("docs", {})
    assert db.dumpindices("docs") == {}

    db.insert("docs", "r1", {"body": "hello"})
    assert db.dumpdb()[0]["index_data"] == []


def test_rows_inserted_before_text_index_are_not_indexed():
    random.seed(0)
    db = memorydb.MemoryDb()

    db.insert("docs", "r1", {"body": "apple"})
    db.textIndex("docs", {"body": {}})

    assert "index_data" not in db.dumpdb()[0]
    assert db.textAnalysis("docs") is None


# ---------------------------------------------------------------------------
# MemoryDb.insert
# ---------------------------------------------------------------------------

def test_insert_without_index_stores_row_without_index_data():
    random.seed(0)
    db = memorydb.MemoryDb()

    db.insert("notes", "r1", {"body": "hello world"})

    rows = db.dumpdb()
    assert len(rows) == 1
    row = rows[0]
    assert row["partition_key"] == "notes"
    assert row["row_key"] == "r1"
    assert row["body"] == "hello world"
    assert "timestamp" in row
    assert "index_data" not in row


def test_insert_sets_keys_and_timestamp(monkeypatch):
    random.seed(0)
    monkeypatch.setattr(memorydb, "time", lambda: 1234.5)
    db = memorydb.MemoryDb()
    item = {"body": "hello"}

    db.insert("docs", "r1", item)

    assert item["partition_key"] == "docs"
    assert item["row_key"] == "r1"
    assert item["timestamp"] == 1234.5
    assert db.dumpdb()[0] == {
        "body": "hello",
        "partition_key": "docs",
        "row_key": "r1",
        "timestamp": 1234.5,
    }


def test_insert_builds_weighted_index_data():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"body": {"weight": 2}})

    db.insert("docs", "r1", {"body": "a b a"})

    # runs: a(1), b(1), a(1); each run weighted by 2 -> a=4, b=2
    assert db.dumpdb()[0]["index_data"] == [("a", 4), ("b", 2)]


def test_insert_combines_configured_fields_and_ignores_others():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"title": {"weight": 2}, "body": {}})

    db.insert("docs", "r1", {"title": "a", "body": "a b", "note": "q"})

    # title: a=2 ; body: a=1, b=1 ; "note" is not configured and is ignored
    assert db.dumpdb()[0]["index_data"] == [("a", 3), ("b", 1)]


def test_insert_stores_empty_index_data_when_no_configured_field_present():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"title": {}})

    db.insert("docs", "r1", {"body": "something"})

    assert db.dumpdb()[0]["index_data"] == []


def test_insert_stores_copy_so_later_changes_do_not_leak():
    random.seed(0)
    db = memorydb.MemoryDb()
    item = {"body": "original"}

    db.insert("docs", "r1", item)
    item["body"] = "changed"
    item["extra"] = 1

    row = db.dumpdb()[0]
    assert row["body"] == "original"
    assert "extra" not in row


# ---------------------------------------------------------------------------
# MemoryDb.textAnalysis
# ---------------------------------------------------------------------------

def test_text_analysis_sums_and_spreads_per_term():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"body": {}})
    db.insert("docs", "d1", {"body": "apple apple banana"})
    db.insert("docs", "d2", {"body": "apple cherry"})
    db.insert("docs", "d3", {"body": "banana banana"})

    analysis = db.textAnalysis("docs")

    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    data = analysis.dump()
    assert sorted(data.keys()) == ["apple", "banana", "cherry"]
    assert data["apple"]["sum"] == 3
    assert data["banana"]["sum"] == 3
    assert data["cherry"]["sum"] == 1
    assert data["apple"]["spread"] == {2: {"value": "d1"}, 1: {"value": "d2"}}
    assert data["banana"]["spread"] == {1: {"value": "d1"}, 2: {"value": "d3"}}
    assert data["cherry"]["spread"] == {1: {"value": "d2"}}


def test_text_analysis_chains_rows_sharing_term_and_count():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"body": {}})
    db.insert("docs", "d1", {"body": "apple"})
    db.insert("docs", "d2", {"body": "apple"})
    db.insert("docs", "d3", {"body": "apple"})

    data = db.textAnalysis("docs").dump()

    assert data == {
        "apple": {
            "sum": 3,
            "spread": {
                1: {"value": "d1", "next": {"value": "d2", "next": {"value": "d3"}}},
            },
        }
    }


def test_text_analysis_counts_each_run_of_repeated_tokens():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("docs", {"body": {}})
    db.insert("docs", "r1", {"body": "a a a b a"})

    data = db.textAnalysis("docs").dump()

    # runs a(3), b(1), a(1) -> a=4, b=1
    assert data == {
        "a": {"sum": 4, "spread": {4: {"value": "r1"}}},
        "b": {"sum": 1, "spread": {1: {"value": "r1"}}},
    }


def test_text_analysis_returns_none_without_index_data():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.insert("docs", "r1", {"body": "apple"})   # inserted before indexing
    db.textIndex("docs", {"title": {}})
    db.insert("docs", "r2", {"body": "pear"})    # indexed, but no configured field

    assert db.textAnalysis("docs") is None
    assert db.textAnalysis("empty") is None


def test_text_analysis_only_uses_requested_partition():
    random.seed(0)
    db = memorydb.MemoryDb()
    db.textIndex("a", {"body": {}})
    db.textIndex("b", {"body": {}})
    db.insert("a", "r1", {"body": "apple"})
    db.insert("b", "r2", {"body": "pear pear"})

    assert db.textAnalysis("a").dump() == {
        "apple": {"sum": 1, "spread": {1: {"value": "r1"}}}
    }
    assert sorted(db.textAnalysis("b").dump().keys()) == ["pear"]


# ---------------------------------------------------------------------------
# FullTextAnalysis (text_analysis.py)
# ---------------------------------------------------------------------------

def test_full_text_analysis_dump_and_terms():
    data = {
        "apple": {"sum": 3, "spread": {2: {"value": "d1"}, 1: {"value": "d2"}}},
        "pear": {"sum": 1, "spread": {1: {"value": "d3"}}},
    }
    analysis = text_analysis.FullTextAnalysis(data)

    assert analysis.dump() is data
    assert sorted(analysis.terms()) == ["apple", "pear"]


def test_full_text_analysis_scores_and_top():
    data = {
        "apple": {
            "sum": 6,
            "spread": {
                1: {"value": "d2"},
                3: {"value": "d1", "next": {"value": "d4"}},
                2: {"value": "d3"},
            },
        }
    }
    analysis = text_analysis.FullTextAnalysis(data)

    assert analysis.scores("apple") == [3, 2, 1]
    assert analysis.top("apple") == "d1"
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_full_text_analysis_all_collects_chain_rows_repeatably():
    data = {
        "apple": {
            "sum": 4,
            "spread": {
                2: {"value": "d1", "next": {"value": "d2"}},
                1: {"value": "d3"},
            },
        }
    }
    analysis = text_analysis.FullTextAnalysis(data)

    assert sorted(analysis.all("apple")) == ["d1", "d2", "d3"]
    # a second call must return the same rows
    assert sorted(analysis.all("apple")) == ["d1", "d2", "d3"]


def test_full_text_analysis_filter_selects_matching_terms():
    data = {
        "apple": {"sum": 3, "spread": {1: {"value": "d1"}}},
        "apricot": {"sum": 1, "spread": {1: {"value": "d2"}}},
        "banana": {"sum": 2, "spread": {2: {"value": "d3"}}},
    }
    analysis = text_analysis.FullTextAnalysis(data)

    selected = analysis.filter(lambda term: term.startswith("ap"))

    assert sorted(selected.keys()) == ["apple", "apricot"]
    assert selected["apple"] is data["apple"]
    assert analysis.filter(lambda term: False) == {}
    assert analysis.filter("ap") == []
    assert analysis.filter(None) == []


def test_full_text_analysis_serialize_orders_by_count_then_chain():
    data = {
        "apple": {
            "sum": 4,
            "spread": {
                2: {"value": "r1", "next": {"value": "r2"}},
                1: {"value": "r3"},
            },
        }
    }
    analysis = text_analysis.FullTextAnalysis(data)

    selected = analysis.filter(lambda term: True)
    assert analysis.serialize(selected) == ["r1", "r2", "r3"]
    assert analysis.serialize(None) is None
    assert analysis.serialize({}) == []

    # a row shared by two terms appears only once in the output
    shared = text_analysis.FullTextAnalysis({
        "apple": {"sum": 1, "spread": {1: {"value": "r1"}}},
        "pear": {"sum": 2, "spread": {2: {"value": "r1", "next": {"value": "r2"}}}},
    })
    ids = shared.serialize(shared.filter(lambda term: True))
    assert len(ids) == 2
    assert sorted(ids) == ["r1", "r2"]


def test_full_text_analysis_highlight_wraps_whole_words_only():
    analysis = text_analysis.FullTextAnalysis({})
    text = "the cat sat, catalog and Cat"

    assert analysis.highlight("cat", text, "<b>{{_}}</b>") == \
        "the <b>cat</b> sat, catalog and Cat"
    assert analysis.highlight("dog", text, "<b>{{_}}</b>") == text
