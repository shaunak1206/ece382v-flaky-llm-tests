import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


def test_textindex_creates_partition_scope_and_stores_descriptor():
    db = MemoryDb()
    db.textIndex("books", {"title": {}})

    assert "books" in db.all_indices
    assert db.all_indices["books"]["title"] == {}


def test_textindex_merges_multiple_calls_into_same_partition():
    db = MemoryDb()
    db.textIndex("books", {"title": {}})
    db.textIndex("books", {"body": {"weight": 2}})

    assert db.all_indices["books"]["title"] == {}
    assert db.all_indices["books"]["body"] == {"weight": 2}


def test_insert_adds_partition_row_timestamp_and_copy(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 123.456)

    item = {"name": "Alice"}
    db.insert("users", "u1", item)

    stored = db.get("users", "u1")
    assert stored["partition_key"] == "users"
    assert stored["row_key"] == "u1"
    assert stored["timestamp"] == 123.456
    assert stored["name"] == "Alice"

    item["name"] = "Bob"
    assert db.get("users", "u1")["name"] == "Alice"


def test_insert_without_indexing_does_not_add_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.insert("users", "u1", {"name": "Alice"})
    stored = db.get("users", "u1")

    assert "index_data" not in stored


def test_insert_with_text_index_adds_index_data_for_tokenized_field(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("docs", {"title": {}})
    db.insert("docs", "d1", {"title": "hello hello world"})

    stored = db.get("docs", "d1")
    assert "index_data" in stored
    assert stored["index_data"] == [("hello", 2), ("world", 1)]


def test_insert_with_weighted_text_index_applies_weight(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("docs", {"title": {"weight": 3}})
    db.insert("docs", "d1", {"title": "alpha alpha beta"})

    stored = db.get("docs", "d1")
    assert stored["index_data"] == [("alpha", 6), ("beta", 3)]


def test_insert_ignores_nonindexed_fields_in_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("docs", {"title": {}})
    db.insert("docs", "d1", {"title": "alpha beta", "body": "ignored ignored"})

    stored = db.get("docs", "d1")
    assert stored["index_data"] == [("alpha", 1), ("beta", 1)]


def test_text_analysis_returns_none_when_no_indexed_items_exist():
    db = MemoryDb()
    db.textIndex("docs", {"title": {}})
    db.insert("docs", "d1", {"title": "plain text without index data"})

    assert db.textAnalysis("docs") is None


def test_text_analysis_builds_fulltextanalysis_for_multiple_items():
    db = MemoryDb()
    db.textIndex("docs", {"title": {}})
    db.insert("docs", "d1", {"title": "alpha beta"})
    db.insert("docs", "d2", {"title": "alpha alpha"})
    db.insert("docs", "d3", {"title": "beta"})

    analysis = db.textAnalysis("docs")

    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"alpha", "beta"}
    assert analysis.scores("alpha") == [2, 1]
    assert analysis.top("alpha") == "d2"
    assert analysis.all("alpha") == ["d2", "d1", "d2"]
    assert analysis.top("beta") == "d1"


def test_text_analysis_uses_numeric_scores_and_spread_mapping():
    db = MemoryDb()
    db.textIndex("docs", {"title": {}})
    db.insert("docs", "d1", {"title": "gamma"})
    db.insert("docs", "d2", {"title": "gamma gamma"})
    analysis = db.textAnalysis("docs")

    dumped = analysis.dump()
    assert "gamma" in dumped
    assert dumped["gamma"]["sum"] == 3
    assert set(dumped["gamma"]["spread"].keys()) == {1, 2}


def test_fulltextanalysis_scores_returns_sorted_numeric_keys_descending():
    analysis = FullTextAnalysis(
        {
            "term": {
                "sum": 6,
                "spread": {
                    1: {"value": "a"},
                    3: {"value": "b"},
                    2: {"value": "c"},
                },
            }
        }
    )

    assert analysis.scores("term") == [3, 2, 1]


def test_fulltextanalysis_scores_and_top_return_none_for_missing_term():
    analysis = FullTextAnalysis({})

    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_fulltextanalysis_all_collects_values_across_next_chain():
    analysis = FullTextAnalysis(
        {
            "term": {
                "sum": 3,
                "spread": {
                    2: {"value": "first", "next": {"value": "second"}},
                    1: {"value": "third"},
                },
            }
        }
    )

    assert analysis.all("term") == ["first", "second", "third"]


def test_fulltextanalysis_filter_returns_only_matching_terms():
    analysis = FullTextAnalysis(
        {
            "alpha": {"sum": 5, "spread": {1: {"value": "a"}}},
            "beta": {"sum": 2, "spread": {1: {"value": "b"}}},
        }
    )

    filtered = analysis.filter(lambda term: term.startswith("a"))

    assert list(filtered.keys()) == ["alpha"]
    assert filtered["alpha"]["sum"] == 5


def test_fulltextanalysis_filter_non_callable_returns_empty_list():
    analysis = FullTextAnalysis({"alpha": {"sum": 1, "spread": {1: {"value": "a"}}}})

    assert analysis.filter(None) == []


def test_fulltextanalysis_serialize_returns_unique_values_in_score_order():
    analysis = FullTextAnalysis(
        {
            "alpha": {
                "sum": 3,
                "spread": {
                    2: {"value": "x", "next": {"value": "y"}},
                    1: {"value": "z"},
                },
            },
            "beta": {
                "sum": 2,
                "spread": {
                    1: {"value": "y"},
                },
            },
        }
    )

    result = analysis.serialize(analysis.dump())

    assert result == ["x", "y", "z"]


def test_fulltextanalysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})

    text = "cat scatter catty cat"
    rendered = analysis.highlight("cat", text, "<b>{{_}}</b>")

    assert rendered == "<b>cat</b> scatter catty <b>cat</b>"


def test_fulltextanalysis_highlight_is_case_insensitive_for_replacement_text():
    analysis = FullTextAnalysis({})

    text = "Dog dog DOG"
    rendered = analysis.highlight("dog", text, "<i>{{_}}</i>")

    assert rendered == "<i>dog</i> <i>dog</i> <i>dog</i>"
