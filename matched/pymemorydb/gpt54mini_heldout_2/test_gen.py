import pytest

try:
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis


def make_db_with_indexed_items():
    db = MemoryDb()
    db.textIndex("docs", {"body": {}})
    db.insert("docs", "a", {"body": "hello hello world"})
    db.insert("docs", "b", {"body": "hello there"})
    db.insert("docs", "c", {"body": "world world"})
    return db


def test_text_index_registers_partition_fields():
    db = MemoryDb()
    db.textIndex("docs", {"title": {"weight": 2}, "body": {}})

    assert "docs" in db.all_indices
    assert db.all_indices["docs"]["title"]["weight"] == 2
    assert db.all_indices["docs"]["body"] == {}


def test_text_index_merges_additional_fields_into_existing_partition():
    db = MemoryDb()
    db.textIndex("docs", {"title": {"weight": 2}})
    db.textIndex("docs", {"body": {}})

    assert set(db.all_indices["docs"].keys()) == {"title", "body"}


def test_insert_adds_partition_row_and_timestamp_and_copies_item():
    db = MemoryDb()
    item = {"name": "alpha"}
    db.insert("docs", "1", item)

    stored = db.get("docs", "1")
    assert stored["partition_key"] == "docs"
    assert stored["row_key"] == "1"
    assert "timestamp" in stored
    assert "partition_key" not in item
    assert "row_key" not in item


def test_insert_populates_index_data_for_configured_text_fields():
    db = MemoryDb()
    db.textIndex("docs", {"body": {}})
    db.insert("docs", "1", {"body": "hello hello world"})

    stored = db.get("docs", "1")
    assert stored["index_data"] == [("hello", 2), ("world", 1)]


def test_insert_uses_weight_when_defined_in_text_index():
    db = MemoryDb()
    db.textIndex("docs", {"body": {"weight": 3}})
    db.insert("docs", "1", {"body": "hello hello world"})

    stored = db.get("docs", "1")
    assert stored["index_data"] == [("hello", 6), ("world", 3)]


def test_insert_does_not_add_index_data_when_partition_has_no_index():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "hello world"})

    stored = db.get("docs", "1")
    assert "index_data" not in stored


def test_get_returns_single_item_and_raises_404_for_missing_row():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "alpha"})

    assert db.get("docs", "1")["body"] == "alpha"
    with pytest.raises(ValueError) as exc:
        db.get("docs", "missing")
    assert exc.value.args[0] == 404


def test_get_with_list_returns_items_in_requested_order():
    db = MemoryDb()
    db.insert("docs", "a", {"body": "first"})
    db.insert("docs", "b", {"body": "second"})
    db.insert("docs", "c", {"body": "third"})

    result = db.get("docs", ["c", "a"])
    assert [item["row_key"] for item in result] == ["c", "a"]


def test_get_all_is_sorted_by_timestamp():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "first"})
    db.insert("docs", "2", {"body": "second"})
    db.insert("docs", "3", {"body": "third"})

    result = db.getAll("docs")
    assert [item["row_key"] for item in result] == ["1", "2", "3"]


def test_delete_removes_only_matching_row():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "keep"})
    db.insert("docs", "2", {"body": "remove"})
    db.delete("docs", "2")

    assert [item["row_key"] for item in db.getAll("docs")] == ["1"]


def test_query_requires_callable_expression():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "alpha"})

    with pytest.raises(ValueError):
        db.query("docs", "not callable")


def test_query_filters_items_and_orders_by_ts_descending_when_present():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "alpha", "ts": 1})
    db.insert("docs", "2", {"body": "beta", "ts": 3})
    db.insert("docs", "3", {"body": "gamma", "ts": 2})

    result = db.query("docs", lambda item: True)
    assert [item["row_key"] for item in result] == ["2", "3", "1"]


def test_change_id_moves_item_to_new_row_key():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "alpha"})
    item = db.get("docs", "1")

    db.changeId(item, "new")
    assert db.get("docs", "new")["body"] == "alpha"
    with pytest.raises(ValueError):
        db.get("docs", "1")


def test_update_replaces_existing_item():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "old"})
    updated = db.get("docs", "1")
    updated["body"] = "new"

    db.update(updated)
    assert db.get("docs", "1")["body"] == "new"


def test_text_analysis_returns_full_text_analysis_with_terms_scores_top_and_all():
    db = make_db_with_indexed_items()

    analysis = db.textAnalysis("docs")
    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"hello", "world", "there"}
    assert analysis.scores("hello") == [2, 1]
    assert analysis.top("hello") == "a"
    assert analysis.all("hello") == ["a", "b"]


def test_text_analysis_returns_none_when_partition_has_no_indexed_items():
    db = MemoryDb()
    db.insert("docs", "1", {"body": "plain text without index"})
    assert db.textAnalysis("docs") is None


def test_full_text_analysis_filter_returns_matching_terms_sorted_by_sum():
    data = {
        "hello": {"sum": 5, "spread": {2: {"value": "a"}, 1: {"value": "b"}}},
        "world": {"sum": 2, "spread": {1: {"value": "c"}}},
        "there": {"sum": 3, "spread": {3: {"value": "d"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: "o" in term)
    assert list(filtered.keys()) == ["hello", "world"]


def test_full_text_analysis_serialize_deduplicates_ids_in_score_order():
    data = {
        "hello": {"sum": 5, "spread": {2: {"value": "a", "next": {"value": "b"}}, 1: {"value": "c"}}},
        "world": {"sum": 2, "spread": {1: {"value": "b"}, 3: {"value": "d"}}},
    }
    analysis = FullTextAnalysis(data)

    assert analysis.serialize(data) == ["a", "b", "c", "d"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})
    result = analysis.highlight("cat", "cat scatter catalog cat", "<b>{{_}}</b>")

    assert result == "<b>cat</b> scatter catalog <b>cat</b>"
