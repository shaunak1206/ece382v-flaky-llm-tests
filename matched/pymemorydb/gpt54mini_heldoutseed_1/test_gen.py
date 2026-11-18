import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


def test_text_index_creates_partition_and_merges_descriptors():
    db = MemoryDb()

    db.textIndex("posts", {"title": {"weight": 2}})
    db.textIndex("posts", {"body": 1})

    assert "posts" in db.all_indices
    assert db.all_indices["posts"]["title"] == {"weight": 2}
    assert db.all_indices["posts"]["body"] == 1


def test_insert_adds_partition_row_timestamp_and_copies_index_data():
    db = MemoryDb()
    db.textIndex("posts", {"title": {"weight": 2}})

    item = {"title": "hello world hello"}
    db.insert("posts", "1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "posts"
    assert stored["row_key"] == "1"
    assert "timestamp" in stored
    assert stored["title"] == "hello world hello"
    assert stored["index_data"] == [("hello", 4), ("world", 2)]
    assert "partition_key" not in item
    assert "row_key" not in item
    assert "timestamp" not in item


def test_insert_does_not_attach_index_data_when_no_index_exists():
    db = MemoryDb()

    db.insert("posts", "1", {"title": "hello world"})
    stored = db.table[0]

    assert "index_data" not in stored
    assert stored["title"] == "hello world"


def test_get_returns_single_item_by_row_key():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a"})
    db.insert("posts", "2", {"title": "b"})

    item = db.get("posts", "2")

    assert item["row_key"] == "2"
    assert item["title"] == "b"


def test_get_raises_404_for_missing_item():
    db = MemoryDb()

    with pytest.raises(ValueError) as exc:
        db.get("posts", "missing")

    assert exc.value.args[0] == 404


def test_get_returns_list_for_multiple_row_keys_in_order():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a"})
    db.insert("posts", "2", {"title": "b"})
    db.insert("posts", "3", {"title": "c"})

    items = db.get("posts", ["3", "1"])

    assert [item["row_key"] for item in items] == ["3", "1"]


def test_get_all_sorts_by_timestamp_ascending():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a"})
    db.insert("posts", "2", {"title": "b"})
    db.insert("posts", "3", {"title": "c"})

    items = db.getAll("posts")

    assert [item["row_key"] for item in items] == ["1", "2", "3"]


def test_query_requires_callable_expression():
    db = MemoryDb()

    with pytest.raises(ValueError) as exc:
        db.query("posts", "not-callable")

    assert "expression function is required" in str(exc.value)


def test_query_filters_items_and_sorts_by_ts_descending():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a", "ts": 10})
    db.insert("posts", "2", {"title": "b", "ts": 20})
    db.insert("posts", "3", {"title": "c", "ts": 15})

    results = db.query("posts", lambda item: item["row_key"] != "1")

    assert [item["row_key"] for item in results] == ["2", "3"]


def test_change_id_updates_row_key_and_keeps_item_in_table():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a"})
    item = db.get("posts", "1")

    db.changeId(item, "99")

    assert db.get("posts", "99")["title"] == "a"
    with pytest.raises(ValueError):
        db.get("posts", "1")


def test_update_replaces_existing_item():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "old"})
    updated = db.get("posts", "1")
    updated["title"] = "new"

    db.update(updated)

    assert db.get("posts", "1")["title"] == "new"


def test_stats_delegates_to_aggregate_run():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "a"})
    db.insert("posts", "2", {"title": "b"})

    class DummyAgg:
        def run(self, items):
            return {"count": len(items), "keys": [item["row_key"] for item in items]}

    result = db.stats("posts", DummyAgg())

    assert result == {"count": 2, "keys": ["1", "2"]}


def test_text_analysis_returns_full_text_analysis_with_terms_scores_top_and_all():
    db = MemoryDb()
    db.textIndex("posts", {"title": {"weight": 2}})
    db.insert("posts", "1", {"title": "alpha beta alpha"})
    db.insert("posts", "2", {"title": "alpha"})
    db.insert("posts", "3", {"title": "beta"})

    analysis = db.textAnalysis("posts")

    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"alpha", "beta"}
    assert analysis.scores("alpha") == [4, 2]
    assert analysis.top("alpha") == "1"
    assert analysis.all("alpha") == ["1", "2"]
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_text_analysis_returns_none_when_no_indexed_items_exist():
    db = MemoryDb()
    db.insert("posts", "1", {"title": "alpha beta"})

    assert db.textAnalysis("posts") is None


def test_full_text_analysis_filter_and_serialize():
    analysis = FullTextAnalysis(
        {
            "alpha": {
                "sum": 6,
                "spread": {
                    4: {"value": "a1", "next": {"value": "a2"}},
                    2: {"value": "a3"},
                },
            },
            "beta": {
                "sum": 2,
                "spread": {
                    2: {"value": "b1"},
                },
            },
        }
    )

    filtered = analysis.filter(lambda term: term == "alpha")
    assert list(filtered.keys()) == ["alpha"]

    serialized = analysis.serialize(filtered)
    assert serialized == ["a1", "a2", "a3"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})

    result = analysis.highlight("cat", "cat scatter cat", "<b>{{_}}</b>")

    assert result == "<b>cat</b> scatter <b>cat</b>"


def test_full_text_analysis_dump_returns_original_data():
    data = {"alpha": {"sum": 1, "spread": {1: {"value": "x"}}}}
    analysis = FullTextAnalysis(data)

    assert analysis.dump() is data
