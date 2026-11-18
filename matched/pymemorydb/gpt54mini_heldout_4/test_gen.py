import pytest

try:
    from memorydb import MemoryDb
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb

try:
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.text_analysis import FullTextAnalysis


def test_text_index_creates_and_updates_partition_schema():
    db = MemoryDb()

    db.textIndex("posts", {"title": {"weight": 2}})
    assert "posts" in db.all_indices
    assert db.all_indices["posts"]["title"] == {"weight": 2}

    db.textIndex("posts", {"body": {"weight": 1}})
    assert db.all_indices["posts"]["title"] == {"weight": 2}
    assert db.all_indices["posts"]["body"] == {"weight": 1}


def test_insert_sets_partition_row_timestamp_and_copies_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 123.456)

    item = {"name": "Alice"}
    db.insert("users", "u1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "users"
    assert stored["row_key"] == "u1"
    assert stored["timestamp"] == 123.456
    assert "partition_key" not in item
    assert "row_key" not in item
    assert "timestamp" not in item


def test_insert_adds_index_data_when_index_defined(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.textIndex("posts", {"title": {"weight": 2}, "body": {"weight": 1}})
    db.insert("posts", "p1", {"title": "hello hello", "body": "world"})

    stored = db.get("posts", "p1")
    assert stored["index_data"] == [("hello", 4), ("world", 1)]


def test_insert_without_index_does_not_add_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.insert("posts", "p1", {"title": "hello world"})
    stored = db.get("posts", "p1")
    assert "index_data" not in stored


def test_get_returns_single_item_and_raises_for_missing():
    db = MemoryDb()
    db.table = [
        {"partition_key": "users", "row_key": "u1", "name": "Alice"},
    ]

    assert db.get("users", "u1")["name"] == "Alice"
    with pytest.raises(ValueError) as exc:
        db.get("users", "missing")
    assert exc.value.args[0] == 404


def test_get_list_returns_items_in_requested_order():
    db = MemoryDb()
    db.table = [
        {"partition_key": "users", "row_key": "u1", "name": "Alice"},
        {"partition_key": "users", "row_key": "u2", "name": "Bob"},
        {"partition_key": "users", "row_key": "u3", "name": "Cara"},
    ]

    results = db.get("users", ["u3", "u1"])
    assert [r["row_key"] for r in results] == ["u3", "u1"]


def test_get_all_sorts_by_timestamp_ascending():
    db = MemoryDb()
    db.table = [
        {"partition_key": "users", "row_key": "u2", "timestamp": 20},
        {"partition_key": "users", "row_key": "u1", "timestamp": 10},
        {"partition_key": "users", "row_key": "u3", "timestamp": 30},
    ]

    assert [x["row_key"] for x in db.getAll("users")] == ["u1", "u2", "u3"]


def test_query_orders_by_ts_descending_and_filters():
    db = MemoryDb()
    db.table = [
        {"partition_key": "users", "row_key": "u1", "ts": 1, "active": True},
        {"partition_key": "users", "row_key": "u2", "ts": 3, "active": False},
        {"partition_key": "users", "row_key": "u3", "ts": 2, "active": True},
    ]

    results = db.query("users", lambda item: item["active"])
    assert [x["row_key"] for x in results] == ["u3", "u1"]


def test_sample_returns_empty_list_for_empty_partition():
    db = MemoryDb()
    assert db.sample("missing") == []


def test_sample_uses_random_item_from_partition(monkeypatch):
    db = MemoryDb()
    db.table = [
        {"partition_key": "users", "row_key": "u1"},
        {"partition_key": "users", "row_key": "u2"},
        {"partition_key": "users", "row_key": "u3"},
    ]

    calls = []

    def fake_randint(a, b):
        calls.append((a, b))
        return 1

    monkeypatch.setattr("memorydb.randint", fake_randint)
    result = db.sample("users", 3)

    assert calls == [(1, 2), (1, 2), (1, 2)]
    assert [r["row_key"] for r in result] == ["u2", "u2", "u2"]


def test_text_analysis_builds_full_text_analysis_and_dump():
    db = MemoryDb()
    db.all_indices = {
        "posts": {"title": {"weight": 1}},
    }
    db.table = [
        {"partition_key": "posts", "row_key": "p1", "index_data": [("hello", 1)]},
        {"partition_key": "posts", "row_key": "p2", "index_data": [("hello", 2)]},
    ]

    analysis = db.textAnalysis("posts")
    assert isinstance(analysis, FullTextAnalysis)
    dumped = analysis.dump()
    assert "hello" in dumped
    assert dumped["hello"]["sum"] == 3


def test_text_analysis_returns_none_when_no_indexed_items_exist():
    db = MemoryDb()
    db.table = [
        {"partition_key": "posts", "row_key": "p1", "title": "hello"},
    ]

    assert db.textAnalysis("posts") is None


def test_full_text_analysis_terms_scores_top_and_all():
    data = {
        "hello": {
            "sum": 3,
            "spread": {
                2: {"value": "p2"},
                1: {"value": "p1"},
            },
        }
    }
    analysis = FullTextAnalysis(data)

    assert analysis.terms() == ["hello"]
    assert analysis.scores("hello") == [2, 1]
    assert analysis.top("hello") == "p2"
    assert analysis.all("hello") == ["p2", "p1"]


def test_full_text_analysis_filter_by_callable_and_reject_non_callable():
    data = {
        "hello": {"sum": 3, "spread": {1: {"value": "p1"}}},
        "world": {"sum": 1, "spread": {1: {"value": "p2"}}},
    }
    analysis = FullTextAnalysis(data)

    assert analysis.filter("not callable") == {}
    filtered = analysis.filter(lambda term: term.startswith("h"))
    assert list(filtered.keys()) == ["hello"]
    assert filtered["hello"]["sum"] == 3


def test_full_text_analysis_serialize_flattens_unique_values_in_score_order():
    data = {
        "hello": {
            "sum": 3,
            "spread": {
                2: {"value": "p2", "next": {"value": "p3"}},
                1: {"value": "p1"},
            },
        },
        "world": {
            "sum": 1,
            "spread": {
                1: {"value": "p3"},
            },
        },
    }
    analysis = FullTextAnalysis(data)

    serialized = analysis.serialize(data)
    assert serialized == ["p2", "p3", "p1"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})
    text = "hello shell hello!"
    result = analysis.highlight("hello", text, "<b>{{_}}</b>")

    assert result == "<b>hello</b> shell <b>hello</b>!"
