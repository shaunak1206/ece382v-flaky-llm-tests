import pytest

try:
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis
except Exception:  # pragma: no cover
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis


def test_text_index_creates_partition_scope_and_merges_fields():
    db = MemoryDb()

    db.textIndex("posts", {"title": 2, "body": 1})
    db.textIndex("posts", {"tags": 3})

    assert "posts" in db.all_indices
    assert db.all_indices["posts"] == {"title": 2, "body": 1, "tags": 3}


def test_insert_adds_keys_timestamp_and_copies_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("pymemorydb.memorydb.time", lambda: 123.456, raising=False)

    item = {"name": "Alice"}
    db.insert("users", "u1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "users"
    assert stored["row_key"] == "u1"
    assert stored["timestamp"] == 123.456
    assert stored["name"] == "Alice"
    assert "partition_key" not in item
    assert "row_key" not in item
    assert "timestamp" not in item


def test_insert_attaches_index_data_when_partition_is_indexed(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("pymemorydb.memorydb.time", lambda: 1.0, raising=False)

    db.textIndex("posts", {"title": 1, "body": {"weight": 2}})
    db.insert("posts", "p1", {"title": "hello hello", "body": "hello world world"})

    stored = db.table[0]
    assert "index_data" in stored
    assert stored["index_data"] == [("hello", 4), ("world", 4)]


def test_index_item_returns_none_without_index_configuration():
    db = MemoryDb()
    assert db.indexItem({"partition_key": "nope", "row_key": "1", "title": "x"}) is None


def test_get_returns_single_item_and_raises_for_missing_row():
    db = MemoryDb()
    db.insert("users", "u1", {"name": "Alice"})

    item = db.get("users", "u1")
    assert item["name"] == "Alice"

    with pytest.raises(ValueError) as exc:
        db.get("users", "missing")
    assert exc.value.args[0] == 404


def test_get_with_list_returns_items_in_requested_order():
    db = MemoryDb()
    db.insert("users", "u1", {"name": "Alice"})
    db.insert("users", "u2", {"name": "Bob"})

    items = db.get("users", ["u2", "u1"])
    assert [i["row_key"] for i in items] == ["u2", "u1"]


def test_get_raises_for_invalid_param_type():
    db = MemoryDb()
    with pytest.raises(ValueError, match="row_key or an array of row keys"):
        db.get("users", 123)


def test_delete_removes_matching_row_only():
    db = MemoryDb()
    db.insert("users", "u1", {"name": "Alice"})
    db.insert("users", "u2", {"name": "Bob"})
    db.delete("users", "u1")

    assert [i["row_key"] for i in db.table] == ["u2"]


def test_getall_sorts_by_timestamp(monkeypatch):
    db = MemoryDb()
    times = iter([20.0, 10.0, 30.0])
    monkeypatch.setattr("pymemorydb.memorydb.time", lambda: next(times), raising=False)

    db.insert("users", "u1", {"name": "Alice"})
    db.insert("users", "u2", {"name": "Bob"})
    db.insert("users", "u3", {"name": "Carol"})

    assert [i["row_key"] for i in db.getAll("users")] == ["u2", "u1", "u3"]


def test_query_filters_and_sorts_by_ts_descending():
    db = MemoryDb()
    db.table = [
        {"partition_key": "p", "row_key": "a", "ts": 1, "active": True},
        {"partition_key": "p", "row_key": "b", "ts": 3, "active": False},
        {"partition_key": "p", "row_key": "c", "ts": 2, "active": True},
    ]

    results = db.query("p", lambda item: item["active"])
    assert [i["row_key"] for i in results] == ["c", "a"]


def test_query_raises_for_non_callable_expression():
    db = MemoryDb()
    with pytest.raises(ValueError, match="expression function is required"):
        db.query("p", "not callable")


def test_update_and_changeid_replace_existing_item():
    db = MemoryDb()
    db.insert("users", "u1", {"name": "Alice"})

    db.update({"partition_key": "users", "row_key": "u1", "name": "Alicia"})
    assert db.get("users", "u1")["name"] == "Alicia"

    db.changeId({"partition_key": "users", "row_key": "u1", "name": "Alicia"}, "u9")
    assert db.get("users", "u9")["name"] == "Alicia"

    with pytest.raises(ValueError, match="item.partition_key is required"):
        db.update({"row_key": "x"})

    with pytest.raises(ValueError, match="item.row_key is required"):
        db.changeId({"partition_key": "users"}, "u2")


def test_updateall_replaces_multiple_items():
    db = MemoryDb()
    db.insert("users", "u1", {"name": "Alice"})
    db.insert("users", "u2", {"name": "Bob"})

    db.updateAll(
        [
            {"partition_key": "users", "row_key": "u1", "name": "Alicia"},
            {"partition_key": "users", "row_key": "u2", "name": "Bobby"},
        ]
    )

    assert [i["name"] for i in db.getAll("users")] == ["Alicia", "Bobby"]


def test_text_analysis_builds_full_text_analysis_from_indexed_rows():
    db = MemoryDb()
    db.textIndex("posts", {"title": 1})

    db.insert("posts", "p1", {"title": "hello hello"})
    db.insert("posts", "p2", {"title": "hello world"})
    db.insert("posts", "p3", {"title": "world"})

    analysis = db.textAnalysis("posts")
    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ["hello", "world"]

    assert analysis.scores("hello") == [2]
    assert analysis.top("hello") == "p1"
    assert analysis.all("hello") == ["p1", "p2"]
    assert analysis.top("world") == "p2"


def test_text_analysis_returns_none_when_no_indexed_items_exist():
    db = MemoryDb()
    db.insert("posts", "p1", {"title": "hello"})
    assert db.textAnalysis("posts") is None


def test_fulltextanalysis_dump_terms_and_missing_term_behavior():
    analysis = FullTextAnalysis({"x": {"sum": 3, "spread": {1: {"value": "a"}}}})

    assert analysis.dump() == {"x": {"sum": 3, "spread": {1: {"value": "a"}}}}
    assert analysis.terms() == ["x"]
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None
    assert analysis.all("missing") == []


def test_fulltextanalysis_filter_and_serialize():
    data = {
        "alpha": {
            "sum": 5,
            "spread": {
                2: {"value": "id2", "next": {"value": "id3"}},
                1: {"value": "id1"},
            },
        },
        "beta": {"sum": 1, "spread": {1: {"value": "id4"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term.startswith("a"))
    assert list(filtered.keys()) == ["alpha"]
    assert filtered["alpha"]["sum"] == 5

    serialized = analysis.serialize(filtered)
    assert serialized == ["id2", "id3", "id1"]


def test_fulltextanalysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})
    text = "hello there, shell and hello!"
    rendered = analysis.highlight("hello", text, "<b>{{_}}</b>")

    assert rendered == "<b>hello</b> there, shell and <b>hello</b>!"


def test_sample_returns_empty_list_for_empty_partition():
    db = MemoryDb()
    assert db.sample("missing") == []


def test_sample_returns_items_from_partition(monkeypatch):
    db = MemoryDb()
    db.table = [
        {"partition_key": "p", "row_key": "a"},
        {"partition_key": "p", "row_key": "b"},
        {"partition_key": "p", "row_key": "c"},
    ]

    monkeypatch.setattr("pymemorydb.memorydb.randint", lambda a, b: 1, raising=False)
    assert [i["row_key"] for i in db.sample("p", 3)] == ["b", "b", "b"]
