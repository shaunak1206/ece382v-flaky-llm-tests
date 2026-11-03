import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


def test_text_index_creates_and_merges_partition_fields():
    db = MemoryDb()

    db.textIndex("p1", {"title": {"weight": 2}})
    db.textIndex("p1", {"body": {"weight": 1}})

    assert db.all_indices["p1"]["title"]["weight"] == 2
    assert db.all_indices["p1"]["body"]["weight"] == 1


def test_insert_adds_partition_row_timestamp_and_copy(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 123.456, raising=False)

    item = {"name": "alpha"}
    db.insert("p1", "r1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "p1"
    assert stored["row_key"] == "r1"
    assert stored["timestamp"] == 123.456

    item["name"] = "changed"
    assert db.table[0]["name"] == "alpha"


def test_insert_indexes_text_fields_when_text_index_defined(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)
    db.textIndex("p1", {"title": {"weight": 2}, "body": {"weight": 1}})

    db.insert("p1", "r1", {"title": "hello hello world", "body": "world"})

    stored = db.table[0]
    assert stored["index_data"] == [("hello", 4), ("world", 3)]


def test_insert_without_text_index_does_not_add_index_data(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)

    db.insert("p1", "r1", {"title": "hello world"})

    assert "index_data" not in db.table[0]


def test_get_by_row_key_returns_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)

    db.insert("p1", "r1", {"v": 1})
    assert db.get("p1", "r1")["v"] == 1


def test_get_by_row_key_raises_404_for_missing_item():
    db = MemoryDb()
    with pytest.raises(ValueError) as exc:
        db.get("p1", "missing")
    assert exc.value.args[0] == 404


def test_get_with_list_returns_items_in_requested_order(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)

    db.insert("p1", "r1", {"v": 1})
    db.insert("p1", "r2", {"v": 2})
    db.insert("p1", "r3", {"v": 3})

    results = db.get("p1", ["r3", "r1"])
    assert [item["row_key"] for item in results] == ["r3", "r1"]
    assert [item["v"] for item in results] == [3, 1]


def test_query_orders_by_ts_descending(monkeypatch):
    db = MemoryDb()
    times = iter([10.0, 20.0, 30.0])
    monkeypatch.setattr("memorydb.time", lambda: next(times), raising=False)

    db.insert("p1", "r1", {"v": 1, "ts": 100})
    db.insert("p1", "r2", {"v": 2, "ts": 200})
    db.insert("p1", "r3", {"v": 3, "ts": 150})

    results = db.query("p1", lambda _: True)
    assert [item["row_key"] for item in results] == ["r2", "r3", "r1"]


def test_query_requires_callable_expression():
    db = MemoryDb()
    with pytest.raises(ValueError) as exc:
        db.query("p1", "not callable")
    assert "expression function is required" in str(exc.value)


def test_update_replaces_existing_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)

    db.insert("p1", "r1", {"v": 1})
    db.update({"partition_key": "p1", "row_key": "r1", "v": 99})

    assert len(db.table) == 1
    assert db.table[0]["v"] == 99


def test_change_id_updates_row_key(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)

    db.insert("p1", "r1", {"v": 1})
    item = db.get("p1", "r1")
    db.changeId(item, "r2")

    assert len(db.table) == 1
    assert db.table[0]["row_key"] == "r2"
    assert db.get("p1", "r2")["v"] == 1


def test_text_analysis_returns_full_text_analysis_with_expected_terms(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0, raising=False)
    db.textIndex("p1", {"title": {"weight": 1}})

    db.insert("p1", "r1", {"title": "apple banana"})
    db.insert("p1", "r2", {"title": "apple apple"})
    db.insert("p1", "r3", {"title": "banana"})

    analysis = db.textAnalysis("p1")
    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"apple", "banana"}

    data = analysis.dump()
    assert data["apple"]["sum"] == 3
    assert data["banana"]["sum"] == 2


def test_full_text_analysis_scores_top_all_and_serialize():
    data = {
        "apple": {
            "sum": 3,
            "spread": {
                2: {"value": "r2", "next": {"value": "r1"}},
                1: {"value": "r3"},
            },
        },
        "banana": {
            "sum": 2,
            "spread": {
                1: {"value": "r1"},
                3: {"value": "r4"},
            },
        },
    }
    analysis = FullTextAnalysis(data)

    assert analysis.scores("apple") == [2, 1]
    assert analysis.top("apple") == "r2"
    assert analysis.all("apple") == ["r2", "r1", "r3"]

    serialized = analysis.serialize({"apple": data["apple"], "banana": data["banana"]})
    assert serialized == ["r2", "r1", "r3", "r4"]


def test_full_text_analysis_filter_returns_matching_terms():
    data = {
        "apple": {"sum": 3, "spread": {2: {"value": "r2"}}},
        "banana": {"sum": 1, "spread": {1: {"value": "r1"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term.startswith("a"))
    assert list(filtered.keys()) == ["apple"]
    assert filtered["apple"]["sum"] == 3


def test_full_text_analysis_highlight_replaces_whole_words_only():
    analysis = FullTextAnalysis({})
    text = "apple pineapple apple"
    rendered = analysis.highlight("apple", text, "<b>{{_}}</b>")

    assert rendered == "<b>apple</b> pineapple <b>apple</b>"


def test_full_text_analysis_missing_term_behaviors():
    analysis = FullTextAnalysis({})

    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None
    assert analysis.filter(None) == []
    assert analysis.serialize(None) is None
