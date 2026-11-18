import importlib
import pytest


def _import_modules():
    candidates = [
        "memorydb",
        "pymemorydb.memorydb",
        "pymemorydb",
    ]
    mem_mod = None
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "MemoryDb"):
                mem_mod = mod
                break
        except Exception:
            continue
    if mem_mod is None:
        raise ImportError("Could not import MemoryDb module")

    ta_candidates = [
        "text_analysis",
        "pymemorydb.text_analysis",
    ]
    ta_mod = None
    for name in ta_candidates:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "FullTextAnalysis"):
                ta_mod = mod
                break
        except Exception:
            continue
    if ta_mod is None:
        raise ImportError("Could not import text_analysis module")

    return mem_mod, ta_mod


memorydb, text_analysis = _import_modules()
MemoryDb = memorydb.MemoryDb
FullTextAnalysis = text_analysis.FullTextAnalysis


def test_text_index_creates_partition_entry_and_merges_fields():
    db = MemoryDb()
    db.textIndex("p1", {"title": True})
    db.textIndex("p1", {"body": {"weight": 2}})
    assert "p1" in db.all_indices
    assert db.all_indices["p1"]["title"] is True
    assert db.all_indices["p1"]["body"] == {"weight": 2}


def test_insert_adds_partition_row_timestamp_and_copies_item(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 123.456)

    item = {"name": "alpha"}
    db.insert("p1", "r1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "p1"
    assert stored["row_key"] == "r1"
    assert stored["timestamp"] == 123.456
    assert stored["name"] == "alpha"
    assert "partition_key" not in item
    assert "row_key" not in item
    assert "timestamp" not in item


def test_insert_attaches_index_data_when_indexed(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)
    db.textIndex("p1", {"title": True, "body": {"weight": 2}})

    db.insert("p1", "r1", {"title": "hello hello", "body": "world"})

    stored = db.table[0]
    assert "index_data" in stored
    assert stored["index_data"] == [("hello", 2), ("world", 2)]


def test_insert_does_not_attach_index_data_without_text_index(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.insert("p1", "r1", {"title": "hello world"})

    stored = db.table[0]
    assert "index_data" not in stored


def test_dumpdb_returns_shallow_copy_of_table():
    db = MemoryDb()
    db.table.append({"partition_key": "p1", "row_key": "r1"})
    dumped = db.dumpdb()

    assert dumped == db.table
    assert dumped is not db.table


def test_get_returns_single_item_and_raises_for_missing():
    db = MemoryDb()
    db.table.append({"partition_key": "p1", "row_key": "r1", "x": 1})

    assert db.get("p1", "r1")["x"] == 1
    with pytest.raises(ValueError) as exc:
        db.get("p1", "missing")
    assert exc.value.args[0] == 404


def test_get_returns_items_for_multiple_row_keys_in_order():
    db = MemoryDb()
    db.table.extend(
        [
            {"partition_key": "p1", "row_key": "r1", "x": 1},
            {"partition_key": "p1", "row_key": "r2", "x": 2},
            {"partition_key": "p1", "row_key": "r3", "x": 3},
        ]
    )

    results = db.get("p1", ["r3", "r1"])
    assert [item["row_key"] for item in results] == ["r3", "r1"]


def test_get_raises_for_invalid_param_type():
    db = MemoryDb()
    with pytest.raises(ValueError):
        db.get("p1", 123)


def test_get_all_sorts_by_timestamp():
    db = MemoryDb()
    db.table.extend(
        [
            {"partition_key": "p1", "row_key": "r1", "timestamp": 3},
            {"partition_key": "p1", "row_key": "r2", "timestamp": 1},
            {"partition_key": "p1", "row_key": "r3", "timestamp": 2},
        ]
    )

    assert [x["row_key"] for x in db.getAll("p1")] == ["r2", "r3", "r1"]


def test_delete_removes_only_matching_row():
    db = MemoryDb()
    db.table.extend(
        [
            {"partition_key": "p1", "row_key": "r1"},
            {"partition_key": "p1", "row_key": "r2"},
            {"partition_key": "p2", "row_key": "r1"},
        ]
    )

    db.delete("p1", "r1")
    assert {(x["partition_key"], x["row_key"]) for x in db.table} == {("p1", "r2"), ("p2", "r1")}


def test_query_filters_and_sorts_by_ts_descending():
    db = MemoryDb()
    db.table.extend(
        [
            {"partition_key": "p1", "row_key": "r1", "ts": 10, "kind": "a"},
            {"partition_key": "p1", "row_key": "r2", "ts": 20, "kind": "a"},
            {"partition_key": "p1", "row_key": "r3", "ts": 15, "kind": "b"},
        ]
    )

    results = db.query("p1", lambda item: item["kind"] == "a")
    assert [x["row_key"] for x in results] == ["r2", "r1"]


def test_query_raises_for_non_callable_expression():
    db = MemoryDb()
    with pytest.raises(ValueError):
        db.query("p1", "not callable")


def test_change_id_updates_row_key_and_keeps_item_present():
    db = MemoryDb()
    item = {"partition_key": "p1", "row_key": "r1", "x": 1}
    db.table.append(item)

    db.changeId(item, "r9")

    assert len(db.table) == 1
    assert db.table[0]["row_key"] == "r9"
    assert db.table[0]["x"] == 1


def test_update_replaces_existing_item():
    db = MemoryDb()
    db.table.append({"partition_key": "p1", "row_key": "r1", "x": 1})
    updated = {"partition_key": "p1", "row_key": "r1", "x": 99}

    db.update(updated)

    assert len(db.table) == 1
    assert db.table[0]["x"] == 99


def test_text_analysis_returns_full_text_analysis_for_indexed_tokens(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)
    db.textIndex("p1", {"title": True, "body": {"weight": 2}})

    db.insert("p1", "r1", {"title": "alpha beta", "body": "beta"})
    db.insert("p1", "r2", {"title": "alpha", "body": "gamma"})
    result = db.textAnalysis("p1")

    assert isinstance(result, FullTextAnalysis)
    assert set(result.terms()) == {"alpha", "beta", "gamma"}
    assert result.top("alpha") == "r1"


def test_text_analysis_returns_none_when_no_index_data():
    db = MemoryDb()
    db.table.append({"partition_key": "p1", "row_key": "r1", "name": "x"})
    assert db.textAnalysis("p1") is None


def test_full_text_analysis_basic_methods():
    data = {
        "alpha": {"sum": 3, "spread": {2: {"value": "r2"}, 1: {"value": "r1"}}},
        "beta": {"sum": 1, "spread": {1: {"value": "r3"}}},
    }
    fta = FullTextAnalysis(data)

    assert fta.dump() == data
    assert set(fta.terms()) == {"alpha", "beta"}
    assert fta.scores("alpha") == [2, 1]
    assert fta.top("alpha") == "r2"
    assert fta.scores("missing") is None
    assert fta.top("missing") is None


def test_full_text_analysis_all_returns_all_values_across_spread():
    data = {
        "alpha": {
            "sum": 3,
            "spread": {
                2: {"value": "r2", "next": {"value": "r1"}},
                1: {"value": "r3"},
            },
        }
    }
    fta = FullTextAnalysis(data)

    assert fta.all("alpha") == ["r2", "r1", "r3"]


def test_full_text_analysis_filter_returns_matching_terms_sorted_by_sum():
    data = {
        "alpha": {"sum": 5, "spread": {1: {"value": "r1"}}},
        "beta": {"sum": 2, "spread": {1: {"value": "r2"}}},
        "gamma": {"sum": 9, "spread": {1: {"value": "r3"}}},
    }
    fta = FullTextAnalysis(data)

    filtered = fta.filter(lambda term: term.startswith(("a", "g")))
    assert list(filtered.keys()) == ["gamma", "alpha"]


def test_full_text_analysis_filter_non_callable_returns_empty_list():
    fta = FullTextAnalysis({"alpha": {"sum": 1, "spread": {1: {"value": "r1"}}}})
    assert fta.filter(None) == []


def test_full_text_analysis_serialize_deduplicates_values_and_preserves_order():
    data = {
        "alpha": {
            "sum": 3,
            "spread": {
                2: {"value": "r2", "next": {"value": "r1"}},
                1: {"value": "r3"},
            },
        },
        "beta": {
            "sum": 2,
            "spread": {
                1: {"value": "r1"},
            },
        },
    }
    fta = FullTextAnalysis(data)

    assert fta.serialize(data) == ["r2", "r1", "r3"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    fta = FullTextAnalysis({})
    text = "alpha alphabet alpha"
    rendered = "<b>{{_}}</b>"

    assert fta.highlight("alpha", text, rendered) == "<b>alpha</b> alphabet <b>alpha</b>"
