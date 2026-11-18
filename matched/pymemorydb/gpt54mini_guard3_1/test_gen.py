import pytest

import memorydb
import text_analysis


def test_textindex_creates_partition_index_mapping():
    db = memorydb.MemoryDb()

    db.textIndex("p1", {"title": {"weight": 2}, "body": 1})

    assert "p1" in db.all_indices
    assert db.all_indices["p1"]["title"] == {"weight": 2}
    assert db.all_indices["p1"]["body"] == 1


def test_textindex_merges_additional_fields_into_existing_partition():
    db = memorydb.MemoryDb()

    db.textIndex("p1", {"title": 2})
    db.textIndex("p1", {"body": {"weight": 3}})

    assert db.all_indices["p1"] == {"title": 2, "body": {"weight": 3}}


def test_insert_adds_partition_row_timestamp_and_copies_item(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 123.456)

    item = {"name": "alpha"}
    db.insert("p1", "r1", item)

    assert item == {"name": "alpha", "partition_key": "p1", "row_key": "r1", "timestamp": 123.456}
    stored = db.get("p1", "r1")
    assert stored is not item
    assert stored["partition_key"] == "p1"
    assert stored["row_key"] == "r1"
    assert stored["timestamp"] == 123.456


def test_insert_populates_index_data_when_text_index_is_defined(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 2}, "body": 1})
    db.insert("p1", "r1", {"title": "hello hello world", "body": "world hello"})

    stored = db.get("p1", "r1")
    assert stored["index_data"] == [("hello", 4), ("world", 2)]


def test_insert_does_not_add_index_data_without_defined_partition_index(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.insert("p1", "r1", {"title": "hello world"})

    stored = db.get("p1", "r1")
    assert "index_data" not in stored


def test_get_by_row_key_returns_inserted_item(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 2.0)

    db.insert("p1", "r1", {"value": 10})

    result = db.get("p1", "r1")
    assert result["value"] == 10
    assert result["partition_key"] == "p1"
    assert result["row_key"] == "r1"


def test_get_with_missing_row_key_raises_404(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 2.0)

    db.insert("p1", "r1", {"value": 10})

    with pytest.raises(ValueError) as excinfo:
        db.get("p1", "missing")
    assert excinfo.value.args[0] == 404


def test_get_all_returns_items_sorted_by_timestamp(monkeypatch):
    db = memorydb.MemoryDb()
    times = iter([30.0, 10.0, 20.0])
    monkeypatch.setattr(memorydb, "time", lambda: next(times))

    db.insert("p1", "r1", {"v": 1})
    db.insert("p1", "r2", {"v": 2})
    db.insert("p1", "r3", {"v": 3})

    result = db.getAll("p1")
    assert [item["row_key"] for item in result] == ["r2", "r3", "r1"]


def test_changeid_moves_row_key_and_preserves_item_contents(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.insert("p1", "r1", {"value": "x"})
    item = db.get("p1", "r1")

    db.changeId(item, "r2")

    assert db.get("p1", "r2")["value"] == "x"
    with pytest.raises(ValueError):
        db.get("p1", "r1")


def test_update_replaces_existing_item(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.insert("p1", "r1", {"value": "old"})
    updated = {"partition_key": "p1", "row_key": "r1", "value": "new"}

    db.update(updated)

    result = db.get("p1", "r1")
    assert result["value"] == "new"


def test_textanalysis_returns_fulltextanalysis_with_expected_terms_and_scores(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.textIndex("p1", {"title": {"weight": 2}})
    db.insert("p1", "r1", {"title": "alpha beta beta"})
    db.insert("p1", "r2", {"title": "alpha alpha"})
    analysis = db.textAnalysis("p1")

    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    assert set(analysis.terms()) == {"alpha", "beta"}
    assert analysis.scores("alpha") == [4, 2]
    assert analysis.top("alpha") == "r2"
    assert analysis.scores("beta") == [4]


def test_textanalysis_all_returns_all_rows_in_score_groups(monkeypatch):
    db = memorydb.MemoryDb()
    monkeypatch.setattr(memorydb, "time", lambda: 1.0)

    db.textIndex("p1", {"title": 1})
    db.insert("p1", "r1", {"title": "alpha beta"})
    db.insert("p1", "r2", {"title": "alpha"})
    db.insert("p1", "r3", {"title": "beta beta"})

    analysis = db.textAnalysis("p1")

    assert analysis.all("alpha") == ["r1", "r2"]
    assert analysis.all("beta") == ["r1", "r3"]


def test_fulltextanalysis_filter_and_serialize(monkeypatch):
    data = {
        "alpha": {"sum": 10, "spread": {4: {"value": "r2"}, 2: {"value": "r1"}}},
        "beta": {"sum": 3, "spread": {3: {"value": "r3"}}},
        "gamma": {"sum": 7, "spread": {7: {"value": "r4"}}},
    }
    analysis = text_analysis.FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term in {"alpha", "gamma"})
    assert list(filtered.keys()) == ["alpha", "gamma"]

    assert analysis.serialize(filtered) == ["r2", "r1", "r4"]


def test_fulltextanalysis_highlight_replaces_exact_term_only():
    analysis = text_analysis.FullTextAnalysis({})

    result = analysis.highlight("cat", "cat scatter catapult cat", "<em>{{_}}</em>")

    assert result == "<em>cat</em> scatter catapult <em>cat</em>"
