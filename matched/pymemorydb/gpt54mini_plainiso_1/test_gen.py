import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


def test_text_index_creates_partition_mapping():
    db = MemoryDb()
    db.textIndex("books", {"title": 2, "body": 1})

    assert "books" in db.all_indices
    assert db.all_indices["books"]["title"] == 2
    assert db.all_indices["books"]["body"] == 1


def test_text_index_merges_additional_fields_into_existing_partition():
    db = MemoryDb()
    db.textIndex("books", {"title": 2})
    db.textIndex("books", {"body": 1})

    assert db.all_indices["books"] == {"title": 2, "body": 1}


def test_insert_stores_partition_row_timestamp_and_copy(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 123.456)

    item = {"name": "Alice"}
    db.insert("people", "1", item)

    stored = db.get("people", "1")
    assert stored["partition_key"] == "people"
    assert stored["row_key"] == "1"
    assert stored["timestamp"] == 123.456
    assert stored["name"] == "Alice"

    # Ensure the stored row is a copy, not the original dict object.
    item["name"] = "Changed"
    assert db.get("people", "1")["name"] == "Alice"


def test_insert_populates_index_data_when_text_index_is_defined(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)
    db.textIndex("docs", {"title": 2, "body": 1})

    db.insert("docs", "a", {"title": "foo foo bar", "body": "bar baz"})

    stored = db.get("docs", "a")
    assert stored["index_data"] == [("foo", 4), ("bar", 3), ("baz", 1)]


def test_insert_does_not_add_index_data_without_text_index(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)

    db.insert("docs", "a", {"title": "foo bar"})
    stored = db.get("docs", "a")

    assert "index_data" not in stored


def test_get_missing_row_raises_404():
    db = MemoryDb()
    db.insert("people", "1", {"name": "Alice"})

    with pytest.raises(ValueError) as exc:
        db.get("people", "missing")

    assert exc.value.args[0] == 404


def test_get_list_returns_rows_in_requested_order():
    db = MemoryDb()
    db.insert("people", "a", {"name": "Alice"})
    db.insert("people", "b", {"name": "Bob"})
    db.insert("people", "c", {"name": "Carol"})

    rows = db.get("people", ["c", "a"])

    assert [r["row_key"] for r in rows] == ["c", "a"]


def test_get_all_returns_rows_sorted_by_timestamp(monkeypatch):
    db = MemoryDb()
    times = iter([30.0, 10.0, 20.0])
    monkeypatch.setattr("memorydb.time", lambda: next(times))

    db.insert("people", "a", {"name": "A"})
    db.insert("people", "b", {"name": "B"})
    db.insert("people", "c", {"name": "C"})

    rows = db.getAll("people")
    assert [r["row_key"] for r in rows] == ["b", "c", "a"]


def test_sample_returns_empty_list_for_empty_partition():
    db = MemoryDb()
    assert db.sample("missing") == []


def test_sample_can_be_deterministic_with_monkeypatched_randint(monkeypatch):
    db = MemoryDb()
    db.insert("people", "a", {"name": "A"})
    db.insert("people", "b", {"name": "B"})
    db.insert("people", "c", {"name": "C"})

    monkeypatch.setattr("memorydb.randint", lambda start, stop: stop)
    sample = db.sample("people", 3)

    assert [r["row_key"] for r in sample] == ["c", "c", "c"]


def test_query_requires_callable_expression():
    db = MemoryDb()

    with pytest.raises(ValueError) as exc:
        db.query("people", "not-callable")

    assert "required" in str(exc.value)


def test_query_returns_matching_rows_sorted_by_ts_desc():
    db = MemoryDb()
    db.insert("people", "a", {"name": "Alice", "ts": 1})
    db.insert("people", "b", {"name": "Bob", "ts": 3})
    db.insert("people", "c", {"name": "Carol", "ts": 2})

    rows = db.query("people", lambda row: True)
    assert [r["row_key"] for r in rows] == ["b", "c", "a"]


def test_change_id_updates_row_key_and_replaces_existing_row(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)
    db.insert("people", "a", {"name": "Alice"})

    item = db.get("people", "a")
    db.changeId(item, "z")

    with pytest.raises(ValueError):
        db.get("people", "a")
    assert db.get("people", "z")["name"] == "Alice"


def test_update_replaces_existing_row(monkeypatch):
    db = MemoryDb()
    monkeypatch.setattr("memorydb.time", lambda: 1.0)
    db.insert("people", "a", {"name": "Alice"})

    updated = db.get("people", "a")
    updated["name"] = "Alicia"
    db.update(updated)

    assert db.get("people", "a")["name"] == "Alicia"


def test_text_analysis_returns_full_text_analysis_with_expected_terms():
    db = MemoryDb()
    db.textIndex("docs", {"title": 2, "body": 1})
    db.insert("docs", "a", {"title": "foo foo bar", "body": "bar baz"})
    db.insert("docs", "b", {"title": "foo", "body": "baz baz"})

    analysis = db.textAnalysis("docs")

    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"foo", "bar", "baz"}


def test_full_text_analysis_scores_top_all_and_dump():
    data = {
        "foo": {
            "sum": 7,
            "spread": {
                4: {"value": "a", "next": {"value": "b"}},
                3: {"value": "c"},
            },
        }
    }
    analysis = FullTextAnalysis(data)

    assert analysis.dump() == data
    assert analysis.scores("foo") == [4, 3]
    assert analysis.top("foo") == "a"
    assert analysis.all("foo") == ["a", "b", "c"]


def test_full_text_analysis_filter_and_serialize():
    data = {
        "foo": {"sum": 7, "spread": {4: {"value": "a"}, 3: {"value": "b"}}},
        "bar": {"sum": 2, "spread": {1: {"value": "c"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term.startswith("f"))
    assert list(filtered.keys()) == ["foo"]

    ids = analysis.serialize(filtered)
    assert ids == ["a", "b"]


def test_full_text_analysis_highlight_replaces_whole_words_only():
    analysis = FullTextAnalysis({})
    result = analysis.highlight("foo", "foo bar foobar foo.", "<b>{{_}}</b>")

    assert result == "<b>foo</b> bar foobar <b>foo</b>."
