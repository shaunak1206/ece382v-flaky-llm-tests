import pytest
import time
from pymemorydb import memorydb, text_analysis


@pytest.fixture
def db():
    db = memorydb.MemoryDb()
    yield db
    db.resetDatabase()


def test_insert_adds_item_with_keys_and_timestamp(db):
    db.textIndex("test", {"content": {}})
    item = {"content": "hello world"}
    db.insert("test", "row1", item)

    stored = db.get("test", "row1")
    assert stored["partition_key"] == "test"
    assert stored["row_key"] == "row1"
    assert "timestamp" in stored
    assert isinstance(stored["timestamp"], float)
    assert stored["content"] == "hello world"
    assert "index_data" in stored
    # index_data is list of tuples (token, score)
    assert isinstance(stored["index_data"], list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in stored["index_data"])


def test_textIndex_creates_and_updates_indices(db):
    assert db.all_indices == {}

    db.textIndex("partA", {"field1": {}})
    assert "partA" in db.all_indices
    assert db.all_indices["partA"]["field1"] == {}

    db.textIndex("partA", {"field2": {"weight": 5}})
    assert "field2" in db.all_indices["partA"]
    assert db.all_indices["partA"]["field2"]["weight"] == 5


def test_indexItem_counts_words_with_default_weight(db):
    db.textIndex("p", {"text": {}})
    item = {"partition_key": "p", "row_key": "r1", "text": "foo foo bar"}
    idx = db.indexItem(item)
    # "foo" appears twice weight 1, "bar" once
    assert idx == [("foo", 2), ("bar", 1)] or idx == [("bar", 1), ("foo", 2)]


def test_indexItem_weights_work(db):
    db.textIndex("p", {"text": {"weight": 3}})
    item = {"partition_key": "p", "row_key": "r1", "text": "foo foo bar"}
    idx = dict(db.indexItem(item))
    # Each occurrence counted * weight=3
    assert idx.get("foo") == 6
    assert idx.get("bar") == 3


def test_textAnalysis_returns_FullTextAnalysis_object_and_data(db):
    db.textIndex("test", {"content": {}})
    db.insert("test", "r1", {"content": "a a b"})
    db.insert("test", "r2", {"content": "a b b"})

    analysis = db.textAnalysis("test")
    assert isinstance(analysis, text_analysis.FullTextAnalysis)

    data = analysis.dump()
    assert isinstance(data, dict)
    assert any(isinstance(v.get("sum", None), int) for v in data.values())
    # terms match keys
    terms = analysis.terms()
    assert set(terms) == set(data.keys())
    # all() returns list of row_key values present in analysis
    results = analysis.all(terms[0])
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


def test_textAnalysis_returns_none_for_empty_partition_key(db):
    # No items inserted, textAnalysis returns None
    assert db.textAnalysis("nonexistent") is None


def test_textIndex_and_insert_with_multiple_fields(db):
    db.textIndex("p", {"field1": {}, "field2": {"weight": 2}})
    item = {"field1": "a b b", "field2": "x y y y", "other": "skip me"}
    db.insert("p", "r1", item)
    stored = db.get("p", "r1")
    index_data = dict(stored.get("index_data", []))
    # field1 tokens counted once
    assert index_data.get("a") == 1
    assert index_data.get("b") == 2
    # field2 tokens weighted *2
    assert index_data.get("x") == 2
    assert index_data.get("y") == 6


def test_get_single_and_multiple(db):
    db.insert("p", "r1", {"val": "v1"})
    db.insert("p", "r2", {"val": "v2"})

    single = db.get("p", "r1")
    assert single["row_key"] == "r1"

    multiple = db.get("p", ["r1", "r2"])
    assert isinstance(multiple, list)
    assert {i["row_key"] for i in multiple} == {"r1", "r2"}


def test_get_raises_on_missing_key(db):
    with pytest.raises(ValueError) as e:
        db.get("p", "missing")
    assert e.value.args[0] == 404


def test_get_raises_on_invalid_param_type(db):
    with pytest.raises(ValueError):
        db.get("p", 1234)


def test_changeId_updates_row_key_and_preserves_partition_key(db):
    db.insert("p", "oldkey", {"val": "v"})
    item = db.get("p", "oldkey")
    db.changeId(item, "newkey")

    with pytest.raises(ValueError):
        db.get("p", "oldkey")

    updated = db.get("p", "newkey")
    assert updated["row_key"] == "newkey"
    assert updated["partition_key"] == "p"
    assert updated["val"] == "v"


def test_changeId_raises_if_item_not_in_table(db):
    item = {"partition_key": "p", "row_key": "missing"}
    with pytest.raises(ValueError) as e:
        db.changeId(item, "newkey")
    assert e.value.args[0] == 404


def test_insert_without_indexing_when_no_indices_configured(db):
    item = {"text": "no index"}
    db.insert("no_index", "r1", item)
    stored = db.get("no_index", "r1")
    assert "index_data" not in stored


def test_textAnalysis_handles_chained_next_entries(db):
    db.textIndex("p", {"content": {}})
    # Two items with same token separate chains
    item1 = {"content": "1 1"}
    item2 = {"content": "1"}
    db.insert("p", "r1", item1)
    db.insert("p", "r2", item2)

    analysis = db.textAnalysis("p")
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    spread = analysis.data.get(1) or analysis.data.get("1") or {}
    if spread:
        # At least one token chain present
        for v in spread.get("spread", {}).values():
            # chains do not cause infinite loops
            count = 0
            next_ = v.get("next", None)
            while next_ is not None:
                count += 1
                assert count < 21
                next_ = next_.get("next", None)


def test_textIndex_allows_redefinition(db):
    db.textIndex("p", {"field": {}})
    db.textIndex("p", {"field": {"weight": 10}})
    assert db.all_indices["p"]["field"]["weight"] == 10


def test_indexItem_skips_unindexed_fields(db):
    db.textIndex("p", {"field1": {}})
    item = {"partition_key": "p", "row_key": "r1", "field1": "hello world", "field2": "should skip"}
    index = dict(db.indexItem(item))
    # "should skip" tokens should not appear
    assert "should" not in index
    assert "hello" in index


def test_indexItem_empty_string_results_in_empty_index(db):
    db.textIndex("p", {"field1": {}})
    item = {"partition_key": "p", "row_key": "r1", "field1": ""}
    index = db.indexItem(item)
    assert index == []


def test_textIndex_creates_empty_dict_when_calling_with_new_partition(db):
    db.textIndex("new_part", {})
    assert "new_part" in db.all_indices
    assert isinstance(db.all_indices["new_part"], dict)


# -- Tests for text_analysis.FullTextAnalysis --

def test_FullTextAnalysis_dump_and_terms():
    data = {'1': {"sum": 2, "spread": {"1": {"value": "a"}}}}
    fta = text_analysis.FullTextAnalysis(data)
    assert fta.dump() == data
    assert fta.terms() == ['1']


def test_FullTextAnalysis_scores_and_top():
    data = {'term': {"sum": 10, "spread": {3: {"value": "x"}, 5: {"value": "y"}}}}
    fta = text_analysis.FullTextAnalysis(data)
    scores = fta.scores('term')
    assert scores == [5, 3]
    assert fta.top('term') == "y"
    assert fta.scores('missing') is None
    assert fta.top('missing') is None


def test_FullTextAnalysis_all_returns_all_values():
    data = {'term': {"spread": {1: {"value": "a", "next": {"value": "b"}}}}}
    fta = text_analysis.FullTextAnalysis(data)
    all_vals = fta.all('term')
    assert all_vals == ['a', 'b']


def test_FullTextAnalysis_filter_returns_sorted_filtered():
    data = {
        'a': {'sum': 10, 'spread': {}},
        'b': {'sum': 5, 'spread': {}},
        'c': {'sum': 20, 'spread': {}},
    }
    fta = text_analysis.FullTextAnalysis(data)
    # filter that passes everything
    filtered = fta.filter(lambda x: True)
    assert set(filtered.keys()) == {'c', 'a', 'b'}
    # filter with no callable
    filtered2 = fta.filter("not callable")
    assert filtered2 == []


def test_FullTextAnalysis_serialize_and_highlight():
    data = {'term': {"sum": 10, "spread": {"1": {"value": "a"}}}}
    fta = text_analysis.FullTextAnalysis(data)
    filtered = fta.filter(lambda t: True)
    ids = fta.serialize(filtered)
    assert "a" in ids

    rendered = fta.highlight("word", "some word here", lambda t: f"<b>{t}</b>")
    assert "<b>word</b>" in rendered
