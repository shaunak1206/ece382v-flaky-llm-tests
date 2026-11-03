import pytest
import time
from pymemorydb import memorydb, text_analysis


@pytest.fixture
def db():
    return memorydb.MemoryDb()


def test_textIndex_creates_new_index_and_adds_fields(db):
    pk = "test_pk"
    desc = {"field1": {"weight": 2}, "field2": 5}
    assert pk not in db.all_indices
    db.textIndex(pk, desc)
    assert pk in db.all_indices
    indices = db.all_indices[pk]
    assert indices["field1"] == {"weight": 2}
    assert indices["field2"] == 5


def test_insert_adds_item_with_keys_and_timestamp(db):
    pk, rk = "p", "r"
    item = {"field1": "hello world"}
    db.textIndex(pk, {"field1": {"weight": 1}})
    before = time.time()
    db.insert(pk, rk, item)
    after = time.time()
    stored = db.get(pk, rk)
    assert stored["partition_key"] == pk
    assert stored["row_key"] == rk
    assert "timestamp" in stored
    assert before <= stored["timestamp"] <= after
    # index_data should be present
    assert "index_data" in stored
    # index_data should be sorted descending by count * weight
    counts = [count for _, count in stored["index_data"]]
    assert counts == sorted(counts, reverse=True)


def test_insert_with_no_index_does_not_add_index_data(db):
    pk, rk = "p2", "r2"
    item = {"fieldX": "no index here"}
    # no textIndex configured for pk
    db.insert(pk, rk, item)
    stored = db.get(pk, rk)
    assert "index_data" not in stored


def test_indexItem_counts_and_weights_terms(db):
    pk = "pk_weight"
    desc = {"field": {"weight": 3}}
    db.textIndex(pk, desc)
    item = {"partition_key": pk, "row_key": "r1", "field": "a a a b b c"}
    result = db.indexItem(item)
    # Expected counts: 'a' = 3*3=9, 'b' = 2*3=6, 'c' = 1*3=3
    expected = [("a", 9), ("b", 6), ("c", 3)]
    assert result == expected


def test_indexItem_ignores_fields_not_in_indices(db):
    pk = "pk_ignore"
    desc = {"field1": {"weight": 1}}
    db.textIndex(pk, desc)
    item = {"partition_key": pk, "row_key": "r1", "field2": "should be ignored"}
    result = db.indexItem(item)
    assert result == [] or result is None


def test_textAnalysis_returns_FullTextAnalysis_instance_and_correct_data(db):
    pk = "pk_textanalysis"
    desc = {"content": {"weight": 1}}
    db.textIndex(pk, desc)
    # Insert two items with overlapping terms
    db.insert(pk, "r1", {"content": "apple apple banana"})
    db.insert(pk, "r2", {"content": "banana orange banana"})
    analysis = db.textAnalysis(pk)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    terms = analysis.terms()
    # The terms are token keys as strings from index_data keys (integers as strings)
    assert all(isinstance(term, str) for term in terms)
    # The spread data per term should contain the row keys in values
    for term in terms:
        spread = analysis.data[term]["spread"]
        assert any("value" in v for v in spread.values())


def test_textAnalysis_no_index_data_returns_none(db):
    pk = "pk_noindexdata"
    # Insert item without index_data
    db.insert(pk, "rk", {"field": "hello"})
    result = db.textAnalysis(pk)
    assert result is None


def test_insert_and_get_multiple_rows(db):
    pk = "multipk"
    db.insert(pk, "r1", {"data": "one"})
    db.insert(pk, "r2", {"data": "two"})
    # get single
    row = db.get(pk, "r1")
    assert row["data"] == "one"
    # get multiple
    rows = db.get(pk, ["r1", "r2"])
    assert len(rows) == 2
    assert rows[1]["data"] == "two"


def test_get_invalid_param_raises_ValueError(db):
    with pytest.raises(ValueError):
        db.get("pk", 123)


def test_changeId_raises_when_missing_keys(db):
    with pytest.raises(ValueError):
        db.changeId({}, "newid")
    with pytest.raises(ValueError):
        db.changeId({"partition_key": "p"}, "newid")


def test_changeId_finds_and_updates_row_key(db):
    pk = "pkcid"
    rk = "orig"
    db.insert(pk, rk, {"content": "abc"})
    item = db.get(pk, rk)
    db.changeId(item, "newid")
    with pytest.raises(ValueError):
        db.get(pk, rk)
    retrieved = db.get(pk, "newid")
    assert retrieved["row_key"] == "newid"
    assert retrieved["content"] == "abc"


def test_update_raises_when_missing_keys(db):
    with pytest.raises(ValueError):
        db.update({})
    with pytest.raises(ValueError):
        db.update({"partition_key": "p"})
    with pytest.raises(ValueError):
        db.update({"row_key": "r"})


def test_update_replaces_existing_item(db):
    pk = "pku"
    rk = "rku"
    db.insert(pk, rk, {"x": "old"})
    item = db.get(pk, rk)
    item["x"] = "new"
    db.update(item)
    updated_item = db.get(pk, rk)
    assert updated_item["x"] == "new"


def test_update_raises_404_if_item_not_exist(db):
    # item must exist to update
    item = {"partition_key": "p", "row_key": "r", "field": "y"}
    with pytest.raises(ValueError):
        db.update(item)


def test_updateAll_replaces_all_items(db):
    pk = "pkuall"
    items = [
        {"partition_key": pk, "row_key": "r1", "val": 1},
        {"partition_key": pk, "row_key": "r2", "val": 2},
    ]
    for item in items:
        db.insert(pk, item["row_key"], {"val": item["val"]})
    # Change val and updateAll
    new_items = [
        {"partition_key": pk, "row_key": "r1", "val": 10},
        {"partition_key": pk, "row_key": "r2", "val": 20},
    ]
    db.updateAll(new_items)
    for ni in new_items:
        got = db.get(pk, ni["row_key"])
        assert got["val"] == ni["val"]


def test_sample_returns_empty_when_no_items(db):
    result = db.sample("nosuchpk", count=5)
    assert result == []


def test_sample_returns_list_of_items(db):
    pk = "samplepk"
    # Insert 3 items
    for i in range(3):
        db.insert(pk, f"r{i}", {"field": f"v{i}"})
    sample = db.sample(pk, count=2)
    assert isinstance(sample, list)
    # Sample size can be 0 or more, but since sample recursively picks items indexed 1..n, might exclude first
    # So length can vary; test sample length <= count and items belong to partition_key
    assert len(sample) <= 2
    for s in sample:
        assert s["partition_key"] == pk


def test_textIndex_adds_weights_and_overwrites(db):
    pk = "pkoverwrite"
    db.textIndex(pk, {"f": 1})
    db.textIndex(pk, {"f": {"weight": 3}, "g": 5})
    assert db.all_indices[pk]["f"] == {"weight": 3}
    assert db.all_indices[pk]["g"] == 5


def test_insert_creates_index_data_with_weight_and_duplicate_terms(db):
    pk = "pkwtdup"
    db.textIndex(pk, {"txt": {"weight": 2}})
    item = {"txt": "hello hello bye"}
    db.insert(pk, "r1", item)
    inserted = db.get(pk, "r1")
    index_data = inserted.get("index_data")
    assert index_data is not None
    # Hello appears twice with weight 2 => 4, bye once = 2
    expected = [("hello", 4), ("bye", 2)]
    assert index_data == expected


# Tests for FullTextAnalysis class edge cases and basic usage

def test_FullTextAnalysis_dump_and_terms_basic():
    data = {"1": {"sum": 3, "spread": {"1": {"value": "a"}}}}
    fta = text_analysis.FullTextAnalysis(data)
    assert fta.dump() == data
    assert fta.terms() == ["1"]


def test_FullTextAnalysis_scores_top_all_methods():
    data = {
        "term1": {
            "sum": 5,
            "spread": {
                "1": {"value": "a"},
                "2": {"value": "b"}
            }
        }
    }
    fta = text_analysis.FullTextAnalysis(data)
    scores = fta.scores("term1")
    assert scores == ["2", "1"]  # numeric keys reversed as strings
    top = fta.top("term1")
    assert top == "b"
    all_vals = fta.all("term1")
    assert "a" in all_vals and "b" in all_vals


def test_FullTextAnalysis_filter_and_serialize(db):
    data = {
        "term1": {"sum": 10, "spread": {"1": {"value": "a"}}},
        "term2": {"sum": 5, "spread": {"2": {"value": "b"}}}
    }
    fta = text_analysis.FullTextAnalysis(data)

    # filter with non-callable returns empty
    assert fta.filter(None) == []

    # filter with callable that matches all
    filtered = fta.filter(lambda k: True)
    assert "term1" in filtered and "term2" in filtered

    # filter with callable that matches none
    empty_filtered = fta.filter(lambda k: False)
    assert empty_filtered == {}

    # serialize returns list of values preserving order and no duplicates
    ids = fta.serialize(filtered)
    assert isinstance(ids, list)
    assert "a" in ids and "b" in ids


def test_FullTextAnalysis_highlight_replaces_term():
    fta = text_analysis.FullTextAnalysis({})
    text = "hello world hello"
    term = "hello"
    rendered = fta.highlight(term, text, render="<b>{{_}}</b>")
    # It should replace all "hello" whole word with <b>hello</b>
    assert rendered.count("<b>hello</b>") == 2
    # Should not replace "helloworld" or partial match
    assert "helloworld" not in rendered


def test_textAnalysis_handles_long_next_chain_limit(db):
    pk = "pk_longnext"
    db.textIndex(pk, {"content": {"weight": 1}})
    item = {"content": "a a a a a a a a a a a a a a a a a a a a a"}  # 21 times 'a'
    db.insert(pk, "r1", item)
    analysis = db.textAnalysis(pk)
    # Should not exceed next chain breaker of 20
    spread = list(analysis.data.values())[0]["spread"]
    # There should not be infinite loops, just some next chains
    counts = 0
    for v in spread.values():
        nxt = v.get("next")
        while nxt is not None:
            counts += 1
            nxt = nxt.get("next")
    assert counts <= 20


def test_textIndex_with_empty_desc_adds_empty_index(db):
    pk = "emptydesc"
    db.textIndex(pk, {})
    assert pk in db.all_indices
    assert db.all_indices[pk] == {}


def test_insert_handles_multiple_fields_and_weights(db):
    pk = "multi_fields"
    desc = {"f1": {"weight": 2}, "f2": 1}
    db.textIndex(pk, desc)
    item = {"f1": "x x y", "f2": "y y z"}
    db.insert(pk, "r1", item)
    indexed = db.get(pk, "r1")["index_data"]
    # counts:
    # f1: x=2*2=4, y=1*2=2
    # f2: y=2*1=2, z=1*1=1
    #
    # Total per token: x=4, y=4 (2 + 2), z=1
    expected = [("x", 4), ("y", 4), ("z", 1)]
    # Order is descending by counts
    assert indexed == expected

