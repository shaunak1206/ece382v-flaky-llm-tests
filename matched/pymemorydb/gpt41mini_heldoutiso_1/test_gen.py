import pytest
import time
from pymemorydb import memorydb, text_analysis

@pytest.fixture
def db():
    db = memorydb.MemoryDb()
    yield db
    db.resetDatabase()

def test_textIndex_creates_new_index(db):
    partition_key = "pk1"
    desc = {"field1": {}, "field2": {"weight": 2}}
    assert partition_key not in db.all_indices
    db.textIndex(partition_key, desc)
    assert partition_key in db.all_indices
    assert "field1" in db.all_indices[partition_key]
    assert "field2" in db.all_indices[partition_key]
    assert db.all_indices[partition_key]["field2"]["weight"] == 2

def test_textIndex_overwrites_existing_index(db):
    pk = "pk"
    db.all_indices[pk] = {"a": {"weight": 1}}
    desc = {"b": {}, "c": {"weight": 3}}
    db.textIndex(pk, desc)
    assert "a" in db.all_indices[pk]
    assert "b" in db.all_indices[pk]
    assert "c" in db.all_indices[pk]
    assert db.all_indices[pk]["c"]["weight"] == 3

def test_insert_adds_item_and_indexes(db):
    pk = "users"
    row_key = "u1"
    db.textIndex(pk, {"name": {"weight": 2}})
    item = {"name": "alpha beta beta"}
    db.insert(pk, row_key, item)
    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == pk
    assert stored["row_key"] == row_key
    assert "timestamp" in stored
    assert "index_data" in stored
    # 'beta' occurs twice, with weight 2 -> 4
    beta_entry = next((t for t,v in stored["index_data"] if t=="beta"), None)
    assert beta_entry == "beta"
    # highest count should be 'beta' (4)
    top_term = stored["index_data"][0][0]
    assert top_term == "beta"

def test_insert_without_index(db):
    pk = "nopk"
    row_key = "r1"
    item = {"desc": "some text"}
    # no index defined for pk so indexItem returns None
    db.insert(pk, row_key, item)
    assert "index_data" not in db.table[0]

def test_indexItem_calculates_weighted_counts(db):
    pk = "p"
    db.all_indices[pk] = {"text": {"weight": 3}}
    item = {"partition_key": pk, "row_key": "r", "text": "a a b"}
    result = db.indexItem(item)
    # 'a' occurs twice, so count: (2 * 3) = 6, 'b' once 3
    expected = [("a", 6), ("b", 3)]
    assert result == expected

def test_textAnalysis_returns_FullTextAnalysis_object(db):
    pk = "pkA"
    db.textIndex(pk, {"content": {}})
    # Insert multiple items with index data
    items = [
        {"content": "hello world"},
        {"content": "hello hello world"},
        {"content": "world world world"},
    ]
    for i, item in enumerate(items):
        db.insert(pk, f"id{i}", item)
    analysis = db.textAnalysis(pk)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    terms = analysis.terms()
    # terms like 'hello' and 'world' should be keys
    assert any(t in ["hello", "world"] for t in terms)

def test_textAnalysis_returns_none_if_no_index_data(db):
    pk = "pkX"
    item = {"field": "val"}
    db.insert(pk, "r1", item)
    result = db.textAnalysis(pk)
    assert result is None

def test_textAnalysis_multiple_next_links(db):
    pk = "pkN"
    db.textIndex(pk, {"text": {}})
    # Insert multiple items with overlapping index data 'test'
    items = [{"text": "test test"} for i in range(5)]
    for i, item in enumerate(items):
        db.insert(pk, f"id{i}", item)
    analysis = db.textAnalysis(pk)
    spread = analysis.data
    # 'test' should have multiple linked nodes
    found = False
    for key in spread:
        for score in spread[key]['spread']:
            value = spread[key]['spread'][score]
            next_count = 0
            while value is not None:
                next_count += 1
                value = value.get('next', None)
            if next_count > 1:
                found = True
    assert found

def test_insert_and_get(db):
    pk = "pkGet"
    row_key = "rKey"
    item = {"field": "value"}
    db.insert(pk, row_key, item)
    got = db.get(pk, row_key)
    assert got["field"] == "value"
    # Get raises if no such id
    with pytest.raises(ValueError) as e:
        db.get(pk, "missing")
    assert e.value.args[0] == 404

def test_insert_and_get_list(db):
    pk = "pkL"
    db.insert(pk, "r1", {"field": "v1"})
    db.insert(pk, "r2", {"field": "v2"})
    results = db.get(pk, ["r1", "r2"])
    assert len(results) == 2
    assert any(r["field"] == "v1" for r in results)
    assert any(r["field"] == "v2" for r in results)

def test_get_invalid_param_type_raises(db):
    with pytest.raises(ValueError):
        db.get("pk", 123)

def test_textIndex_and_insert_combined(db):
    pk = "combined"
    db.textIndex(pk, {"desc": {"weight": 5}})
    db.insert(pk, "id1", {"desc": "red red blue"})
    stored = db.get(pk, "id1")
    idx_data = stored.get("index_data")
    assert idx_data is not None
    assert idx_data[0][0] == "red"
    assert idx_data[0][1] == 10  # 2 occurrences * weight 5

def test_insert_and_textAnalysis_with_no_index_data_skips_item(db):
    pk = "noindex"
    # do not call textIndex -> no indexing applied
    db.insert(pk, "r1", {"txt": "hello world"})
    result = db.textAnalysis(pk)
    assert result is None

# Tests for FullTextAnalysis class methods

def test_FullTextAnalysis_dump_and_terms():
    data = {
        "10": {
            "sum": 10,
            "spread": {
                "10": {"value": "a"},
                "5": {"value": "b"},
            }
        },
        "5": {
            "sum": 5,
            "spread": {
                "5": {"value": "c"}
            }
        }
    }
    fta = text_analysis.FullTextAnalysis(data)
    assert fta.dump() == data
    terms = fta.terms()
    assert set(terms) == {"10", "5"}

def test_FullTextAnalysis_scores_and_top():
    data = {
        "term": {
            "sum": 3,
            "spread": {
                "3": {"value": "v1"},
                "1": {"value": "v2"}
            }
        }
    }
    fta = text_analysis.FullTextAnalysis(data)
    scores = fta.scores("term")
    assert scores == ["3", "1"]
    top = fta.top("term")
    assert top == "v1"
    # scores returns None for unknown term
    assert fta.scores("missing") is None
    assert fta.top("missing") is None

def test_FullTextAnalysis_all_collects_all_values_in_links():
    data = {
        "term": {
            "sum": 2,
            "spread": {
                "1": {"value": "v1", "next": {"value": "v2", "next": None}}
            }
        }
    }
    fta = text_analysis.FullTextAnalysis(data)
    results = fta.all("term")
    assert results == ["v1", "v2"]

def test_FullTextAnalysis_filter_and_serialize():
    data = {
        "term1": {"sum": 5, "spread": {"5": {"value": "a"}}},
        "term2": {"sum": 3, "spread": {"3": {"value": "b"}}},
        "term3": {"sum": 1, "spread": {"1": {"value": "c"}}}
    }
    fta = text_analysis.FullTextAnalysis(data)
    # Filter returns empty if expression is not callable
    assert fta.filter(None) == []
    # Filter using expression that keeps keys with sum >= 3
    filtered = fta.filter(lambda k: fta.data[k]['sum'] >= 3)
    assert "term1" in filtered and "term2" in filtered and "term3" not in filtered
    serialized = fta.serialize(filtered)
    assert "a" in serialized and "b" in serialized
    # serialize with None returns None
    assert fta.serialize(None) is None

def test_FullTextAnalysis_highlight_replaces_term():
    fta = text_analysis.FullTextAnalysis({})
    text = "This is a test string to test highlighting."
    term = "test"
    render = "*{{_}}*"
    result = fta.highlight(term, text, render)
    assert "*test*" in result
    # The term should be highlighted where exact word boundaries match
    assert result.count("*test*") == 2

def test_textAnalysis_handles_large_next_chain_safely(db):
    pk = "chain"
    db.textIndex(pk, {"txt": {}})
    # Insert items with many repeats for same token
    text = "word word"
    for i in range(25):
        db.insert(pk, f"id{i}", {"txt": text})
    analysis = db.textAnalysis(pk)
    # Should return FullTextAnalysis despite breaker in chain
    assert isinstance(analysis, text_analysis.FullTextAnalysis)

def test_indexItem_ignores_fields_not_in_indices(db):
    pk = "idx_ignore"
    db.all_indices[pk] = {"field1": {}}
    item = {
        "partition_key": pk,
        "row_key": "r",
        "field1": "a b b",
        "field2": "c c c"
    }
    result = db.indexItem(item)
    # Only field1 used
    terms = [t for t,_ in result]
    assert "a" in terms and "b" in terms
    assert "c" not in terms

def test_insert_copies_item(db):
    pk = "copytest"
    db.textIndex(pk, {"text": {}})
    item = {"text": "a a b"}
    db.insert(pk, "rk", item)
    # Change original item after insertion
    item["text"] = "changed"
    stored = db.get(pk, "rk")
    # Stored item text should not be changed
    assert stored["text"] == "a a b"

def test_textIndex_creates_empty_index_if_none_given(db):
    pk = "emptydesc"
    db.textIndex(pk, {})
    assert pk in db.all_indices
    assert db.all_indices[pk] == {}
