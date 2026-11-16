import pytest

try:
    import memorydb as mdb
    import text_analysis as ta
except ImportError:  # pragma: no cover
    from pymemorydb import memorydb as mdb
    from pymemorydb import text_analysis as ta

MemoryDb = mdb.MemoryDb
FullTextAnalysis = ta.FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


@pytest.fixture
def analysis(db):
    db.textIndex("docs", {"body": 1})
    db.insert("docs", "doc1", {"body": "apple apple banana"})
    db.insert("docs", "doc2", {"body": "apple"})
    return db.textAnalysis("docs")


# ---------------------------------------------------------------- textIndex

def test_text_index_creates_partition_entry(db):
    db.textIndex("docs", {"body": 1})
    assert "docs" in db.all_indices
    assert db.dumpindices("docs") == {"body": 1}


def test_text_index_merges_multiple_calls(db):
    db.textIndex("docs", {"body": 1})
    db.textIndex("docs", {"title": {"weight": 3}})
    assert db.all_indices["docs"] == {"body": 1, "title": {"weight": 3}}


def test_text_index_overwrites_existing_field(db):
    db.textIndex("docs", {"body": 1})
    db.textIndex("docs", {"body": {"weight": 5}})
    assert db.all_indices["docs"]["body"] == {"weight": 5}


def test_text_index_partitions_are_independent(db):
    db.textIndex("a", {"x": 1})
    db.textIndex("b", {"y": 1})
    assert set(db.all_indices.keys()) == {"a", "b"}
    assert "y" not in db.all_indices["a"]
    assert "x" not in db.all_indices["b"]


# ------------------------------------------------------------------- insert

def test_insert_sets_metadata_fields(db):
    item = {"name": "n"}
    db.insert("p", "r1", item)
    row = db.get("p", "r1")
    assert row["partition_key"] == "p"
    assert row["row_key"] == "r1"
    assert isinstance(row["timestamp"], float)
    assert row["name"] == "n"


def test_insert_without_index_has_no_index_data(db):
    db.insert("p", "r1", {"body": "hello world"})
    assert "index_data" not in db.get("p", "r1")


def test_insert_with_index_counts_consecutive_terms(db):
    db.textIndex("p", {"body": 1})
    db.insert("p", "r1", {"body": "apple apple banana"})
    assert db.get("p", "r1")["index_data"] == [("apple", 2), ("banana", 1)]


def test_insert_applies_weight(db):
    db.textIndex("p", {"body": {"weight": 3}})
    db.insert("p", "r1", {"body": "hello world"})
    assert db.get("p", "r1")["index_data"] == [("hello", 3), ("world", 3)]


def test_insert_combines_fields_and_ignores_unindexed(db):
    db.textIndex("p", {"title": {"weight": 3}, "body": 1})
    db.insert("p", "r1", {"title": "apple", "body": "apple pie apple",
                          "other": "ignored words"})
    data = db.get("p", "r1")["index_data"]
    assert data == [("apple", 5), ("pie", 1)]
    assert "ignored" not in dict(data)


def test_insert_only_indexes_partitions_with_index(db):
    db.textIndex("indexed", {"body": 1})
    db.insert("indexed", "r1", {"body": "foo"})
    db.insert("plain", "r1", {"body": "foo"})
    assert "index_data" in db.get("indexed", "r1")
    assert "index_data" not in db.get("plain", "r1")


def test_insert_stores_copy_of_item(db):
    item = {"name": "n"}
    db.insert("p", "r1", item)
    item["name"] = "changed"
    assert db.get("p", "r1")["name"] == "n"


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_without_indexed_items(db):
    db.insert("p", "r1", {"body": "hello"})
    assert db.textAnalysis("p") is None


def test_text_analysis_sum_and_spread(analysis):
    assert isinstance(analysis, mdb.text_analysis.FullTextAnalysis)
    data = analysis.dump()
    assert set(data.keys()) == {"apple", "banana"}
    assert data["apple"]["sum"] == 3
    assert data["apple"]["spread"][2]["value"] == "doc1"
    assert data["apple"]["spread"][1]["value"] == "doc2"
    assert data["banana"]["sum"] == 1
    assert data["banana"]["spread"] == {1: {"value": "doc1"}}


def test_text_analysis_chains_items_with_same_score(db):
    db.textIndex("p", {"body": 1})
    db.insert("p", "doc1", {"body": "apple"})
    db.insert("p", "doc2", {"body": "apple"})
    data = db.textAnalysis("p").dump()
    assert data["apple"]["sum"] == 2
    node = data["apple"]["spread"][1]
    assert node["value"] == "doc1"
    assert node["next"]["value"] == "doc2"


def test_text_analysis_ignores_other_partitions(db):
    db.textIndex("a", {"body": 1})
    db.textIndex("b", {"body": 1})
    db.insert("a", "r1", {"body": "alpha"})
    db.insert("b", "r2", {"body": "beta"})
    assert db.textAnalysis("a").terms() == ["alpha"]
    assert db.textAnalysis("b").terms() == ["beta"]


# --------------------------------------------------------- FullTextAnalysis

def test_full_text_analysis_terms_and_dump():
    data = {"x": {"sum": 1, "spread": {1: {"value": "r"}}}}
    fta = FullTextAnalysis(data)
    assert fta.dump() is data
    assert fta.terms() == ["x"]


def test_scores_and_top(analysis):
    assert analysis.scores("apple") == [2, 1]
    assert analysis.top("apple") == "doc1"
    assert analysis.top("banana") == "doc1"


def test_scores_and_top_for_missing_term(analysis):
    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_all_returns_every_row_key(db):
    db.textIndex("p", {"body": 1})
    db.insert("p", "doc1", {"body": "apple"})
    db.insert("p", "doc2", {"body": "apple"})
    db.insert("p", "doc3", {"body": "apple apple"})
    fta = db.textAnalysis("p")
    assert sorted(fta.all("apple")) == ["doc1", "doc2", "doc3"]


def test_all_missing_term_raises_key_error(analysis):
    with pytest.raises(KeyError):
        analysis.all("missing")


def test_filter_selects_matching_terms(analysis):
    result = analysis.filter(lambda t: t.startswith("a"))
    assert list(result.keys()) == ["apple"]
    assert result["apple"] == analysis.dump()["apple"]


def test_filter_non_callable_returns_empty_list(analysis):
    assert analysis.filter("apple") == []


def test_serialize_returns_unique_ids(analysis):
    ids = analysis.serialize(analysis.filter(lambda t: True))
    assert sorted(ids) == ["doc1", "doc2"]
    assert len(ids) == len(set(ids))


def test_serialize_none_returns_none(analysis):
    assert analysis.serialize(None) is None


def test_highlight_wraps_whole_word_only(analysis):
    out = analysis.highlight("apple", "I like apple and pineapple",
                             "<b>{{_}}</b>")
    assert out == "I like <b>apple</b> and pineapple"
