import pytest

try:
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis
except Exception:  # pragma: no cover
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis


def make_db():
    return MemoryDb()


def test_text_index_creates_partition_and_stores_description():
    db = make_db()
    db.textIndex("posts", {"title": {"weight": 2}, "body": {"weight": 1}})

    assert "posts" in db.all_indices
    assert db.all_indices["posts"]["title"]["weight"] == 2
    assert db.all_indices["posts"]["body"]["weight"] == 1


def test_text_index_merges_multiple_calls_for_same_partition():
    db = make_db()
    db.textIndex("posts", {"title": {"weight": 2}})
    db.textIndex("posts", {"body": {"weight": 1}})

    assert set(db.all_indices["posts"].keys()) == {"title", "body"}
    assert db.all_indices["posts"]["title"]["weight"] == 2
    assert db.all_indices["posts"]["body"]["weight"] == 1


def test_insert_adds_partition_row_timestamp_and_copies_item():
    db = make_db()
    item = {"title": "hello world"}
    db.insert("posts", "1", item)

    assert len(db.table) == 1
    stored = db.table[0]
    assert stored["partition_key"] == "posts"
    assert stored["row_key"] == "1"
    assert "timestamp" in stored
    assert stored["title"] == "hello world"
    assert "partition_key" not in item
    assert "row_key" not in item
    assert "timestamp" not in item


def test_insert_adds_index_data_when_text_index_is_defined():
    db = make_db()
    db.textIndex("posts", {"title": {"weight": 2}})
    db.insert("posts", "1", {"title": "a a b"})

    stored = db.get("posts", "1")
    assert "index_data" in stored
    assert stored["index_data"] == [("a", 4), ("b", 2)]


def test_insert_ignores_non_indexed_fields_in_index_data():
    db = make_db()
    db.textIndex("posts", {"title": {"weight": 1}})
    db.insert("posts", "1", {"title": "foo", "body": "bar baz"})

    stored = db.get("posts", "1")
    assert stored["index_data"] == [("foo", 1)]


def test_text_analysis_returns_none_when_no_index_data_exists():
    db = make_db()
    db.insert("posts", "1", {"title": "no index configured"})
    assert db.textAnalysis("posts") is None


def test_text_analysis_returns_full_text_analysis_for_indexed_items():
    db = make_db()
    db.textIndex("posts", {"title": {"weight": 1}})
    db.insert("posts", "1", {"title": "alpha beta"})
    db.insert("posts", "2", {"title": "alpha gamma"})
    db.insert("posts", "3", {"title": "beta"})

    analysis = db.textAnalysis("posts")
    assert isinstance(analysis, FullTextAnalysis)
    assert set(analysis.terms()) == {"alpha", "beta", "gamma"}


def test_full_text_analysis_dump_and_terms():
    data = {
        "alpha": {"sum": 2, "spread": {1: {"value": "1"}}},
        "beta": {"sum": 1, "spread": {1: {"value": "2"}}},
    }
    analysis = FullTextAnalysis(data)

    assert analysis.dump() == data
    assert analysis.terms() == ["alpha", "beta"]


def test_full_text_analysis_scores_and_top():
    data = {
        "alpha": {"sum": 3, "spread": {5: {"value": "r1"}, 2: {"value": "r2"}}},
    }
    analysis = FullTextAnalysis(data)

    assert analysis.scores("alpha") == [5, 2]
    assert analysis.top("alpha") == "r1"


def test_full_text_analysis_scores_and_top_return_none_for_missing_term():
    analysis = FullTextAnalysis({})

    assert analysis.scores("missing") is None
    assert analysis.top("missing") is None


def test_full_text_analysis_all_returns_all_values_across_spread_and_chain():
    data = {
        "alpha": {
            "sum": 6,
            "spread": {
                3: {"value": "r1", "next": {"value": "r2"}},
                1: {"value": "r3"},
            },
        }
    }
    analysis = FullTextAnalysis(data)

    assert analysis.all("alpha") == ["r1", "r2", "r3"]


def test_full_text_analysis_filter_returns_matching_terms_sorted_by_sum_desc():
    data = {
        "alpha": {"sum": 2, "spread": {2: {"value": "a"}}},
        "beta": {"sum": 5, "spread": {5: {"value": "b"}}},
        "gamma": {"sum": 1, "spread": {1: {"value": "c"}}},
    }
    analysis = FullTextAnalysis(data)

    filtered = analysis.filter(lambda term: term != "gamma")
    assert list(filtered.keys()) == ["beta", "alpha"]
    assert filtered["beta"]["sum"] == 5
    assert filtered["alpha"]["sum"] == 2


def test_full_text_analysis_filter_returns_empty_list_for_non_callable():
    analysis = FullTextAnalysis({"alpha": {"sum": 1, "spread": {1: {"value": "x"}}}})

    assert analysis.filter("not callable") == []


def test_full_text_analysis_serialize_flattens_unique_ids_in_score_order():
    data = {
        "alpha": {
            "sum": 6,
            "spread": {
                3: {"value": "r1", "next": {"value": "r2"}},
                1: {"value": "r3"},
            },
        },
        "beta": {
            "sum": 2,
            "spread": {
                2: {"value": "r2"},
            },
        },
    }
    analysis = FullTextAnalysis(data)

    assert analysis.serialize(data) == ["r1", "r2", "r3"]


def test_full_text_analysis_highlight_replaces_whole_word_only():
    analysis = FullTextAnalysis({})
    text = "cat scatter catapult cat"
    rendered = analysis.highlight("cat", text, "<b>{{_}}</b>")

    assert rendered == "<b>cat</b> scatter catapult <b>cat</b>"
