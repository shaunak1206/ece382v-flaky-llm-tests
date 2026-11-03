from pymemorydb.memorydb import MemoryDb
from pymemorydb.text_analysis import FullTextAnalysis

PK = "docs"


def _index_of(db, row_key, partition=PK):
    return db.get(partition, row_key)["index_data"]


def test_textindex_creates_partition_index():
    db = MemoryDb()
    result = db.textIndex(PK, {"body": True})
    assert result is None
    assert db.all_indices == {PK: {"body": True}}


def test_textindex_merges_fields_across_calls():
    db = MemoryDb()
    db.textIndex(PK, {"title": True})
    db.textIndex(PK, {"body": {"weight": 3}})
    assert db.all_indices[PK] == {"title": True, "body": {"weight": 3}}


def test_textindex_replaces_existing_field_description():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.textIndex(PK, {"body": {"weight": 2}})
    assert db.all_indices[PK] == {"body": {"weight": 2}}
    db.insert(PK, "r1", {"body": "hi"})
    assert _index_of(db, "r1") == [("hi", 2)]


def test_textindex_partitions_are_independent():
    db = MemoryDb()
    db.textIndex("a", {"body": True})
    db.textIndex("b", {"title": True})
    assert db.all_indices == {"a": {"body": True}, "b": {"title": True}}


def test_insert_without_index_stores_no_index_data():
    db = MemoryDb()
    db.insert(PK, "r1", {"body": "hello world"})
    row = db.get(PK, "r1")
    assert row["row_key"] == "r1"
    assert row["partition_key"] == PK
    assert "timestamp" in row
    assert "index_data" not in row


def test_insert_copies_item_and_sets_key_fields():
    db = MemoryDb()
    item = {"body": "x"}
    db.insert(PK, "r1", item)
    assert item["partition_key"] == PK
    assert item["row_key"] == "r1"
    item["body"] = "changed"
    assert db.get(PK, "r1")["body"] == "x"


def test_insert_counts_consecutive_repeats_and_accumulates_runs():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "go go go stop"})
    db.insert(PK, "r2", {"body": "a a b a"})
    assert _index_of(db, "r1") == [("go", 3), ("stop", 1)]
    assert _index_of(db, "r2") == [("a", 3), ("b", 1)]


def test_insert_weight_multiplies_counts():
    db = MemoryDb()
    db.textIndex(PK, {"title": {"weight": 3}, "body": True})
    db.insert(PK, "r1", {"title": "cat", "body": "cat dog"})
    assert _index_of(db, "r1") == [("cat", 4), ("dog", 1)]


def test_insert_ignores_unindexed_fields():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "hello", "secret": "zzz"})
    assert _index_of(db, "r1") == [("hello", 1)]


def test_insert_with_empty_index_stores_empty_index_data():
    db = MemoryDb()
    db.textIndex(PK, {})
    db.insert(PK, "r1", {"body": "hello"})
    assert _index_of(db, "r1") == []


def test_insert_only_indexes_partitions_registered_with_textindex():
    db = MemoryDb()
    db.textIndex("a", {"body": True})
    db.insert("a", "r1", {"body": "x"})
    db.insert("b", "r1", {"body": "x"})
    assert "index_data" in db.get("a", "r1")
    assert "index_data" not in db.get("b", "r1")


def test_textanalysis_returns_none_when_nothing_is_indexed():
    db = MemoryDb()
    db.insert(PK, "r1", {"body": "apple"})
    assert db.textAnalysis(PK) is None
    assert db.textAnalysis("missing") is None


def test_textanalysis_terms_cover_all_indexed_tokens():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple banana"})
    db.insert(PK, "r2", {"body": "banana cherry"})
    analysis = db.textAnalysis(PK)
    assert analysis is not None
    assert sorted(analysis.terms()) == ["apple", "banana", "cherry"]


def test_textanalysis_sum_is_total_weight_across_rows():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple apple banana"})
    db.insert(PK, "r2", {"body": "apple"})
    dump = db.textAnalysis(PK).dump()
    assert dump["apple"]["sum"] == 3
    assert dump["banana"]["sum"] == 1


def test_textanalysis_chain_collects_every_row_with_same_count():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    for key in ["r1", "r2", "r3"]:
        db.insert(PK, key, {"body": "apple"})
    analysis = db.textAnalysis(PK)
    assert analysis.dump()["apple"]["sum"] == 3
    assert analysis.scores("apple") == [1]
    assert sorted(analysis.all("apple")) == ["r1", "r2", "r3"]


def test_textanalysis_scores_are_descending_counts():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple apple banana"})
    db.insert(PK, "r2", {"body": "apple"})
    analysis = db.textAnalysis(PK)
    assert analysis.scores("apple") == [2, 1]
    assert analysis.scores("banana") == [1]
    assert analysis.scores("missing") is None


def test_textanalysis_applies_field_weights():
    db = MemoryDb()
    db.textIndex(PK, {"title": {"weight": 2}, "body": True})
    db.insert(PK, "r1", {"title": "apple", "body": "apple pie"})
    dump = db.textAnalysis(PK).dump()
    assert dump["apple"]["sum"] == 3
    assert dump["pie"]["sum"] == 1


def test_textanalysis_keeps_partitions_isolated():
    db = MemoryDb()
    db.textIndex("a", {"body": True})
    db.textIndex("b", {"body": True})
    db.insert("a", "r1", {"body": "apple"})
    db.insert("b", "r2", {"body": "apple apple"})
    analysis_a = db.textAnalysis("a")
    analysis_b = db.textAnalysis("b")
    assert analysis_a.dump()["apple"]["sum"] == 1
    assert analysis_a.all("apple") == ["r1"]
    assert analysis_b.dump()["apple"]["sum"] == 2
    assert analysis_b.all("apple") == ["r2"]


def test_top_returns_row_with_highest_count():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple apple banana"})
    db.insert(PK, "r2", {"body": "apple"})
    analysis = db.textAnalysis(PK)
    assert analysis.top("apple") == "r1"
    assert analysis.top("banana") == "r1"
    assert analysis.top("missing") is None


def test_filter_keeps_terms_matching_predicate():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple banana"})
    db.insert(PK, "r2", {"body": "banana blueberry"})
    result = db.textAnalysis(PK).filter(lambda term: term.startswith("b"))
    assert set(result.keys()) == {"banana", "blueberry"}
    assert result["banana"]["sum"] == 2


def test_filter_with_non_callable_returns_empty_list():
    analysis = FullTextAnalysis({})
    assert analysis.filter("b") == []


def test_serialize_returns_unique_row_keys_in_term_order():
    db = MemoryDb()
    db.textIndex(PK, {"body": True})
    db.insert(PK, "r1", {"body": "apple apple banana"})
    db.insert(PK, "r2", {"body": "apple"})
    db.insert(PK, "r3", {"body": "banana"})
    analysis = db.textAnalysis(PK)
    filtered = analysis.filter(lambda term: True)
    assert analysis.serialize(filtered) == ["r1", "r2", "r3"]


def test_highlight_wraps_whole_word_matches_only():
    analysis = FullTextAnalysis({})
    out = analysis.highlight("fox", "the fox and foxes", "<b>{{_}}</b>")
    assert out == "the <b>fox</b> and foxes"


def test_highlight_is_case_sensitive():
    analysis = FullTextAnalysis({})
    assert analysis.highlight("fox", "Fox fox", "[{{_}}]") == "Fox [fox]"
