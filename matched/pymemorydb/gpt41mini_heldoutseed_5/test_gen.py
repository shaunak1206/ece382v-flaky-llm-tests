import pytest
import time
from pymemorydb import memorydb, text_analysis

@pytest.fixture
def db():
    return memorydb.MemoryDb()

def test_textIndex_creates_and_updates_indices(db):
    partition_key = 'p1'
    desc1 = {'field1': {}, 'field2': {'weight': 2}}
    desc2 = {'field3': {}}
    # Initially empty
    assert partition_key not in db.all_indices
    # Create indices
    db.textIndex(partition_key, desc1)
    assert partition_key in db.all_indices
    assert db.all_indices[partition_key] == desc1
    # Update indices (adds field3)
    db.textIndex(partition_key, desc2)
    assert db.all_indices[partition_key] == {**desc1, **desc2}

def test_insert_adds_item_with_keys_and_timestamp(db):
    partition_key = 'p1'
    row_key = 'r1'
    item = {'field': 'some value'}
    before = time.time()
    db.insert(partition_key, row_key, item)
    after = time.time()
    stored = db.table[-1]
    # Check keys set
    assert stored['partition_key'] == partition_key
    assert stored['row_key'] == row_key
    # Check item data copied
    assert stored['field'] == 'some value'
    # Timestamp between before and after call
    assert before <= stored['timestamp'] <= after
    # index_data likely None if no index defined
    assert 'index_data' not in stored or isinstance(stored['index_data'], list)

def test_insert_with_index_data_populates_index_data(db):
    partition_key = 'pi'
    row_key = 'r1'
    desc = {'text': {'weight': 3}}
    db.textIndex(partition_key, desc)
    item = {'text': 'hello hello world'}
    db.insert(partition_key, row_key, item)
    stored = db.table[-1]
    assert 'index_data' in stored
    # Should be sorted by value descending, expect 'hello' weighted more
    assert isinstance(stored['index_data'], list)
    keys = [k for k, _ in stored['index_data']]
    assert 'hello' in keys and 'world' in keys
    # The count for 'hello' should be higher (2*weight=6)
    counts = dict(stored['index_data'])
    assert counts['hello'] > counts['world']

def test_textAnalysis_returns_FullTextAnalysis_with_correct_structure(db):
    pk = 'pk1'
    desc = {'content': {}}
    db.textIndex(pk, desc)
    # Insert two items with overlapping 'content'
    db.insert(pk, 'r1', {'content': '1 2 2'})
    db.insert(pk, 'r2', {'content': '2 3 3 3'})
    fta = db.textAnalysis(pk)
    assert isinstance(fta, text_analysis.FullTextAnalysis)
    terms = fta.terms()
    # Terms should be string keys of numbers from text index
    expected_terms = {1,2,3}
    assert set(map(int, terms)) == expected_terms or set(terms) == set(str(i) for i in expected_terms)
    # Check sum values and spread exist for some term
    for term in terms:
        data = fta.data.get(term)
        assert 'sum' in data
        assert 'spread' in data
        assert isinstance(data['spread'], dict)

def test_textAnalysis_returns_none_if_no_index_data(db):
    pk = 'pk2'
    db.insert(pk, 'r1', {'field': 'no index data'})
    # No indices configured for pk, so insert has no index_data
    result = db.textAnalysis(pk)
    assert result is None

def test_indexItem_returns_none_if_no_indices(db):
    item = {'partition_key':'p', 'row_key':'r', 'foo': 'bar'}
    result = db.indexItem(item)
    assert result is None

def test_indexItem_calculates_words_and_weights(db):
    pk = 'pkey'
    desc = {'description': {'weight': 2}, 'title': {}}
    db.textIndex(pk, desc)
    item = {'partition_key': pk, 'row_key': 'rk', 'description': 'one one two', 'title': 'two two'}
    res = db.indexItem(item)
    # Should have counts with weights applied ('one': 2 * 2=4 counts, 'two': weighted in desc and once unweighted)
    counts = dict(res)
    assert counts.get('one', 0) == 4
    # 'two' should also appear
    assert 'two' in counts

def test_textIndex_adds_weight_key_to_index(db):
    pk = 'pweight'
    desc = {'text': {'weight': 5}}
    db.textIndex(pk, desc)
    assert pk in db.all_indices
    assert db.all_indices[pk]['text']['weight'] == 5

def test_insert_and_get_retrieve_same_item(db):
    pk = 'myp'
    rk = 'myr'
    item = {'foo': 'bar'}
    db.insert(pk, rk, item)
    got = db.get(pk, rk)
    assert got['foo'] == 'bar'
    assert got['partition_key'] == pk
    assert got['row_key'] == rk

def test_get_raises_valueerror_on_missing_row_key(db):
    pk = 'pkx'
    with pytest.raises(ValueError) as e:
        db.get(pk, 'nonexistent')
    assert e.value.args[0] == 404

def test_get_with_list_returns_multiple_items(db):
    pk = 'pklist'
    items = [{'foo': 'a'}, {'foo': 'b'}, {'foo': 'c'}]
    for i, item in enumerate(items):
        db.insert(pk, f'r{i}', item)
    results = db.get(pk, ['r0', 'r1'])
    assert len(results) == 2
    assert results[0]['foo'] == 'a'
    assert results[1]['foo'] == 'b'

def test_textIndex_allows_empty_description(db):
    pk = 'emptydesc'
    db.textIndex(pk, {})
    assert pk in db.all_indices
    assert db.all_indices[pk] == {}

def test_textAnalysis_handles_linked_next_chains(db):
    pk = 'linktest'
    desc = {'field': {}}
    db.textIndex(pk, desc)
    # Manually create items with 'index_data' containing chained 'next'
    item1 = {'partition_key': pk, 'row_key': 'r1', 'index_data': [(1,1)]}
    item2 = {'partition_key': pk, 'row_key': 'r2', 'index_data': [(1,1)]}
    # Simulate chaining in tokenized_data inside textAnalysis via insertions
    db.table.append(item1)
    db.table.append(item2)
    analysis = db.textAnalysis(pk)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    terms = analysis.terms()
    assert any(str(1) == t or 1 == t for t in terms)

def test_textAnalysis_respects_max_breaker_limit(db):
    pk = 'breaker'
    desc = {'field': {}}
    db.textIndex(pk, desc)
    # Create 25 items with same index_data to trigger breaker > 20 limit
    for i in range(25):
        item = {'partition_key': pk, 'row_key': f'r{i}', 'index_data': [(1,1)]}
        db.table.append(item)
    res = db.textAnalysis(pk)
    assert isinstance(res, text_analysis.FullTextAnalysis)

def test_indexItem_skips_fields_not_in_indices(db):
    pk = 'skipfield'
    desc = {'title': {}}
    db.textIndex(pk, desc)
    item = {'partition_key': pk, 'row_key': 'rk', 'body': 'ignore me'}
    res = db.indexItem(item)
    # Should be empty since 'body' not indexed
    assert res == []

def test_insert_copies_item_and_preserves_non_index_fields(db):
    pk = 'copytest'
    rk = 'rk1'
    item = {'title': 'hello', 'randominfo': 123}
    db.textIndex(pk, {'title': {}})
    db.insert(pk, rk, item)
    stored = db.get(pk, rk)
    assert stored['randominfo'] == 123
    assert stored['title'] == 'hello'
    assert stored is not item  # Insert makes a copy

def test_insert_multiple_increments_table_length(db):
    pk = 'multi'
    db.textIndex(pk, {'f': {}})
    len_before = len(db.table)
    for i in range(5):
        db.insert(pk, f'r{i}', {'f': 'word'})
    assert len(db.table) == len_before + 5

def test_insert_index_data_weight_default_is_one(db):
    pk = 'weightdef'
    db.textIndex(pk, {'field': {}})  # no weight specified
    item = {'partition_key': pk, 'row_key': 'r1', 'field': 'x x y'}
    res = db.indexItem(item)
    counts = dict(res)
    # 'x' count should be 2, weight default 1
    assert counts.get('x', 0) == 2

def test_textIndex_called_multiple_times_overwrites_keys(db):
    pk = 'multiindx'
    db.textIndex(pk, {'a': {}})
    db.textIndex(pk, {'b': {}})
    assert 'a' in db.all_indices[pk]
    assert 'b' in db.all_indices[pk]

def test_textIndex_creates_empty_dict_when_missing(db):
    pk = 'new-partition'
    # Confirm no indices initially
    assert pk not in db.all_indices
    db.textIndex(pk, {'field': {}})
    assert isinstance(db.all_indices[pk], dict)
    assert 'field' in db.all_indices[pk]
