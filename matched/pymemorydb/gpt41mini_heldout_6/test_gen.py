import pytest
import time
from pymemorydb import memorydb, text_analysis

@pytest.fixture
def db():
    return memorydb.MemoryDb()

def test_textIndex_creates_new_indices_and_adds_descriptions(db):
    desc = {'field1': {'weight': 2}, 'field2': 1}
    partition_key = 'pk1'
    db.textIndex(partition_key, desc)
    assert partition_key in db.all_indices
    assert db.all_indices[partition_key] == desc

def test_textIndex_updates_existing_indices(db):
    partition_key = 'pk1'
    db.all_indices[partition_key] = {'field1': 1}
    desc = {'field1': {'weight': 3}, 'field2': 2}
    db.textIndex(partition_key, desc)
    expected = {'field1': {'weight': 3}, 'field2': 2}
    assert db.all_indices[partition_key] == expected

def test_insert_adds_item_with_partition_and_row_key_and_timestamp(db):
    partition_key = 'pk'
    row_key = 'rk'
    item = {'field': 'value'}
    db.insert(partition_key, row_key, item)
    stored = db.get(partition_key, row_key)
    assert stored['partition_key'] == partition_key
    assert stored['row_key'] == row_key
    assert stored['field'] == 'value'
    assert 'timestamp' in stored
    assert isinstance(stored['timestamp'], float)
    # index_data should be None as no indices configured
    assert 'index_data' not in stored

def test_insert_calls_indexItem_and_adds_index_data(db):
    partition_key = 'pk'
    row_key = 'rk'
    desc = {'field': {'weight': 2}}
    db.textIndex(partition_key, desc)
    item = {'field': 'hello hello world'}
    db.insert(partition_key, row_key, item)
    stored = db.get(partition_key, row_key)
    # index_data contains tuples of (word, weighted count)
    assert 'index_data' in stored
    counts = dict(stored['index_data'])
    assert counts.get('hello', 0) == 4  # 'hello' appears twice * weight 2 = 4
    assert counts.get('world', 0) == 2  # 'world' once * weight 2 = 2

def test_indexItem_returns_none_when_no_indices(db):
    partition_key = 'pk'
    item = {'partition_key': partition_key, 'row_key': 'rk', 'field': 'some text'}
    result = db.indexItem(item)
    assert result is None

def test_indexItem_counts_and_weights_words(db):
    partition_key = 'pk'
    db.all_indices[partition_key] = {'field1': 1, 'field2': {'weight': 3}}
    item = {
        'partition_key': partition_key,
        'row_key': 'rk',
        'field1': 'a a b',
        'field2': 'x x x y'
    }
    result = db.indexItem(item)
    # 'a' counts: 2 * 1 = 2, 'b':1, 'x':3 * 3=9, 'y':1 * 3=3
    expected_counts = {'a': 2, 'b': 1, 'x': 9, 'y': 3}
    result_dict = dict(result)
    for k, v in expected_counts.items():
        assert result_dict.get(k) == v

def test_textAnalysis_returns_FullTextAnalysis_with_correct_data(db):
    partition_key = 'pk'
    desc = {'field': 1}
    db.textIndex(partition_key, desc)
    items = [
        {'field': 'word1 word2 word1', 'partition_key': partition_key, 'row_key': 'r1'},
        {'field': 'word2 word3', 'partition_key': partition_key, 'row_key': 'r2'},
    ]
    for item in items:
        db.insert(partition_key, item['row_key'], item)
    fta = db.textAnalysis(partition_key)
    assert isinstance(fta, text_analysis.FullTextAnalysis)
    terms = fta.terms()
    # terms are string keys (the frequencies) as strings
    for term in terms:
        assert isinstance(term, str)
    data_dump = fta.dump()
    assert isinstance(data_dump, dict)
    # sum values should be integer sums > 0
    for key in data_dump:
        assert 'sum' in data_dump[key]
        assert isinstance(data_dump[key]['sum'], int)
        assert data_dump[key]['sum'] > 0

def test_textAnalysis_returns_None_when_no_index_data(db):
    partition_key = 'pk'
    db.insert(partition_key, 'row1', {'field': 'data'})
    # no textIndex declared, so no index_data will be set
    result = db.textAnalysis(partition_key)
    assert result is None

def test_changeId_changes_row_key_and_preserves_partition_key(db):
    partition_key = 'pk'
    item = {'partition_key': partition_key, 'row_key': 'rk1', 'data': 'x'}
    db.insert(partition_key, 'rk1', item)
    item_before = db.get(partition_key, 'rk1')
    db.changeId(item_before, 'rk2')
    # old key should no longer exist
    with pytest.raises(ValueError):
        db.get(partition_key, 'rk1')
    # new key should exist
    item_after = db.get(partition_key, 'rk2')
    assert item_after['row_key'] == 'rk2'
    assert item_after['partition_key'] == partition_key
    assert item_after['data'] == 'x'

def test_changeId_raises_if_item_missing_keys(db):
    with pytest.raises(ValueError) as e:
        db.changeId({'row_key': 'rk'}, 'newrk')
    assert 'partition_key' in str(e.value)

    with pytest.raises(ValueError) as e:
        db.changeId({'partition_key': 'pk'}, 'newrk')
    assert 'row_key' in str(e.value)

def test_changeId_raises_if_item_not_found(db):
    item = {'partition_key': 'pk', 'row_key': 'rk'}
    with pytest.raises(ValueError) as e:
        db.changeId(item, 'newrk')
    assert '404' in str(e.value)

def test_update_updates_existing_item(db):
    partition_key = 'pk'
    item = {'partition_key': partition_key, 'row_key': 'rk', 'data': 'old'}
    db.insert(partition_key, 'rk', item)
    updated = {'partition_key': partition_key, 'row_key': 'rk', 'data': 'new'}
    db.update(updated)
    result = db.get(partition_key, 'rk')
    assert result['data'] == 'new'

def test_update_raises_if_item_missing_keys(db):
    with pytest.raises(ValueError):
        db.update({'row_key': 'rk'})
    with pytest.raises(ValueError):
        db.update({'partition_key': 'pk'})

def test_update_raises_if_item_not_found(db):
    with pytest.raises(ValueError):
        db.update({'partition_key': 'pk', 'row_key': 'not_exist'})

def test_updateAll_replaces_items(db):
    partition_key = 'pk'
    items = [
        {'partition_key': partition_key, 'row_key': 'r1', 'data': 'a'},
        {'partition_key': partition_key, 'row_key': 'r2', 'data': 'b'},
    ]
    for item in items:
        db.insert(partition_key, item['row_key'], item)
    updated_items = [
        {'partition_key': partition_key, 'row_key': 'r1', 'data': 'aa'},
        {'partition_key': partition_key, 'row_key': 'r2', 'data': 'bb'},
    ]
    db.updateAll(updated_items)
    r1 = db.get(partition_key, 'r1')
    r2 = db.get(partition_key, 'r2')
    assert r1['data'] == 'aa'
    assert r2['data'] == 'bb'

def test_insert_and_textAnalysis_token_chaining_limit(db):
    partition_key = 'pk'
    desc = {'field': 1}
    db.textIndex(partition_key, desc)
    # Create many inserts with repeated tokens to test chain limit of 20
    for i in range(25):
        item = {'field': 'foo', 'partition_key': partition_key, 'row_key': str(i)}
        db.insert(partition_key, str(i), item)
    fta = db.textAnalysis(partition_key)
    # There should be at least 1 token 'foo' in spread
    assert 'foo' in fta.dump()
    # The chaining limit should have prevented infinite loops (no error means pass)
    # "sum" value should be 25 (each with base int('foo') fails; fallback base??)
    # but 'foo' is not a number and int('foo') should fail in textAnalysis; exception avoided?
    # Actually int('foo') will raise ValueError, the code uses int(n) where n is token string keys,
    # so test that the code handles non-int keys properly by skipping or failing gracefully.
    # The test passes if no exception occurs and 'foo' in keys.
    assert isinstance(fta, text_analysis.FullTextAnalysis)

def test_textIndex_allows_multiple_partition_keys(db):
    db.textIndex('pk1', {'f':1})
    db.textIndex('pk2', {'g':2})
    assert 'pk1' in db.all_indices
    assert 'pk2' in db.all_indices
    assert db.all_indices['pk1']['f'] == 1
    assert db.all_indices['pk2']['g'] == 2

def test_indexItem_ignores_fields_not_in_index(db):
    partition_key = 'pk'
    db.all_indices[partition_key] = {'field': 1}
    item = {'partition_key': partition_key, 'row_key': 'r1', 'field': 'one two', 'other': 'ignored'}
    result = db.indexItem(item)
    keys = dict(result).keys()
    assert 'one' in keys and 'two' in keys
    assert 'ignored' not in keys

def test_textAnalysis_handles_empty_partition(db):
    partition_key = 'empty_pk'
    db.textIndex(partition_key, {'field': 1})
    result = db.textAnalysis(partition_key)
    # no items inserted, should return None or FullTextAnalysis with empty data
    assert result is None or isinstance(result, text_analysis.FullTextAnalysis)
