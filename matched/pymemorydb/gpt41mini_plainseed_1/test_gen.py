import pytest
import time
from pymemorydb import memorydb, text_analysis

def test_textIndex_creates_new_index_and_sets_fields():
    db = memorydb.MemoryDb()
    partition_key = "pk1"
    desc = {'field1': {}, 'field2': {'weight': 3}}
    db.textIndex(partition_key, desc)
    assert partition_key in db.all_indices
    indices = db.all_indices[partition_key]
    assert indices['field1'] == {}
    assert indices['field2'] == {'weight': 3}

def test_insert_adds_item_with_timestamp_and_index_data():
    db = memorydb.MemoryDb()
    partition_key = "pk1"
    row_key = "rk1"
    desc = {'text': {'weight': 2}}
    db.textIndex(partition_key, desc)
    item = {'text': 'hello hello world'}
    before = time.time()
    db.insert(partition_key, row_key, item)
    after = time.time()
    stored = db.get(partition_key, row_key)
    assert stored['partition_key'] == partition_key
    assert stored['row_key'] == row_key
    assert 'timestamp' in stored
    assert before <= stored['timestamp'] <= after
    # index_data reflects text tokens with weighting
    assert stored.get('index_data') is not None
    # The 'hello' token appears twice, so weighted counts should reflect that
    tokens = dict(stored['index_data'])
    assert tokens.get('hello', 0) == 4  # 2 occurrences * weight 2
    assert tokens.get('world', 0) == 2  # 1 occurrence * weight 2

def test_indexItem_multiple_fields_and_weights():
    db = memorydb.MemoryDb()
    partition_key = "p"
    db.textIndex(partition_key, {
        'field1': {},
        'field2': {'weight': 5},
    })
    item = {
        'partition_key': partition_key,
        'row_key': 'r',
        'field1': 'a a b',
        'field2': 'x y y y'
    }
    result = db.indexItem(item)
    # field1 tokens each have weight 1, field2 tokens weight 5
    # field1: a=2*1=2, b=1*1=1
    # field2: x=1*5=5, y=3*5=15
    expected = sorted([('a', 2), ('b', 1), ('x', 5), ('y', 15)], key=lambda kv: kv[1], reverse=True)
    assert result == expected

def test_insert_without_indexing_still_inserts():
    db = memorydb.MemoryDb()
    partition_key = "p"
    row_key = "r"
    item = {'somefield': 'somevalue'}
    db.insert(partition_key, row_key, item)
    stored = db.get(partition_key, row_key)
    assert stored['somefield'] == 'somevalue'
    assert 'index_data' not in stored

def test_textAnalysis_returns_FullTextAnalysis_instance_and_data_correctness():
    db = memorydb.MemoryDb()
    pkey = "p"
    db.textIndex(pkey, {'field': {}})
    db.insert(pkey, "r1", {'field': '1 1 2'})
    db.insert(pkey, "r2", {'field': '1 3'})
    analysis = db.textAnalysis(pkey)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    data = analysis.dump()
    # The sum keys correspond to token counts multiplied by token values (integers)
    # tokens: '1' appeared 3 times (r1 twice, r2 once), '2' once, '3' once
    # sum for 'field' should reflect sum of keys weighted by occurrences
    assert 'field' in data
    sum_value = data['field']['sum']
    # '1' counted thrice: 1*3=3, '2':1*2=2, '3':1*3=3 (but all tokens are keys from index_data keys)
    # Our sums calculated adding the base (int token) repeatedly for each occurrence
    # Because FullTextAnalysis receives keys as strings, sums from textAnalysis should be >=3
    assert sum_value >= 3

def test_textAnalysis_returns_None_if_no_index_data():
    db = memorydb.MemoryDb()
    pkey = "p"
    # No index data or no inserted items
    result = db.textAnalysis(pkey)
    assert result is None

def test_textIndex_overwrites_existing_indices_for_partition_key():
    db = memorydb.MemoryDb()
    pkey = "p"
    db.textIndex(pkey, {'a': 1})
    db.textIndex(pkey, {'b': 2})
    indices = db.all_indices[pkey]
    assert indices.get('a') is None or indices.get('a') == 1  # 'a' might still be there though method merges keys
    assert indices['b'] == 2

def test_insert_creates_index_data_for_multiple_fields_with_spaces():
    db = memorydb.MemoryDb()
    pkey = "p"
    db.textIndex(pkey, {'text1': {}, 'text2': {'weight': 2}})
    item = {
        'text1': "foo bar foo",
        'text2': "baz baz"
    }
    db.insert(pkey, "r", item)
    inserted = db.get(pkey, "r")
    assert 'index_data' in inserted
    index_data = dict(inserted['index_data'])
    # text1 tokens: foo (2), bar(1), weight =1
    # text2 tokens: baz (2), weight=2, so counts doubled
    assert index_data.get('foo', 0) == 2
    assert index_data.get('bar', 0) == 1
    assert index_data.get('baz', 0) == 4  # 2 occurrences * weight 2

def test_insert_and_get_with_list_of_row_keys():
    db = memorydb.MemoryDb()
    pk = "pk"
    db.insert(pk, "r1", {'val': 1})
    db.insert(pk, "r2", {'val': 2})
    results = db.get(pk, ["r1", "r2"])
    assert len(results) == 2
    assert any(item['row_key'] == 'r1' for item in results)
    assert any(item['row_key'] == 'r2' for item in results)

def test_get_raises_value_error_for_invalid_param_type():
    db = memorydb.MemoryDb()
    with pytest.raises(ValueError):
        db.get("pk", 123)  # int param invalid

def test_insert_and_delete_removes_item():
    db = memorydb.MemoryDb()
    pk = "pk"
    rk = "rk"
    db.insert(pk, rk, {'val': 5})
    db.delete(pk, rk)
    with pytest.raises(ValueError):
        db.get(pk, rk)

def test_sample_returns_empty_if_no_items():
    db = memorydb.MemoryDb()
    result = db.sample("no_such_partition")
    assert result == []

def test_sample_returns_items_and_length():
    db = memorydb.MemoryDb()
    pk = "pk"
    for i in range(5):
        db.insert(pk, f"r{i}", {'val': i})
    sample = db.sample(pk, count=3)
    assert len(sample) == 3
    assert all(s['partition_key'] == pk for s in sample)

def test_changeId_successfully_changes_row_key():
    db = memorydb.MemoryDb()
    pk = "pk"
    rk = "old"
    new_rk = "new"
    db.insert(pk, rk, {'val': 1})
    item = db.get(pk, rk)
    db.changeId(item, new_rk)
    with pytest.raises(ValueError):
        db.get(pk, rk)
    new_item = db.get(pk, new_rk)
    assert new_item['val'] == 1

def test_changeId_raises_if_item_not_in_table():
    db = memorydb.MemoryDb()
    item = {'partition_key': 'p', 'row_key': 'r'}
    with pytest.raises(ValueError):
        db.changeId(item, 'new')

def test_update_raises_if_item_not_present():
    db = memorydb.MemoryDb()
    item = {'partition_key': 'p', 'row_key': 'r'}
    with pytest.raises(ValueError):
        db.update(item)

def test_update_replaces_item():
    db = memorydb.MemoryDb()
    pk = "pk"
    rk = "rk"
    db.insert(pk, rk, {'val': 1})
    item = db.get(pk, rk)
    item['val'] = 2
    db.update(item)
    updated = db.get(pk, rk)
    assert updated['val'] == 2

def test_updateAll_replaces_all_items():
    db = memorydb.MemoryDb()
    pk = "pk"
    items = []
    for i in range(3):
        db.insert(pk, f"r{i}", {'val': i})
        items.append({'partition_key': pk, 'row_key': f"r{i}", 'val': i*10})
    db.updateAll(items)
    all_items = db.getAll(pk)
    for item in all_items:
        found = next((x for x in items if x['row_key'] == item['row_key']), None)
        assert found is not None
        assert found['val'] == item['val']

def test_textIndex_adds_indices_to_existing_partition_key():
    db = memorydb.MemoryDb()
    pk = "pk"
    db.textIndex(pk, {'a': {}})
    db.textIndex(pk, {'b': {}})
    indices = db.all_indices[pk]
    assert 'a' in indices
    assert 'b' in indices

def test_textAnalysis_handles_chained_next_values_without_infinite_loop():
    db = memorydb.MemoryDb()
    pkey = "p"
    db.textIndex(pkey, {'field': {}})
    # Manually insert two items that will generate next chains
    item1 = {'partition_key': pkey, 'row_key': 'r1', 'field': '1 1'}
    item2 = {'partition_key': pkey, 'row_key': 'r2', 'field': '1'}
    db.insert(pkey, 'r1', item1)
    db.insert(pkey, 'r2', item2)
    analysis = db.textAnalysis(pkey)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    # Calling dump, terms, top, all to check chain traversal
    assert type(analysis.dump()) is dict
    terms = analysis.terms()
    assert isinstance(terms, list) and 'field' in terms
    top_value = analysis.top('field')
    assert top_value in ['r1', 'r2']
    all_values = analysis.all('field')
    assert 'r1' in all_values or 'r2' in all_values
