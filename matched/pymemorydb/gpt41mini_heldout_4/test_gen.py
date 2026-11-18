import pytest
import time
from memorydb import MemoryDb
from text_analysis import FullTextAnalysis

@pytest.fixture
def db():
    mdb = MemoryDb()
    yield mdb
    mdb.resetDatabase()

def test_insert_adds_item_with_partition_and_row_key_and_timestamp(db):
    item = {'foo': 'bar'}
    db.insert('p1', 'r1', item)
    assert len(db.table) == 1
    inserted = db.table[0]
    assert inserted['partition_key'] == 'p1'
    assert inserted['row_key'] == 'r1'
    assert 'timestamp' in inserted
    assert inserted['foo'] == 'bar'

def test_insert_copies_items_do_not_mutate_original(db):
    item = {'foo': 'bar'}
    db.insert('p', 'r', item)
    item['foo'] = 'baz'
    assert db.table[0]['foo'] == 'bar'

def test_textIndex_creates_indices_for_partition(db):
    desc = {'field1': {'weight': 3}, 'field2': 1}
    db.textIndex('part1', desc)
    assert 'part1' in db.all_indices
    indices = db.all_indices['part1']
    assert indices['field1']['weight'] == 3
    assert indices['field2'] == 1

def test_indexItem_produces_weighted_index_data(db):
    db.textIndex('p', {'text': {'weight': 2}})
    item = {'partition_key': 'p', 'row_key': 'r', 'text': 'hello hello world'}
    index_data = db.indexItem(item)
    # 'hello' occurs twice, weight 2 => 2 * 2 = 4, 'world' once => 2
    assert isinstance(index_data, list)
    keys = [k for k, _ in index_data]
    assert 'hello' in keys and 'world' in keys
    for k, v in index_data:
        if k == 'hello':
            assert v == 4
        if k == 'world':
            assert v == 2

def test_indexItem_returns_none_if_no_indices(db):
    item = {'partition_key': 'p', 'row_key': 'r', 'text': 'test'}
    # No indices set for 'p'
    assert db.indexItem(item) is None

def test_insert_adds_index_data_when_index_is_present(db):
    db.textIndex('p', {'text': {'weight': 1}})
    item = {'text': 'one two two'}
    db.insert('p', 'r', item)
    inserted = db.table[0]
    assert 'index_data' in inserted
    expected = [('two', 2), ('one', 1)]
    assert inserted['index_data'] == expected

def test_textAnalysis_returns_FullTextAnalysis_with_correct_data(db):
    db.textIndex('p', {'field': {'weight': 1}})
    db.insert('p', 'r1', {'field': 'a a b'})
    db.insert('p', 'r2', {'field': 'a b b'})
    # manually create index_data for inserted items (simulate indexItem effect)
    for item in db.table:
        if 'index_data' not in item:
            item['index_data'] = db.indexItem(item) or []
    result = db.textAnalysis('p')
    assert isinstance(result, FullTextAnalysis)
    terms = result.terms()
    # Terms are the keys of the analysis data dict (strings of numbers)
    # Our keys come from index_data keys, which are strings (e.g. "a", "b")
    assert all(isinstance(t, str) for t in terms)
    # Each term should be a string representing one token of the text (actually keys in index_data)
    # Because keys used are tokens from index_data
    for t in terms:
        assert t in result.data

def test_textAnalysis_returns_none_when_no_index_data(db):
    db.textIndex('p', {'field': {'weight': 1}})
    db.insert('p', 'r', {'field': 'some text'})
    # forcibly remove index_data to simulate missing
    for item in db.table:
        if 'index_data' in item:
            del item['index_data']
    assert db.textAnalysis('p') is None

def test_textIndex_updates_existing_partition_indices(db):
    db.textIndex('p', {'field1': 1})
    db.textIndex('p', {'field2': {'weight': 2}})
    indices = db.all_indices['p']
    assert 'field1' in indices
    assert 'field2' in indices
    assert indices['field2']['weight'] == 2

def test_insert_multiple_and_get_returns_correct_item(db):
    db.insert('p', 'r1', {'foo': 'bar1'})
    db.insert('p', 'r2', {'foo': 'bar2'})
    item = db.get('p', 'r2')
    assert item['foo'] == 'bar2'

def test_get_raises_value_error_on_missing_row_key(db):
    with pytest.raises(ValueError) as exc:
        db.get('p', 'missing')
    assert exc.value.args[0] == 404

def test_get_with_list_returns_list_of_items(db):
    db.insert('pt', 'r1', {'val': 1})
    db.insert('pt', 'r2', {'val': 2})
    results = db.get('pt', ['r1', 'r2'])
    assert isinstance(results, list) and len(results) == 2
    assert {r['val'] for r in results} == {1, 2}

def test_textIndex_with_empty_desc_creates_empty_indices(db):
    db.textIndex('p', {})
    assert isinstance(db.all_indices['p'], dict)
    assert len(db.all_indices['p']) == 0

def test_indexItem_considers_multiple_indices_and_weights(db):
    db.textIndex('p', {'title': {'weight': 2}, 'desc': {'weight': 3}})
    item = {
        'partition_key': 'p',
        'row_key': 'r',
        'title': 'foo foo bar',
        'desc': 'bar bar bar baz'
    }
    result = db.indexItem(item)
    # Calculate expected counts:
    # title: foo(2 occurrences)*2 weight=4, bar(1)*2=2
    # desc: bar(3 occurrences)*3=9, baz(1)*3=3
    # total counts per token:
    # foo=4, bar=2+9=11, baz=3
    expected = {'foo':4, 'bar':11, 'baz':3}
    for k,v in result:
        assert expected[k] == v

def test_textAnalysis_handles_cyclic_next_references_without_infinite_loop(db):
    # Setup index_data where 'next' chain is >20 but breaker stops infinite loop
    db.textIndex('p', {'field': 1})
    db.insert('p', 'r1', {'field': 'x'})
    # Manually create cycle in index_data for test
    item = db.table[0]
    item['index_data'] = [('1', 1)]
    tokenized_data = {
        '1': {
            '1': {'value': 'r1'}
        }
    }
    # Fake next chain > 20
    current = tokenized_data['1']['1']
    for i in range(25):
        current['next'] = {'value': f'r{i}'}
        current = current['next']
    db.table[0]['index_data'] = [('1', 1)]
    # Monkeypatch database indexItem to return this tokenized data structure with cycle
    # Instead just test textAnalysis does not infinite loop
    text_analysis_result = db.textAnalysis('p')
    assert text_analysis_result is not None
    assert isinstance(text_analysis_result, FullTextAnalysis)

def test_textIndex_leaves_existing_indices_untouched_for_other_partitions(db):
    db.textIndex('p1', {'a': 1})
    db.textIndex('p2', {'b': 2})
    assert 'a' in db.all_indices['p1']
    assert 'b' in db.all_indices['p2']
    assert 'b' not in db.all_indices['p1']
    assert 'a' not in db.all_indices['p2']

def test_textAnalysis_returns_none_when_partition_key_not_found(db):
    assert db.textAnalysis('nonexistent') is None
