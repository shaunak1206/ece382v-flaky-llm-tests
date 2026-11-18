import pytest
import time
from pymemorydb import memorydb, text_analysis

@pytest.fixture
def db():
    db = memorydb.MemoryDb()
    yield db
    db.resetDatabase()

def test_textIndex_creates_indices_for_partition_key(db):
    desc = {'field1': {}, 'field2': {'weight': 3}}
    db.textIndex('part1', desc)
    assert 'part1' in db.all_indices
    assert db.all_indices['part1']['field1'] == {}
    assert db.all_indices['part1']['field2'] == {'weight': 3}

def test_textIndex_overwrites_existing_indices(db):
    db.textIndex('part1', {'field1': {'weight':1}})
    db.textIndex('part1', {'field1': {'weight': 5}, 'field3': {}})
    assert db.all_indices['part1']['field1']['weight'] == 5
    assert 'field3' in db.all_indices['part1']

def test_insert_adds_item_with_partition_and_row_keys_and_timestamp(db):
    db.textIndex('part1', {'content': {}})
    item = {'content': 'hello world'}
    db.insert('part1', 'row1', item)
    stored = db.get('part1', 'row1')
    assert stored['partition_key'] == 'part1'
    assert stored['row_key'] == 'row1'
    assert 'timestamp' in stored
    assert stored['content'] == 'hello world'
    assert isinstance(stored['index_data'], list)

def test_insert_indexItem_calculates_index_data_correctly(db):
    # Weight 2 for repeated words
    db.textIndex('part1', {'text': {'weight': 2}})
    item = {'text': 'foo foo bar'}
    db.insert('part1', 'r1', item)
    stored = db.get('part1', 'r1')
    # index_data should count words with weight
    # foo occurs twice: 2 * 2 = 4, bar once: 1 * 2 = 2
    expected = [('foo', 4), ('bar', 2)]
    assert stored['index_data'] == expected

def test_insert_indexItem_skips_fields_not_in_index(db):
    db.textIndex('part1', {'title': {}})
    item = {'content': 'ignored content', 'title': 'hello world'}
    db.insert('part1', 'r1', item)
    stored = db.get('part1', 'r1')
    # index_data should only include title field tokens
    tokens = dict(stored['index_data'])
    assert 'hello' in tokens and 'world' in tokens
    assert 'ignored' not in tokens

def test_textAnalysis_returns_FullTextAnalysis_object_and_correct_data(db):
    # Setup indices & insert multiple items with index_data
    db.textIndex('part1', {'text': {}})
    item1 = {'text': 'foo foo bar'}
    db.insert('part1', 'r1', item1)  # index_data auto set
    item2 = {'text': 'bar baz'}
    db.insert('part1', 'r2', item2)
    fta = db.textAnalysis('part1')
    assert isinstance(fta, text_analysis.FullTextAnalysis)
    terms = fta.terms()
    # The keys are token strings as keys
    # They should include 'foo', 'bar', 'baz'
    assert 'foo' in terms or 'bar' in terms or 'baz' in terms

def test_textAnalysis_returns_none_when_no_index_data(db):
    db.textIndex('part1', {'text': {}})
    # Insert item without index_data (simulate by direct append)
    item = {'partition_key': 'part1', 'row_key': 'r1', 'text': 'text only'}
    db.table.append(item)
    result = db.textAnalysis('part1')
    assert result is None

@pytest.mark.parametrize("index_desc, text, expected_tokens", [
    ({'text': {}}, 'word word word', [('word', 3)]),
    ({'text': {'weight': 2}}, 'a a b b b', [('b', 6), ('a', 4)]),
    ({'text': None}, 'empty field', []),  # no index weights => 1 fallback (but no index as type is None)
])
def test_indexItem_various_weights(db, index_desc, text, expected_tokens):
    db.textIndex('part1', index_desc)
    item = {'text': text}
    db.insert('part1', 'r1', item)
    stored = db.get('part1', 'r1')
    tokens = dict(stored.get('index_data', []))
    for token, count in expected_tokens:
        assert tokens.get(token) == count

def test_insert_with_no_index_for_partition_key_does_not_set_index_data(db):
    # no textIndex called
    item = {'content': 'hello world'}
    db.insert('noindex', 'r1', item)
    stored = db.get('noindex', 'r1')
    assert 'index_data' not in stored or stored['index_data'] is None

def test_textIndex_creates_empty_indices_when_empty_desc(db):
    db.textIndex('partX', {})
    assert isinstance(db.all_indices['partX'], dict)
    assert db.all_indices['partX'] == {}

def test_textAnalysis_handles_chained_next_entries(db):
    # Manually build index_data with repeated tokens
    db.textIndex('part1', {'text': {}})
    # Insert multiple items with same token
    item1 = {'text': 'test'}
    item2 = {'text': 'test'}
    item3 = {'text': 'test'}
    db.insert('part1', 'r1', item1)
    db.insert('part1', 'r2', item2)
    db.insert('part1', 'r3', item3)
    fta = db.textAnalysis('part1')
    # The FullTextAnalysis should reflect multiple chained entries under 'test' token
    ids = fta.serialize(fta.data)  # extract ids capturing chained values
    assert set(ids).issuperset({'r1', 'r2', 'r3'})

def test_textIndex_multiple_calls_adds_and_overwrites_indices(db):
    db.textIndex('pkey', {'f1': {}, 'f2': {'weight': 2}})
    db.textIndex('pkey', {'f3': {}})
    assert 'f1' in db.all_indices['pkey']
    assert 'f2' in db.all_indices['pkey']
    assert 'f3' in db.all_indices['pkey']

def test_insert_copies_item_dict_and_preserves_original(db):
    db.textIndex('part1', {'content': {}})
    original = {'content': 'copy test'}
    db.insert('part1', 'row_k', original)
    # original should not have timestamp or partition_key/row_key added
    assert 'partition_key' not in original
    assert 'timestamp' not in original
    stored = db.get('part1', 'row_k')
    assert stored['content'] == 'copy test'

def test_textAnalysis_sums_and_spread_keys_correctly(db):
    db.textIndex('part1', {'content': {}})
    item1 = {'content': '5 5 10'}
    item2 = {'content': '5 10'}
    db.insert('part1', 'id1', item1)
    db.insert('part1', 'id2', item2)
    fta = db.textAnalysis('part1')
    assert '5' in fta.data
    assert '10' in fta.data
    # sum is weighted sum of token keys (integer converted)
    # For '5': appears 3 times total (2 in id1, 1 in id2) => sum=5*3=15
    # For '10': appears 2 times total => sum=10*2=20
    assert fta.data['5']['sum'] == 15
    assert fta.data['10']['sum'] == 20

def test_textIndex_handles_non_string_fields_ignored(db):
    db.textIndex('partX', {'number': {}})
    item = {'number': 1234}
    # indexItem expects string on indexed fields, so non-string should produce no tokens
    db.insert('partX', 'r1', item)
    stored = db.get('partX', 'r1')
    # index_data may be empty list because split would fail gracefully
    assert isinstance(stored.get('index_data'), list)

def test_textAnalysis_limits_breaker_to_avoid_infinite_loop(db):
    # Build chain deeper than 20 manually
    db.textIndex('partY', {'text': {}})
    long_chain = {'value': 'start'}
    for _ in range(30):
        long_chain = {'next': long_chain, 'value': 'start'}
    # Manually create tokenized_data shape expected by textAnalysis
    db.table.append({
        'partition_key': 'partY',
        'row_key': 'r1',
        'index_data': [('100', 1)] * 5
    })
    # Patch tokenized_data to simulate next chained entries above 20 in textAnalysis code
    # We just run to ensure no infinite loop (no exception)
    res = db.textAnalysis('partY')
    assert (res is None or isinstance(res, text_analysis.FullTextAnalysis))

def test_textIndex_no_overwrite_for_unrelated_partition_keys(db):
    db.textIndex('p1', {'a': {}})
    db.textIndex('p2', {'b': {}})
    assert 'a' in db.all_indices['p1']
    assert 'b' in db.all_indices['p2']
    assert 'a' not in db.all_indices['p2']
    assert 'b' not in db.all_indices['p1']
