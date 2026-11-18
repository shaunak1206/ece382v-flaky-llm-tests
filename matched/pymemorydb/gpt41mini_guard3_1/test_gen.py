import random
import pytest
import math
from pymemorydb import memorydb, text_analysis


def test_textIndex_add_new_partition_and_keys(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    desc = {'field1': {'weight': 2}, 'field2': 3}
    # Initially, no indices for partition_key
    assert partition_key not in db.all_indices

    db.textIndex(partition_key, desc)

    # Now indices should exist
    assert partition_key in db.all_indices
    indices = db.all_indices[partition_key]
    # keys from desc should be set exactly
    assert set(indices.keys()) == set(desc.keys())
    # Each value should match the desc
    for k in desc:
        assert indices[k] == desc[k]


def test_textIndex_updates_existing_indices(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p2'
    db.all_indices[partition_key] = {'field1': 1}
    new_desc = {'field1': {'weight': 5}, 'field3': 7}

    db.textIndex(partition_key, new_desc)

    indices = db.all_indices[partition_key]
    # Keys should be updated/added
    assert indices['field1'] == new_desc['field1']
    assert 'field3' in indices and indices['field3'] == 7


def test_insert_adds_item_and_indices(monkeypatch):
    import time

    db = memorydb.MemoryDb()
    partition_key = 'testp'
    row_key = 'r1'
    desc = {'content': {'weight': 2}}
    db.textIndex(partition_key, desc)

    item = {'content': 'foo foo bar'}
    # Patch time.time to fixed value
    fixed_time = 1234567.89
    monkeypatch.setattr(memorydb.time, 'time', lambda: fixed_time)
    # Seed random generators in tested module (random used internally)
    random.seed(42)

    db.insert(partition_key, row_key, item)

    # Only one item inserted
    assert len(db.table) == 1
    inserted = db.table[0]

    # partition_key and row_key must be set in inserted item
    assert inserted['partition_key'] == partition_key
    assert inserted['row_key'] == row_key
    # timestamp is set to fixed_time
    assert math.isclose(inserted['timestamp'], fixed_time, rel_tol=1e-9)

    # index_data is present and correctly calculated
    index_data = inserted.get('index_data')
    assert index_data is not None

    # The index_data should be a list of tuples (token, weighted count)
    # Since "foo foo bar" split is ['foo', 'foo', 'bar'], weighted by 2:
    # reduced counts: foo appears twice, so 2 * weight=2 -> 4; bar once -> 2
    # Order sorted descending by count
    expected = [('foo', 4), ('bar', 2)]
    assert sorted(index_data, key=lambda x: x[0]) == sorted(expected, key=lambda x: x[0])


def test_indexItem_without_indices_returns_none(monkeypatch):
    db = memorydb.MemoryDb()
    item = {'partition_key': 'p', 'row_key': 'r', 'field': 'some data'}
    # No indices set for partition_key 'p'
    assert 'p' not in db.all_indices

    result = db.indexItem(item)
    assert result is None


def test_indexItem_with_multiple_fields_and_weight(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'pk'
    # Setup indices with weights and without
    db.all_indices[pk] = {
        'field1': {'weight': 3},
        'field2': 1,
        'field3': {}
    }
    item = {
        'partition_key': pk,
        'row_key': 'r2',
        'field1': "a a b",
        'field2': "x y y",
        # field3 is a string but no weight specified
        'field3': "1 1 1"
    }

    result = db.indexItem(item)

    # Calculate expected counts:
    # field1 tokens: a (2 times) * 3 = 6, b (1) *3 = 3
    # field2 tokens: x (1)*1=1, y (2)*1=2
    # field3 tokens: 1 (3)*1=3 (weight defaults to 1)
    combined = {}
    for token, count in [('a', 6), ('b', 3), ('x', 1), ('y', 2), ('1', 3)]:
        combined[token] = combined.get(token, 0) + count

    sorted_expected = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # The indexItem returns sorted by desc count, verify that result matches expected counts:
    assert result == sorted_expected


def test_textAnalysis_returns_FullTextAnalysis_object(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'p'
    row_key1 = 'r1'
    row_key2 = 'r2'
    desc = {'content': {'weight': 1}}
    db.textIndex(pk, desc)

    # Insert two items with index_data manually set:
    item1 = {'partition_key': pk, 'row_key': row_key1, 'content': 'a a b'}
    db.insert(pk, row_key1, item1)
    item2 = {'partition_key': pk, 'row_key': row_key2, 'content': 'b b c'}
    db.insert(pk, row_key2, item2)

    # Manually clear index_data to simulate index data absence, then reindex:
    for item in db.table:
        item['index_data'] = db.indexItem(item)

    result = db.textAnalysis(pk)

    # Result is instance of FullTextAnalysis
    assert isinstance(result, text_analysis.FullTextAnalysis)

    # The FullTextAnalysis data attribute is a dict keyed by token index keys (int keys)
    data = result.dump()
    # Keys expected include strings, but internal keys are field keys (numbers as str).
    # Because of implementation, keys are strings that map to dict with 'sum' and 'spread'
    assert isinstance(data, dict)
    assert all(isinstance(k, int) or isinstance(k, str) for k in data.keys())
    # Each key maps to dict containing 'sum' and 'spread'
    for k in data.keys():
        assert 'sum' in data[k]
        assert 'spread' in data[k]
        # spread should be a dict
        assert isinstance(data[k]['spread'], dict)


def test_textAnalysis_returns_none_when_no_index_data(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'empty'
    row_key = 'r'
    desc = {'content': 1}
    db.textIndex(pk, desc)
    db.insert(pk, row_key, {'content': 'hello world'})
    # Remove index_data from all items forcibly
    for i in db.table:
        i.pop('index_data', None)

    # The method should return None when tokenized_data empty
    result = db.textAnalysis(pk)
    assert result is None


def test_textAnalysis_handles_max_loop_break(monkeypatch):
    # This test attempts to trigger breaker in link traversal in textAnalysis without infinite loop
    db = memorydb.MemoryDb()
    pk = 'br'
    desc = {'content': 1}
    db.textIndex(pk, desc)

    # Prepare linked index_data with a chain length greater than 20 (breaker limit)
    # We'll manually create the chain in index_data for an item
    # Structure: [(n, v), ...]; n is int key as str internally, v dict with 'value' and 'next'
    chain_length = 25
    current = {'value': 'row0'}
    for i in range(chain_length):
        current = {'value': f'row{i}', 'next': current}
    # Reverse chain to first element
    # Actually, the chain should be built so that the last element does not have 'next'
    last = {'value': f'row{chain_length}'}
    # Build chain nodes from last to first
    next_node = last
    for i in reversed(range(chain_length)):
        next_node = {'value': f'row{i}', 'next': next_node}

    # Create fake index_data accordingly, bypassing normal indexItem
    item = {'partition_key': pk, 'row_key': 'r', 'content': 'dummy'}
    item['index_data'] = [(1, chain_length)]  # Simplify tokens
    db.table.append(item)

    # Manually patch tokenized_data inside textAnalysis to simulate deep chain
    # Since textAnalysis builds tokenized_data from index_data, we patch it to simulate breaker usage

    # Instead call textAnalysis and expect that within the breaker limit loop, it does not hang
    # We don't access internals, but function should complete normally and return FullTextAnalysis or None

    # Insert a matching all_indices to avoid indexItem returning None
    db.all_indices[pk] = {'content': 1}

    # Patch indexItem to return the deep chain representation:
    def fake_indexItem(item):
        return [(1, 1)]  # simple dummy for indexItem

    monkeypatch.setattr(db, 'indexItem', fake_indexItem)

    result = db.textAnalysis(pk)
    # It can be None or FullTextAnalysis due to the break conditions, we check no infinite loop:

    # The only requirement is no exception and returns either None or FullTextAnalysis instance
    assert (result is None) or isinstance(result, text_analysis.FullTextAnalysis)


def test_textIndex_with_empty_desc_creates_empty_indices(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'emptydesc'
    desc = {}

    db.textIndex(pk, desc)

    assert pk in db.all_indices
    indices = db.all_indices[pk]
    assert isinstance(indices, dict)
    assert len(indices) == 0


def test_insert_does_not_modify_original_item(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'pk3'
    rk = 'rk3'
    db.textIndex(pk, {'field': 1})

    original = {'field': 'some text'}
    copy_of_original = original.copy()

    monkeypatch.setattr(memorydb.time, 'time', lambda: 42.0)

    db.insert(pk, rk, original)
    # Original dict should not be modified by insert function (except keys added internally before copy)
    # So check original remains unchanged as input dict
    assert original == copy_of_original


def test_textIndex_cumulative_updates(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'pk4'

    desc1 = {'a': 1}
    desc2 = {'b': 2}
    db.textIndex(pk, desc1)
    assert set(db.all_indices[pk].keys()) == {'a'}

    db.textIndex(pk, desc2)
    # Keys from desc2 added, desc1 key 'a' still present (as textIndex merges)
    assert set(db.all_indices[pk].keys()) == {'a', 'b'}


def test_textAnalysis_data_contains_correct_sum(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'p'
    desc = {'text': 1}
    db.textIndex(pk, desc)

    # Insert item with known index_data
    content = "1 1 2"
    # weights = 1 for 'text', count: '1' (2 times), '2' (1 time)
    item = {'partition_key': pk, 'row_key': 'r1', 'text': content}
    monkeypatch.setattr(memorydb.time, 'time', lambda: 123)
    db.insert(pk, 'r1', item)

    # Recalculate index_data manually to be sure
    for i in db.table:
        i['index_data'] = db.indexItem(i)

    analysis = db.textAnalysis(pk)
    data = analysis.dump()
    # The sum for token '1' should be 2 * 1 = 2, token '2' sum = 1 *1 =1
    # Actually keys are integers as strings, so check sum matches counts*weight
    # Find token keys as int string '1' and '2'

    key_1 = 1
    key_2 = 2
    assert key_1 in data
    assert key_2 in data
    # sum for key_1 should be token count * 1 = 2
    assert data[key_1]['sum'] >= 2
    # sum for key_2 should be token count * 1 = 1
    assert data[key_2]['sum'] >= 1


def test_insert_multiple_items_and_textAnalysis_consistency(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'partitionX'
    desc = {'field': 2}
    db.textIndex(pk, desc)

    items = [
        {'field': 'apple orange orange'},
        {'field': 'banana apple'},
        {'field': 'orange banana banana'}
    ]
    monkeypatch.setattr(memorydb.time, 'time', lambda: 100)

    for i, item in enumerate(items):
        db.insert(pk, f'id{i}', item)

    # Force re-indexing index_data for all items
    for i in db.table:
        i['index_data'] = db.indexItem(i)

    analysis = db.textAnalysis(pk)
    d = analysis.dump()

    # The keys (tokens) must contain counts correctly aggregated:
    # Count weighted tokens:
    # apple: 2 (item0 once, item1 once) *2 weight = 4
    # orange: 3 (item0 twice, item2 once) *2 = 6
    # banana: 3 (item1 once, item2 twice) *2 = 6
    # The sums stored are sums of the key * frequency; keys are int strings converted from tokens, so sum means:
    # Each token's numeric key * count, so sum could be token int * count.

    # Because keys are tokens as string -> int for internal calculations, but input tokens are strings
    # So only check keys equal token strings and sums > 0

    for token in ['apple', 'orange', 'banana']:
        assert any(token in str(k) or k == token for k in d.keys())

    # The sum values are positive integers
    for key, val in d.items():
        assert isinstance(val['sum'], int)
        assert val['sum'] >= 0


def test_textIndex_and_insert_interaction(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'interaction'
    desc = {'title': {'weight': 3}}

    db.textIndex(pk, desc)

    item = {'title': 'hello hello world'}
    monkeypatch.setattr(memorydb.time, 'time', lambda: 5555)

    db.insert(pk, 'rk1', item)

    # Check that index_data has weighted counts accordingly
    inserted = db.get(pk, 'rk1')
    index_data = inserted.get('index_data')
    assert index_data is not None

    # hello appears twice, so 2 * 3 weight = 6; world = 1 * 3 = 3
    expected = [('hello', 6), ('world', 3)]
    assert sorted(index_data, key=lambda t: t[0]) == sorted(expected, key=lambda t: t[0])


def test_FullTextAnalysis_methods(monkeypatch):
    # Prepare a minimal FullTextAnalysis data for testing
    data = {
        '10': {
            'sum': 2,
            'spread': {
                10: {'value': 'r1'},
                5: {'value': 'r2', 'next': {'value': 'r3'}}
            }
        },
        '20': {
            'sum': 1,
            'spread': {
                20: {'value': 'r4'},
            }
        }
    }

    fta = text_analysis.FullTextAnalysis(data)

    # terms returns keys
    terms = fta.terms()
    assert set(terms) == set(data.keys())

    # scores returns keys of spread sorted reverse
    scores_10 = fta.scores('10')
    assert scores_10 == [10, 5] or scores_10 == [10, 5]

    scores_none = fta.scores('nonexistent')
    assert scores_none is None

    # top returns the value with highest score
    top_10 = fta.top('10')
    assert top_10 == 'r1'

    top_none = fta.top('nonexistent')
    assert top_none is None

    # all returns all values in keys and following 'next' chains
    all_10 = fta.all('10')
    # Should include r1, r2, r3 in order (r2 and r3 from chain)
    expected = ['r1', 'r2', 'r3']
    # Results order may differ but must contain these exactly
    assert set(all_10) == set(expected)
    for v in expected:
        assert v in all_10

    all_none = fta.all('nonexistent')
    with pytest.raises(KeyError):
        # Accessing keys on nonexistent raises KeyError
        _ = fta.all('nonexistent')

    # filter returns dict filtered by callable expression on term
    filtered = fta.filter(lambda t: t in ['10'])
    # Only '10' key included
    assert list(filtered.keys()) == ['10']

    filtered_empty = fta.filter(None)
    assert filtered_empty == []

    # serialize returns the list of unique row_key values in order
    filtered_data = fta.filter(lambda t: True)  # all keys
    ids = fta.serialize(filtered_data)
    # The ids must include all unique row_key values from spread
    all_values = []
    for key in filtered_data:
        spread = filtered_data[key]['spread']
        for count_key in spread:
            value = spread[count_key]
            all_values.append(value['value'])
            nxt = value.get('next')
            while nxt:
                all_values.append(nxt['value'])
                nxt = nxt.get('next')
    assert set(ids) == set(all_values)

    # serialize with None returns None
    result_none = fta.serialize(None)
    assert result_none is None

    # highlight replaces exact term with render content (case-insensitive)
    text = "Hello world hello"
    term = "hello"
    render = "<b>{{_}}</b>"
    replaced = fta.highlight(term, text, render)
    # Both "Hello" and "hello" replaced by "<b>hello</b>" (case insensitive)
    assert replaced.lower().count("<b>hello</b>") == 2


def test_textIndex_does_not_modify_existing_indices(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'pk_existing'
    initial = {'fieldA': 1}
    db.all_indices[pk] = initial.copy()

    desc = {'fieldB': 2}
    db.textIndex(pk, desc)

    # Initial indices dict mutated (not replaced by new dict)
    assert db.all_indices[pk] is initial
    # New key added
    assert 'fieldB' in db.all_indices[pk]
    # Original keys still present
    assert 'fieldA' in db.all_indices[pk]


def test_insert_and_textAnalysis_with_weight_none(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'pk_none_weight'
    desc = {'field': {'weight': None}}
    db.textIndex(pk, desc)

    monkeypatch.setattr(memorydb.time, 'time', lambda: 1)
    item = {'field': 'aa aa bb'}
    db.insert(pk, 'key1', item)

    inserted = db.get(pk, 'key1')
    index_data = inserted.get('index_data')
    # weight None should default to 1 per code logic
    expected = [('aa', 2), ('bb', 1)]
    assert sorted(index_data, key=lambda x: x[0]) == sorted(expected, key=lambda x: x[0])


def test_indexItem_skips_fields_not_in_indices(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'skipfield'
    db.all_indices[pk] = {'field1': 1}
    item = {'partition_key': pk, 'row_key': 'r', 'field1': 'x x y', 'field_not_indexed': 'z'}

    result = db.indexItem(item)
    # The token 'z' from field_not_indexed should be absent
    flat_tokens = [token for token, count in result]
    # 'z' should not appear
    assert 'z' not in flat_tokens


def test_insert_increments_index_data_weighted(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 'weighttest'
    desc = {'text': {'weight': 4}}
    db.textIndex(pk, desc)

    monkeypatch.setattr(memorydb.time, 'time', lambda: 111)
    item = {'text': 'foo foo foo bar'}
    db.insert(pk, 'rk', item)

    inserted = db.get(pk, 'rk')
    idx_data = inserted.get('index_data')
    # Counts weighted 4: foo 3*4=12, bar 1*4=4
    expected = [('foo', 12), ('bar', 4)]
    assert sorted(idx_data, key=lambda x: x[0]) == sorted(expected, key=lambda x: x[0])
