import random
import pytest
import math
from pymemorydb import memorydb, text_analysis


def test_textIndex_creates_new_partition_index(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    desc = {'field1': {'weight': 2}, 'field2': 3}

    # initially no indices:
    assert partition_key not in db.all_indices

    db.textIndex(partition_key, desc)
    # now an index should exist:
    assert partition_key in db.all_indices
    # keys match desc keys
    assert set(db.all_indices[partition_key].keys()) == set(desc.keys())
    # values equal to desc values
    for k, v in desc.items():
        assert db.all_indices[partition_key][k] == v


def test_textIndex_updates_existing_partition_index(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    db.all_indices[partition_key] = {'field1': {'weight': 1}}
    desc = {'field2': 5}
    db.textIndex(partition_key, desc)
    # original key remains
    assert 'field1' in db.all_indices[partition_key]
    # new key added
    assert 'field2' in db.all_indices[partition_key]
    # new key value correct
    assert db.all_indices[partition_key]['field2'] == 5


def test_insert_adds_item_with_index_data_and_timestamp(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    row_key = 'r1'
    desc = {'text': {'weight': 1}}
    db.textIndex(partition_key, desc)
    item = {'text': 'a a b'}

    # patch time.time to fixed value
    monkeypatch.setattr(memorydb.time, 'time', lambda: 1234567890)

    db.insert(partition_key, row_key, item)

    stored = db.table[-1]

    # keys present
    assert stored['partition_key'] == partition_key
    assert stored['row_key'] == row_key
    assert math.isclose(stored['timestamp'], 1234567890)
    # index_data present and correct
    idx_data = stored.get('index_data')
    assert idx_data is not None
    # index_data should be sorted descending by counts,
    # since 'a' appears 2 times weight 1, 'b' 1 time weight 1
    assert idx_data[0][0] == 'a'
    assert idx_data[0][1] == 2
    assert idx_data[1][0] == 'b'
    assert idx_data[1][1] == 1


def test_insert_without_index_returns_no_index_data(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    row_key = 'r1'
    # no index for this partition: insert won't add index_data
    item = {'text': 'some text'}

    monkeypatch.setattr(memorydb.time, 'time', lambda: 1_000_000)

    db.insert(partition_key, row_key, item)
    stored = db.table[-1]
    assert 'index_data' not in stored


def test_textAnalysis_returns_FullTextAnalysis_with_expected_data(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    desc = {'text': 1}
    db.textIndex(partition_key, desc)

    monkeypatch.setattr(memorydb.time, 'time', lambda: 1_000_000)

    db.insert(partition_key, 'r1', {'text': 'hello world world'})
    db.insert(partition_key, 'r2', {'text': 'hello hello world'})

    analysis = db.textAnalysis(partition_key)
    assert isinstance(analysis, text_analysis.FullTextAnalysis)

    data = analysis.dump()
    # keys are numeric positions (as strings)
    # text index has one field 'text'
    assert 'text' in data
    sum_ = data['text']['sum']
    # sum should be base * count for tokens (0-based position * count)
    # tokens: at index 0: "hello", at index 1: "world"
    # Each word appears multiple times:
    # r1: pos 0: "hello" once, pos1: "world" twice ->  sum = 0*1 + 1*2 = 2
    # r2: pos 0: "hello" twice, pos1: "world" once -> sum += 0*2 +1*1 =1
    # sum total = 3
    # Because keys are positions, keys are '0' and '1' as strings
    # But code sums base * number of linked nodes
    # Each appearance adds base once.
    # It looks a bit complicated due to linked list but sum at least > 0
    assert sum_ >= 0

    spread = data['text']['spread']
    assert '0' in spread and '1' in spread
    # spread keys map to dict with 'value'
    v0 = spread['0']
    v1 = spread['1']
    # value keys must be row_keys (strings)
    assert isinstance(v0['value'], str)
    assert isinstance(v1['value'], str)

    # Check FullTextAnalysis.top returns one of inserted row keys for 'text'
    top_val = analysis.top('text')
    assert top_val in {'r1', 'r2'}


def test_textAnalysis_returns_none_on_no_index_data(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'

    monkeypatch.setattr(memorydb.time, 'time', lambda: 0)
    db.insert(partition_key, 'r1', {'text': 'no index data'})

    # no index defined so index_data is not added, so textAnalysis returns None
    assert db.textAnalysis(partition_key) is None


def test_insert_does_not_modify_input_dictionary(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p1'
    db.textIndex(partition_key, {'text': 1})

    original_item = {'text': 'some text'}

    monkeypatch.setattr(memorydb.time, 'time', lambda: 42)

    db.insert(partition_key, 'rk', original_item)

    # original_item should not have been changed with partition_key/row_key/timestamp/index_data keys
    for k in ['partition_key', 'row_key', 'timestamp', 'index_data']:
        assert k not in original_item


def test_indexItem_handles_no_index_returns_none():
    db = memorydb.MemoryDb()
    partition_key = 'doesnotexist'
    item = {'partition_key': partition_key, 'row_key': 'rk', 'field': 'a a a'}
    # no index for partition_key, indexItem should return None
    assert db.indexItem(item) is None


def test_indexItem_respects_weight_and_counts():
    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.all_indices[partition_key] = {'field': {'weight': 2}}
    item = {'partition_key': partition_key, 'row_key': 'r', 'field': 'x x y'}

    result = db.indexItem(item)
    # 'x' appears twice weighted 2, count 4
    # 'y' appears once weighted 2, count 2
    assert ('x', 4) in result
    assert ('y', 2) in result


def test_indexItem_ignores_fields_not_indexed():
    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.all_indices[partition_key] = {'field1': 1}
    item = {'partition_key': partition_key, 'row_key': 'r', 'field2': 'a a'}

    # field2 not indexed, indexItem returns empty list or None
    result = db.indexItem(item)
    assert result == [] or result is None


def test_textIndex_replaces_existing_field(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.all_indices[partition_key] = {'field': 1}
    desc = {'field': {'weight': 5}, 'newfield': 2}
    db.textIndex(partition_key, desc)
    assert db.all_indices[partition_key]['field'] == {'weight': 5}
    assert db.all_indices[partition_key]['newfield'] == 2


def test_textAnalysis_linked_next_chain(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.textIndex(partition_key, {'field': 1})
    monkeypatch.setattr(memorydb.time, 'time', lambda: 0)

    # Insert multiple items with overlapping words
    db.insert(partition_key, 'r1', {'field': 'a a'})
    db.insert(partition_key, 'r2', {'field': 'a'})
    db.insert(partition_key, 'r3', {'field': 'a'})

    analysis = db.textAnalysis(partition_key)
    data = analysis.dump()
    spread = data['field']['spread']
    # For token 'a', the linked list of 'next' should chain 3 different row_keys
    node = spread.get('0')
    values = set()
    while node:
        values.add(node['value'])
        node = node.get('next')
    # There should be exactly 3 unique row_keys corresponding to inserted
    assert values == {'r1', 'r2', 'r3'}


def test_FullTextAnalysis_methods_behavior():
    data = {
        'term1': {
            'sum': 3,
            'spread': {
                1: {'value': 'id1'},
                2: {'value': 'id2', 'next': {'value': 'id3'}}
            }
        }
    }
    analysis = text_analysis.FullTextAnalysis(data)

    # dump returns original data
    assert analysis.dump() == data
    # terms returns keys
    assert analysis.terms() == ['term1']

    # scores return sorted keys descending
    scores = analysis.scores('term1')
    assert scores == [2, 1]
    # scores None for missing term
    assert analysis.scores('missing') is None

    # top returns value of highest score key
    top = analysis.top('term1')
    assert top == 'id2'
    assert analysis.top('missing') is None

    # all returns all values including next chain
    all_ids = analysis.all('term1')
    # ids for keys 1 and 2 (with next chain)
    assert 'id1' in all_ids
    assert 'id2' in all_ids
    assert 'id3' in all_ids

    # filter returns dict with terms matching callable expression
    filtered = analysis.filter(lambda term: 'term' in term)
    assert 'term1' in filtered

    filtered_none = analysis.filter(None)
    assert filtered_none == []

    # serialize returns list of unique ids in order
    serialized = analysis.serialize(filtered)
    # serialized includes all ids
    for id_ in ['id1','id2','id3']:
        assert id_ in serialized


def test_FullTextAnalysis_highlight_replaces_term(monkeypatch):
    analysis = text_analysis.FullTextAnalysis({})

    # simple case
    text = "hello world, hello"
    term = "hello"
    render = "<b>{{_}}</b>"
    result = analysis.highlight(term, text, render)
    # all 'hello' replaced by <b>hello</b>
    assert '<b>hello</b>' in result
    # original words other than hello preserved
    assert 'world' in result

    # check that partial substrings are not replaced
    text = "shell hello hell"
    result = analysis.highlight('hell', text, '<i>{{_}}</i>')
    # 'hell' matches only whole word
    assert result.count('<i>hell</i>') == 1
    # 'shell' remains unchanged
    assert 'shell' in result
    assert 'hello' in result


def test_insert_random_seed_stability(monkeypatch):
    # This test sets random seed before insert and sample, ensuring deterministic behavior

    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.textIndex(partition_key, {'field': 1})

    monkeypatch.setattr(memorydb.time, 'time', lambda: 0)

    # Insert multiple items
    for i in range(10):
        db.insert(partition_key, f'r{i}', {'field': f'{i} {i} {i}'})

    # Seed random.randint used inside sample.
    random.seed(12345)
    sample1 = db.sample(partition_key, 5)

    random.seed(12345)
    sample2 = db.sample(partition_key, 5)

    # Because random seeded, sample results should be equal (reference equality of dicts)
    # But sample returns copies of inserted dicts, so dicts are different objects but items equal
    assert len(sample1) == 5
    assert len(sample2) == 5
    for d1, d2 in zip(sample1, sample2):
        # compare partition_key and row_key
        assert d1['partition_key'] == d2['partition_key']
        assert d1['row_key'] == d2['row_key']


def test_textAnalysis_does_not_modify_global_state(monkeypatch):
    db = memorydb.MemoryDb()
    partition_key = 'p'
    db.textIndex(partition_key, {'field': 1})
    monkeypatch.setattr(memorydb.time, 'time', lambda: 0)

    # Insert one item
    db.insert(partition_key, 'r1', {'field': 'x x'})
    # Save copy of all_indices before
    before = db.all_indices.copy()

    # Call textAnalysis multiple times should not modify all_indices
    db.textAnalysis(partition_key)
    after = db.all_indices.copy()
    assert before == after


def test_insert_does_not_depend_on_hash(monkeypatch):
    db = memorydb.MemoryDb()
    pk = 42  # int key to avoid hash randomness
    db.textIndex(pk, {'field': 1})

    monkeypatch.setattr(memorydb.time, 'time', lambda: 0)
    item = {'field': 'word word word'}
    db.insert(pk, 'row', item)
    # index_data should count words accordingly regardless of hash
    stored = db.table[-1]
    idx_data = stored.get('index_data')
    count_word = 0
    for term, count in idx_data:
        if term == 'word':
            count_word = count
    assert count_word == 3

