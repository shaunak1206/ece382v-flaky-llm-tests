import time

import pytest

from pymemorydb import MemoryDb
from pymemorydb.text_analysis import FullTextAnalysis


def _make_corpus():
    """A partition 'docs' indexed on 'body' with four rows.

    index_data per row (count of consecutive runs, weight 1):
      r1: apple 2, banana 1
      r2: banana 3, apple 1
      r3: apple 3
      r4: apple 1
    """
    db = MemoryDb()
    db.textIndex('docs', {'body': {}})
    db.insert('docs', 'r1', {'body': 'apple apple banana'})
    db.insert('docs', 'r2', {'body': 'banana banana banana apple'})
    db.insert('docs', 'r3', {'body': 'apple apple apple'})
    db.insert('docs', 'r4', {'body': 'apple'})
    return db


# ---------------------------------------------------------------- textIndex

def test_text_index_registers_descriptor_for_partition():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}})
    assert db.all_indices == {'p': {'title': {'weight': 3}}}


def test_text_index_merges_fields_in_same_partition():
    db = MemoryDb()
    db.textIndex('p', {'a': {}})
    db.textIndex('p', {'b': {'weight': 2}})
    assert db.all_indices['p'] == {'a': {}, 'b': {'weight': 2}}


def test_text_index_overwrites_existing_field_descriptor():
    db = MemoryDb()
    db.textIndex('p', {'a': {'weight': 1}})
    db.textIndex('p', {'a': {'weight': 5}})
    assert db.all_indices['p']['a'] == {'weight': 5}


def test_text_index_partitions_are_independent():
    db = MemoryDb()
    db.textIndex('p1', {'a': {}})
    db.textIndex('p2', {'b': {}})
    assert set(db.all_indices.keys()) == {'p1', 'p2'}
    assert 'b' not in db.all_indices['p1']
    assert 'a' not in db.all_indices['p2']


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp():
    db = MemoryDb()
    before = time.time()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    after = time.time()
    assert len(db.table) == 1
    row = db.table[0]
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['name'] == 'x'
    assert before <= row['timestamp'] <= after


def test_insert_stores_shallow_copy_of_item():
    db = MemoryDb()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    assert db.table[0] is not item
    item['later'] = 1
    assert 'later' not in db.table[0]


def test_insert_without_index_has_no_index_data():
    db = MemoryDb()
    item = {'body': 'hello world'}
    db.insert('p', 'r1', item)
    assert 'index_data' not in item
    assert 'index_data' not in db.table[0]


def test_insert_in_unindexed_partition_has_no_index_data():
    db = MemoryDb()
    db.textIndex('indexed', {'body': {}})
    db.insert('other', 'r1', {'body': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_counts_tokens_with_default_weight():
    db = MemoryDb()
    db.textIndex('p', {'body': {}})
    item = {'body': 'a b a'}
    db.insert('p', 'r1', item)
    # 'a' appears in two separate runs of length 1 and they are summed.
    assert db.table[0]['index_data'] == [('a', 2), ('b', 1)]
    assert item['index_data'] == [('a', 2), ('b', 1)]


def test_insert_non_dict_descriptor_uses_weight_one():
    db = MemoryDb()
    db.textIndex('p', {'body': 7})
    db.insert('p', 'r1', {'body': 'x x y'})
    assert db.table[0]['index_data'] == [('x', 2), ('y', 1)]


def test_insert_applies_field_weights_and_sums_across_fields():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}, 'body': {}})
    db.insert('p', 'r1', {'title': 'x y', 'body': 'x'})
    # title: x=3, y=3 ; body: x=1  -> x=4, y=3
    assert db.table[0]['index_data'] == [('x', 4), ('y', 3)]


def test_insert_ignores_unindexed_fields():
    db = MemoryDb()
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'a', 'other': 'zzz zzz'})
    assert db.table[0]['index_data'] == [('a', 1)]


def test_insert_strips_whitespace_and_sorts_by_count_descending():
    db = MemoryDb()
    db.textIndex('p', {'body': {}})
    db.insert('p', 'r1', {'body': '  a a b b b  '})
    assert db.table[0]['index_data'] == [('b', 3), ('a', 2)]


def test_insert_with_no_indexed_fields_gives_empty_index_data():
    db = MemoryDb()
    db.textIndex('p', {'body': {}})
    db.insert('p', 'r1', {'unrelated': 'abc'})
    assert db.table[0]['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_when_nothing_indexed():
    db = MemoryDb()
    db.insert('p', 'r1', {'body': 'hello'})
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('missing-partition') is None


def test_text_analysis_returns_none_when_index_data_empty():
    db = MemoryDb()
    db.textIndex('p', {'body': {}})
    db.insert('p', 'r1', {'unrelated': 'abc'})
    assert db.textAnalysis('p') is None


def test_text_analysis_sums_and_spread():
    db = _make_corpus()
    analysis = db.textAnalysis('docs')
    assert type(analysis).__name__ == 'FullTextAnalysis'
    data = analysis.dump()
    assert sorted(data.keys()) == ['apple', 'banana']
    assert data['apple']['sum'] == 2 + 1 + 3 + 1
    assert data['banana']['sum'] == 1 + 3
    apple_spread = data['apple']['spread']
    assert sorted(apple_spread.keys()) == [1, 2, 3]
    assert apple_spread[2]['value'] == 'r1'
    assert apple_spread[3]['value'] == 'r3'
    # r2 and r4 share the same score (1) and are chained.
    assert apple_spread[1]['value'] == 'r2'
    assert apple_spread[1]['next']['value'] == 'r4'


def test_text_analysis_ignores_other_partitions():
    db = _make_corpus()
    db.textIndex('other', {'body': {}})
    db.insert('other', 'o1', {'body': 'cherry'})
    analysis = db.textAnalysis('docs')
    assert 'cherry' not in analysis.terms()
    other = db.textAnalysis('other')
    assert other.terms() == ['cherry']
    assert other.dump()['cherry']['sum'] == 1


def test_text_analysis_chains_rows_with_same_score():
    db = MemoryDb()
    db.textIndex('p', {'body': {}})
    for i in range(1, 6):
        db.insert('p', 'r%d' % i, {'body': 'x'})
    analysis = db.textAnalysis('p')
    assert analysis.dump()['x']['sum'] == 5
    assert analysis.scores('x') == [1]
    assert analysis.top('x') == 'r1'
    assert analysis.all('x') == ['r1', 'r2', 'r3', 'r4', 'r5']


# ------------------------------------------------------- FullTextAnalysis

def test_fulltextanalysis_scores_top_and_all():
    db = _make_corpus()
    analysis = db.textAnalysis('docs')
    assert sorted(analysis.terms()) == ['apple', 'banana']
    assert analysis.scores('apple') == [3, 2, 1]
    assert analysis.scores('banana') == [3, 1]
    assert analysis.top('apple') == 'r3'
    assert analysis.top('banana') == 'r2'
    assert sorted(analysis.all('apple')) == ['r1', 'r2', 'r3', 'r4']
    assert sorted(analysis.all('banana')) == ['r1', 'r2']


def test_fulltextanalysis_missing_term_behaviour():
    analysis = FullTextAnalysis({'foo': {'sum': 1, 'spread': {1: {'value': 'a'}}}})
    assert analysis.scores('bar') is None
    assert analysis.top('bar') is None
    with pytest.raises(KeyError):
        analysis.all('bar')


def test_fulltextanalysis_all_is_repeatable():
    data = {
        'foo': {
            'sum': 4,
            'spread': {
                2: {'value': 'a'},
                1: {'value': 'b', 'next': {'value': 'c'}},
            },
        }
    }
    analysis = FullTextAnalysis(data)
    first = analysis.all('foo')
    second = analysis.all('foo')
    assert first == ['a', 'b', 'c']
    assert second == first


def test_fulltextanalysis_filter():
    db = _make_corpus()
    analysis = db.textAnalysis('docs')
    only_a = analysis.filter(lambda t: t.startswith('a'))
    assert list(only_a.keys()) == ['apple']
    assert only_a['apple'] is analysis.dump()['apple']
    assert analysis.filter(lambda t: False) == {}
    assert analysis.filter('not callable') == []
    both = analysis.filter(lambda t: True)
    assert sorted(both.keys()) == ['apple', 'banana']


def test_fulltextanalysis_serialize():
    db = _make_corpus()
    analysis = db.textAnalysis('docs')
    assert analysis.serialize(None) is None
    apple_ids = analysis.serialize(analysis.filter(lambda t: t == 'apple'))
    # highest score first, chained rows follow their head row
    assert apple_ids == ['r3', 'r1', 'r2', 'r4']
    banana_ids = analysis.serialize(analysis.filter(lambda t: t == 'banana'))
    assert banana_ids == ['r2', 'r1']
    all_ids = analysis.serialize(analysis.filter(lambda t: True))
    assert sorted(all_ids) == ['r1', 'r2', 'r3', 'r4']
    assert len(all_ids) == len(set(all_ids))


def test_fulltextanalysis_highlight():
    analysis = FullTextAnalysis({})
    out = analysis.highlight('cat', 'the cat sat on the concat cat', '<b>{{_}}</b>')
    assert out == 'the <b>cat</b> sat on the concat <b>cat</b>'
    assert analysis.highlight('dog', 'the cat', '<b>{{_}}</b>') == 'the cat'
