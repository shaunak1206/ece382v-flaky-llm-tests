import re

import pytest

try:
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb import text_analysis
except ImportError:
    from memorydb import MemoryDb
    import text_analysis


def make_db():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'apple apple banana'})
    db.insert('p', 'r2', {'body': 'apple'})
    db.insert('p', 'r3', {'body': 'banana banana banana'})
    return db


# ---------------------------------------------------------------- textIndex

def test_text_index_registers_fields_for_partition():
    db = MemoryDb()
    result = db.textIndex('p', {'body': 1, 'title': {'weight': 3}})
    assert result is None
    assert db.dumpindices('p') == {'body': 1, 'title': {'weight': 3}}


def test_text_index_merges_and_overwrites_fields():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.textIndex('p', {'title': 2, 'body': 5})
    assert db.dumpindices('p') == {'body': 5, 'title': 2}


def test_text_index_partitions_are_independent():
    db = MemoryDb()
    db.textIndex('a', {'x': 1})
    db.textIndex('b', {'y': 1})
    assert db.dumpindices() == {'a': {'x': 1}, 'b': {'y': 1}}


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp_and_stores_copy():
    db = MemoryDb()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert isinstance(item['timestamp'], float)
    assert len(db.table) == 1
    assert db.table[0] is not item
    assert db.table[0]['name'] == 'x'
    assert db.table[0]['row_key'] == 'r1'


def test_insert_without_text_index_has_no_index_data():
    db = MemoryDb()
    db.insert('p', 'r1', {'body': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_counts_and_sorts_terms_by_frequency():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'a b a'})
    # the 'a' occurrences are not consecutive, but both are counted
    assert db.table[0]['index_data'] == [('a', 2), ('b', 1)]


def test_insert_applies_weight_from_dict_description():
    db = MemoryDb()
    db.textIndex('p', {'body': {'weight': 3}})
    db.insert('p', 'r1', {'body': 'x y x'})
    assert db.table[0]['index_data'] == [('x', 6), ('y', 3)]


def test_insert_combines_multiple_indexed_fields():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'cat', 'body': 'cat dog'})
    assert db.table[0]['index_data'] == [('cat', 4), ('dog', 1)]


def test_insert_ignores_fields_that_are_not_indexed():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'alpha', 'other': 'beta gamma', 'n': 5})
    assert db.table[0]['index_data'] == [('alpha', 1)]


def test_insert_strips_surrounding_whitespace():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': '   hello world  '})
    terms = sorted(t for t, _ in db.table[0]['index_data'])
    assert terms == ['hello', 'world']


def test_insert_only_indexes_partitions_with_text_index():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('q', 'r1', {'body': 'hello'})
    db.insert('p', 'r2', {'body': 'hello'})
    by_key = {v['row_key']: v for v in db.table}
    assert 'index_data' not in by_key['r1']
    assert by_key['r2']['index_data'] == [('hello', 1)]


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_without_indexed_items():
    db = MemoryDb()
    db.insert('p', 'r1', {'body': 'hello'})
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('missing') is None


def test_text_analysis_skips_items_inserted_before_index():
    db = MemoryDb()
    db.insert('p', 'old', {'body': 'apple'})
    db.textIndex('p', {'body': 1})
    db.insert('p', 'new', {'body': 'apple'})
    analysis = db.textAnalysis('p')
    assert analysis.all('apple') == ['new']


def test_text_analysis_terms_sums_and_spread():
    analysis = make_db().textAnalysis('p')
    assert type(analysis).__name__ == 'FullTextAnalysis'
    assert sorted(analysis.terms()) == ['apple', 'banana']
    data = analysis.dump()
    assert data['apple']['sum'] == 3
    assert data['banana']['sum'] == 4
    assert data['apple']['spread'][2]['value'] == 'r1'
    assert data['apple']['spread'][1]['value'] == 'r2'
    assert data['banana']['spread'][3]['value'] == 'r3'


def test_text_analysis_chains_items_with_equal_counts():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    for rk in ('r1', 'r2', 'r3'):
        db.insert('p', rk, {'body': 'apple'})
    analysis = db.textAnalysis('p')
    assert analysis.dump()['apple']['sum'] == 3
    assert analysis.scores('apple') == [1]
    assert analysis.top('apple') == 'r1'
    assert analysis.all('apple') == ['r1', 'r2', 'r3']


def test_text_analysis_only_uses_requested_partition_and_reflects_delete():
    db = make_db()
    db.textIndex('other', {'body': 1})
    db.insert('other', 'z1', {'body': 'zebra'})
    analysis = db.textAnalysis('p')
    assert 'zebra' not in analysis.terms()
    db.delete('p', 'r2')
    analysis = db.textAnalysis('p')
    assert analysis.dump()['apple']['sum'] == 2
    assert analysis.scores('apple') == [2]


# --------------------------------------------------------- FullTextAnalysis

def test_scores_and_top():
    analysis = make_db().textAnalysis('p')
    assert analysis.scores('apple') == [2, 1]
    assert analysis.scores('banana') == [3, 1]
    assert analysis.top('apple') == 'r1'
    assert analysis.top('banana') == 'r3'
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None


def test_all_raises_for_unknown_term():
    analysis = make_db().textAnalysis('p')
    with pytest.raises(KeyError):
        analysis.all('missing')


def test_filter_and_serialize():
    analysis = make_db().textAnalysis('p')
    assert analysis.filter('not callable') == []
    assert analysis.filter(lambda t: t == 'nothing') == {}
    only_apple = analysis.filter(lambda t: t.startswith('a'))
    assert list(only_apple.keys()) == ['apple']
    assert only_apple['apple']['sum'] == 3
    everything = analysis.filter(lambda t: True)
    assert sorted(everything.keys()) == ['apple', 'banana']
    ids = analysis.serialize(everything)
    assert sorted(ids) == ['r1', 'r2', 'r3']
    assert len(ids) == len(set(ids))
    assert analysis.serialize(only_apple) == ['r1', 'r2']
    assert analysis.serialize(None) is None
    assert analysis.serialize({}) == []


def test_highlight_wraps_whole_word_matches():
    analysis = text_analysis.FullTextAnalysis({})
    out = analysis.highlight('apple', 'an apple a day', '<b>{{_}}</b>')
    assert out == 'an <b>apple</b> a day'
    assert analysis.highlight('apple', 'pineapple pie', '<b>{{_}}</b>') == 'pineapple pie'
    assert analysis.highlight('apple', 'apple and apple', '[{{_}}]') == '[apple] and [apple]'
    assert analysis.highlight('kiwi', 'an apple a day', '<b>{{_}}</b>') == 'an apple a day'
