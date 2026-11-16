import random

import pytest

try:
    import memorydb
except ImportError:  # pragma: no cover
    from pymemorydb import memorydb

MemoryDb = memorydb.MemoryDb
FullTextAnalysis = memorydb.text_analysis.FullTextAnalysis


def build_basic_db():
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.insert('p', 'r1', {'text': 'apple apple banana'})
    db.insert('p', 'r2', {'text': 'apple'})
    db.insert('p', 'r3', {'text': 'banana banana'})
    return db


def test_textindex_registers_descriptor_for_partition():
    random.seed(1234)
    db = MemoryDb()
    desc = {'title': {'weight': 2}, 'body': True}
    db.textIndex('p', desc)
    assert db.dumpindices('p') == desc
    assert set(db.dumpindices().keys()) == {'p'}


def test_textindex_merges_and_overrides_existing_descriptor():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'title': True, 'body': True})
    db.textIndex('p', {'title': {'weight': 5}, 'extra': True})
    assert db.dumpindices('p') == {
        'title': {'weight': 5},
        'body': True,
        'extra': True,
    }


def test_insert_sets_keys_and_timestamp(monkeypatch):
    random.seed(1234)
    monkeypatch.setattr(memorydb, 'time', lambda: 1234.5)
    db = MemoryDb()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert item['timestamp'] == pytest.approx(1234.5)
    stored = db.get('p', 'r1')
    assert stored['name'] == 'x'
    assert stored['timestamp'] == pytest.approx(1234.5)


def test_insert_stores_shallow_copy_of_item():
    random.seed(1234)
    db = MemoryDb()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    stored = db.dumpdb()[0]
    assert stored is not item
    item['later'] = 1
    assert 'later' not in stored
    assert len(db.dumpdb()) == 1


def test_insert_without_index_adds_no_index_data():
    random.seed(1234)
    db = MemoryDb()
    db.insert('p', 'r1', {'text': 'hello world'})
    assert 'index_data' not in db.get('p', 'r1')


def test_insert_index_data_counts_and_weights():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 2}, 'body': True})
    db.insert('p', 'r1', {'title': 'a a b', 'body': 'b c', 'other': 'zzz'})
    index_data = db.get('p', 'r1')['index_data']
    # a: 2*2, b: 1*2 + 1, c: 1 ; 'other' is not indexed
    assert index_data == [('a', 4), ('b', 3), ('c', 1)]


def test_insert_index_data_strips_whitespace_and_ignores_unindexed_fields():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.insert('p', 'r1', {'text': '  foo bar  ', 'note': 'ignored words'})
    index_data = db.get('p', 'r1')['index_data']
    assert dict(index_data) == {'foo': 1, 'bar': 1}
    assert len(index_data) == 2


def test_insert_index_data_empty_when_no_indexed_field_present():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.insert('p', 'r1', {'other': 'abc'})
    assert db.get('p', 'r1')['index_data'] == []


def test_insert_non_string_indexed_field_raises_and_does_not_store():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'n': True})
    with pytest.raises(AttributeError):
        db.insert('p', 'r1', {'n': 5})
    assert db.dumpdb() == []


def test_insert_index_applies_only_to_indexed_partition():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.insert('q', 'r1', {'text': 'hello'})
    db.insert('p', 'r1', {'text': 'hello'})
    assert 'index_data' not in db.get('q', 'r1')
    assert db.get('p', 'r1')['index_data'] == [('hello', 1)]


def test_textanalysis_returns_none_without_indexed_items():
    random.seed(1234)
    db = MemoryDb()
    assert db.textAnalysis('p') is None
    db.insert('p', 'r1', {'text': 'hello'})  # no index defined
    assert db.textAnalysis('p') is None
    db.textIndex('p', {'text': True})
    db.insert('p', 'r2', {'other': 'hello'})  # index_data is empty
    assert db.textAnalysis('p') is None


def test_textanalysis_returns_fulltextanalysis_with_terms_and_sums():
    random.seed(1234)
    db = build_basic_db()
    analysis = db.textAnalysis('p')
    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ['apple', 'banana']
    data = analysis.dump()
    assert data['apple']['sum'] == 3
    assert data['banana']['sum'] == 3


def test_textanalysis_spread_structure_by_score():
    random.seed(1234)
    db = build_basic_db()
    data = db.textAnalysis('p').dump()
    assert data['apple']['spread'] == {2: {'value': 'r1'}, 1: {'value': 'r2'}}
    assert data['banana']['spread'] == {1: {'value': 'r1'}, 2: {'value': 'r3'}}


def test_textanalysis_chains_rows_with_same_score():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.insert('p', 'r1', {'text': 'apple'})
    db.insert('p', 'r2', {'text': 'apple'})
    db.insert('p', 'r3', {'text': 'apple'})
    analysis = db.textAnalysis('p')
    data = analysis.dump()
    assert data['apple']['sum'] == 3
    assert list(data['apple']['spread'].keys()) == [1]
    head = data['apple']['spread'][1]
    assert head['value'] == 'r1'
    assert head['next']['value'] == 'r2'
    assert head['next']['next']['value'] == 'r3'
    assert analysis.all('apple') == ['r1', 'r2', 'r3']


def test_textanalysis_uses_field_weights():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}, 'body': True})
    db.insert('p', 'r1', {'title': 'cat', 'body': 'cat dog'})
    analysis = db.textAnalysis('p')
    assert analysis.scores('cat') == [4]
    assert analysis.scores('dog') == [1]
    assert analysis.dump()['cat']['sum'] == 4


def test_textanalysis_only_includes_requested_partition():
    random.seed(1234)
    db = MemoryDb()
    db.textIndex('p', {'text': True})
    db.textIndex('q', {'text': True})
    db.insert('p', 'r1', {'text': 'apple'})
    db.insert('q', 'r2', {'text': 'zebra'})
    assert db.textAnalysis('p').terms() == ['apple']
    assert db.textAnalysis('q').terms() == ['zebra']
    assert db.textAnalysis('missing') is None


def test_analysis_scores_and_top():
    random.seed(1234)
    analysis = build_basic_db().textAnalysis('p')
    assert analysis.scores('apple') == [2, 1]
    assert analysis.scores('banana') == [2, 1]
    assert analysis.top('apple') == 'r1'
    assert analysis.top('banana') == 'r3'
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None


def test_analysis_all_returns_every_row_and_is_repeatable():
    random.seed(1234)
    analysis = build_basic_db().textAnalysis('p')
    assert sorted(analysis.all('apple')) == ['r1', 'r2']
    assert sorted(analysis.all('banana')) == ['r1', 'r3']
    assert sorted(analysis.all('apple')) == ['r1', 'r2']
    with pytest.raises(KeyError):
        analysis.all('missing')


def test_analysis_filter_selects_terms():
    random.seed(1234)
    analysis = build_basic_db().textAnalysis('p')
    result = analysis.filter(lambda t: t.startswith('a'))
    assert list(result.keys()) == ['apple']
    assert result['apple'] is analysis.dump()['apple']
    assert analysis.filter(lambda t: False) == {}
    assert analysis.filter('not callable') == []


def test_analysis_serialize_returns_unique_row_ids_in_score_order():
    random.seed(1234)
    analysis = build_basic_db().textAnalysis('p')
    assert analysis.serialize(None) is None
    only_banana = analysis.filter(lambda t: t == 'banana')
    assert analysis.serialize(only_banana) == ['r3', 'r1']
    everything = analysis.filter(lambda t: True)
    ids = analysis.serialize(everything)
    assert len(ids) == 3
    assert sorted(ids) == ['r1', 'r2', 'r3']


def test_fulltextanalysis_direct_construction():
    random.seed(1234)
    data = {'x': {'sum': 5, 'spread': {5: {'value': 'a'}}}}
    analysis = FullTextAnalysis(data)
    assert analysis.dump() is data
    assert analysis.terms() == ['x']
    assert analysis.scores('x') == [5]
    assert analysis.top('x') == 'a'


def test_fulltextanalysis_highlight_wraps_whole_word_matches_only():
    random.seed(1234)
    analysis = FullTextAnalysis({})
    result = analysis.highlight('apple', 'an apple and pineapple', '<b>{{_}}</b>')
    assert result == 'an <b>apple</b> and pineapple'
