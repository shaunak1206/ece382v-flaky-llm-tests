import pytest

try:
    from memorydb import MemoryDb
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb

try:
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.text_analysis import FullTextAnalysis


def build_db():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'a', {'body': 'cat cat dog'})
    db.insert('p', 'b', {'body': 'cat'})
    db.insert('p', 'c', {'body': 'cat dog'})
    return db


# ---------------------------------------------------------------- textIndex

def test_text_index_registers_fields_for_partition():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    indices = db.dumpindices('p')
    assert indices == {'title': {'weight': 3}, 'body': 1}


def test_text_index_merges_and_overwrites_on_repeat_calls():
    db = MemoryDb()
    db.textIndex('p', {'title': 1})
    db.textIndex('p', {'body': 2, 'title': 5})
    assert db.dumpindices('p') == {'title': 5, 'body': 2}


def test_text_index_partitions_are_independent():
    db = MemoryDb()
    db.textIndex('p1', {'a': 1})
    db.textIndex('p2', {'b': 1})
    assert db.dumpindices('p1') == {'a': 1}
    assert db.dumpindices('p2') == {'b': 1}
    assert set(db.dumpindices().keys()) == {'p1', 'p2'}


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp_and_stores_row():
    db = MemoryDb()
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert isinstance(item['timestamp'], float)
    stored = db.get('p', 'r1')
    assert stored['name'] == 'x'
    assert stored['timestamp'] == item['timestamp']
    assert len(db.dumpdb()) == 1


def test_insert_without_index_has_no_index_data():
    db = MemoryDb()
    db.insert('p', 'r1', {'body': 'hello world'})
    assert 'index_data' not in db.get('p', 'r1')


def test_insert_stores_a_copy_of_item():
    db = MemoryDb()
    item = {'body': 'x'}
    db.insert('p', 'r1', item)
    assert db.dumpdb()[0] is not item


def test_insert_with_index_counts_and_sorts_terms():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'a', {'body': 'hello world hello'})
    data = db.get('p', 'a')['index_data']
    assert data == [('hello', 2), ('world', 1)]


def test_insert_index_groups_consecutive_repeats():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'a', {'body': 'go go go stop'})
    data = dict(db.get('p', 'a')['index_data'])
    assert data == {'go': 3, 'stop': 1}


def test_insert_applies_weight_and_combines_fields():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 5}, 'body': 1})
    db.insert('p', 'a', {'title': 'python', 'body': 'python rocks'})
    data = db.get('p', 'a')['index_data']
    assert data == [('python', 6), ('rocks', 1)]


def test_insert_ignores_unindexed_fields():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'a', {'body': 'alpha', 'other': 'beta gamma'})
    data = dict(db.get('p', 'a')['index_data'])
    assert data == {'alpha': 1}


def test_insert_with_index_but_no_matching_fields_gives_empty_index_data():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'a', {'other': 'text'})
    assert db.get('p', 'a')['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_without_indexed_items():
    db = MemoryDb()
    db.insert('p', 'a', {'body': 'hello'})
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('missing') is None


def test_text_analysis_builds_sum_and_spread():
    analysis = build_db().textAnalysis('p')
    assert type(analysis).__name__ == 'FullTextAnalysis'
    data = analysis.dump()
    assert set(data.keys()) == {'cat', 'dog'}
    assert data['cat']['sum'] == 4
    assert data['dog']['sum'] == 2
    assert set(data['cat']['spread'].keys()) == {1, 2}
    assert data['cat']['spread'][2]['value'] == 'a'
    assert data['cat']['spread'][1]['value'] == 'b'
    assert data['cat']['spread'][1]['next']['value'] == 'c'


def test_text_analysis_ignores_other_partitions():
    db = build_db()
    db.textIndex('q', {'body': 1})
    db.insert('q', 'z', {'body': 'zebra'})
    analysis = db.textAnalysis('p')
    assert 'zebra' not in analysis.terms()
    assert db.textAnalysis('q').terms() == ['zebra']


# --------------------------------------------------------- FullTextAnalysis

def test_full_text_analysis_terms_scores_top():
    analysis = build_db().textAnalysis('p')
    assert analysis.terms() == ['cat', 'dog']
    assert analysis.scores('cat') == [2, 1]
    assert analysis.scores('nope') is None
    assert analysis.top('cat') == 'a'
    assert analysis.top('nope') is None


def test_full_text_analysis_all_returns_every_row_key():
    analysis = build_db().textAnalysis('p')
    assert analysis.all('cat') == ['a', 'b', 'c']
    assert analysis.all('dog') == ['a', 'c']


def test_full_text_analysis_filter_and_serialize():
    analysis = build_db().textAnalysis('p')
    assert analysis.filter('not callable') == []
    assert analysis.filter(lambda t: False) == {}
    filtered = analysis.filter(lambda t: t.startswith('c'))
    assert list(filtered.keys()) == ['cat']
    assert analysis.serialize(filtered) == ['a', 'b', 'c']
    assert analysis.serialize(analysis.filter(lambda t: True)) == ['a', 'b', 'c']
    assert analysis.serialize(None) is None


def test_full_text_analysis_direct_construction_and_highlight():
    data = {'x': {'sum': 3, 'spread': {3: {'value': 'r1'}}}}
    analysis = FullTextAnalysis(data)
    assert analysis.dump() is data
    assert analysis.top('x') == 'r1'
    assert analysis.highlight('cat', 'the cat sat', '<b>{{_}}</b>') == 'the <b>cat</b> sat'
    # word boundaries are respected
    assert analysis.highlight('cat', 'concat cat', '[{{_}}]') == 'concat [cat]'
