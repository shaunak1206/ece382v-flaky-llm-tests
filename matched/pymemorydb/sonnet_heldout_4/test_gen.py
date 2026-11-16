import pytest

try:
    from memorydb import MemoryDb
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb

try:
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


@pytest.fixture
def indexed_db(db):
    db.textIndex('p', {'text': 1})
    db.insert('p', 'r1', {'text': 'apple banana apple'})
    db.insert('p', 'r2', {'text': 'apple'})
    db.insert('p', 'r3', {'text': 'apple apple'})
    return db


# ---------------------------------------------------------------- textIndex

def test_textindex_registers_fields_for_partition(db):
    db.textIndex('p', {'title': {'weight': 2}})
    assert db.dumpindices('p') == {'title': {'weight': 2}}
    assert 'p' in db.dumpindices()


def test_textindex_merges_and_overrides_fields(db):
    db.textIndex('p', {'title': 1, 'body': 1})
    db.textIndex('p', {'body': {'weight': 5}, 'tags': 1})
    assert db.dumpindices('p') == {'title': 1, 'body': {'weight': 5}, 'tags': 1}


def test_textindex_partitions_are_independent(db):
    db.textIndex('a', {'x': 1})
    db.textIndex('b', {'y': 1})
    assert db.dumpindices('a') == {'x': 1}
    assert db.dumpindices('b') == {'y': 1}
    assert db.dumpindices('missing') is None


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp_and_stores_copy(db):
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert isinstance(item['timestamp'], float)
    stored = db.dumpdb()
    assert len(stored) == 1
    assert stored[0] is not item
    assert stored[0]['name'] == 'x'
    assert stored[0]['row_key'] == 'r1'


def test_insert_without_index_has_no_index_data(db):
    db.insert('p', 'r1', {'text': 'hello world'})
    assert 'index_data' not in db.get('p', 'r1')


def test_insert_with_index_builds_sorted_index_data(db):
    db.textIndex('p', {'text': 1})
    db.insert('p', 'r1', {'text': 'a b b b c c'})
    assert db.get('p', 'r1')['index_data'] == [('b', 3), ('c', 2), ('a', 1)]


def test_insert_index_applies_weights_across_fields(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': {}})
    db.insert('p', 'r1', {'title': 'hello world', 'body': 'hello hello'})
    assert db.get('p', 'r1')['index_data'] == [('hello', 5), ('world', 3)]


def test_insert_ignores_unindexed_fields_and_other_partitions(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha', 'body': 'beta gamma'})
    db.insert('other', 'r2', {'title': 'alpha'})
    assert db.get('p', 'r1')['index_data'] == [('alpha', 1)]
    assert 'index_data' not in db.get('other', 'r2')


def test_insert_non_adjacent_repeats_are_accumulated(db):
    db.textIndex('p', {'text': 1})
    db.insert('p', 'r1', {'text': 'a b a'})
    data = dict(db.get('p', 'r1')['index_data'])
    assert data == {'a': 2, 'b': 1}


def test_insert_strips_surrounding_whitespace(db):
    db.textIndex('p', {'text': 1})
    db.insert('p', 'r1', {'text': '  hello world  '})
    assert dict(db.get('p', 'r1')['index_data']) == {'hello': 1, 'world': 1}


# ------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_when_nothing_indexed(db):
    db.insert('p', 'r1', {'text': 'hello'})
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('unknown') is None


def test_textanalysis_returns_analysis_with_terms(indexed_db):
    analysis = indexed_db.textAnalysis('p')
    assert type(analysis).__name__ == 'FullTextAnalysis'
    assert sorted(analysis.terms()) == ['apple', 'banana']


def test_textanalysis_sums_and_spread(indexed_db):
    data = indexed_db.textAnalysis('p').dump()
    # apple: score 2 (r1), score 1 (r2), score 2 (r3) -> 2 + 2 + 1
    assert data['apple']['sum'] == 5
    assert data['banana']['sum'] == 1
    spread = data['apple']['spread']
    assert set(spread.keys()) == {1, 2}
    assert spread[2]['value'] == 'r1'
    assert spread[2]['next']['value'] == 'r3'
    assert spread[1]['value'] == 'r2'


def test_textanalysis_only_includes_requested_partition(indexed_db):
    indexed_db.textIndex('q', {'text': 1})
    indexed_db.insert('q', 'x1', {'text': 'zebra'})
    assert sorted(indexed_db.textAnalysis('p').terms()) == ['apple', 'banana']
    assert indexed_db.textAnalysis('q').terms() == ['zebra']


# --------------------------------------------------------- FullTextAnalysis

def test_fulltext_scores_and_top(indexed_db):
    analysis = indexed_db.textAnalysis('p')
    assert analysis.scores('apple') == [2, 1]
    assert analysis.top('apple') == 'r1'
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None


def test_fulltext_all_returns_every_row_key(indexed_db):
    analysis = indexed_db.textAnalysis('p')
    assert analysis.all('apple') == ['r1', 'r3', 'r2']
    assert analysis.all('apple') == ['r1', 'r3', 'r2']  # repeatable
    assert analysis.all('banana') == ['r1']
    with pytest.raises(KeyError):
        analysis.all('missing')


def test_fulltext_filter_and_serialize(indexed_db):
    analysis = indexed_db.textAnalysis('p')
    assert analysis.filter('not callable') == []
    assert analysis.filter(lambda t: False) == {}

    only_banana = analysis.filter(lambda t: t == 'banana')
    assert list(only_banana.keys()) == ['banana']
    assert only_banana['banana']['sum'] == 1

    everything = analysis.filter(lambda t: True)
    assert set(everything.keys()) == {'apple', 'banana'}
    assert analysis.serialize(everything) == ['r1', 'r3', 'r2']
    assert analysis.serialize(None) is None


def test_fulltext_basic_accessors_with_handmade_data():
    data = {
        'cat': {'sum': 3, 'spread': {3: {'value': 'a'}}},
        'dog': {'sum': 2, 'spread': {1: {'value': 'b'}, 2: {'value': 'c'}}},
    }
    analysis = FullTextAnalysis(data)
    assert analysis.dump() is data
    assert sorted(analysis.terms()) == ['cat', 'dog']
    assert analysis.scores('dog') == [2, 1]
    assert analysis.top('dog') == 'c'


def test_fulltext_highlight_wraps_whole_word_matches_only():
    analysis = FullTextAnalysis({})
    result = analysis.highlight('apple', 'I like apple and pineapple', '<b>{{_}}</b>')
    assert result == 'I like <b>apple</b> and pineapple'
