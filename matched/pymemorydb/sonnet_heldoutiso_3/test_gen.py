import time

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
def indexed_db():
    d = MemoryDb()
    d.textIndex('p', {'text': 1})
    d.insert('p', 'r1', {'text': 'apple banana'})
    d.insert('p', 'r2', {'text': 'apple'})
    d.insert('p', 'r3', {'text': 'apple apple'})
    return d


# ---------------------------------------------------------------- textIndex

def test_text_index_creates_partition_entry(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    assert db.all_indices['p'] == {'title': {'weight': 3}, 'body': 1}


def test_text_index_merges_and_overrides_existing_keys(db):
    db.textIndex('p', {'title': 1, 'body': 1})
    db.textIndex('p', {'body': {'weight': 5}, 'tags': 1})
    assert db.all_indices['p'] == {'title': 1, 'body': {'weight': 5}, 'tags': 1}


def test_text_index_is_per_partition(db):
    db.textIndex('a', {'x': 1})
    db.textIndex('b', {'y': 1})
    assert set(db.all_indices.keys()) == {'a', 'b'}
    assert db.all_indices['a'] == {'x': 1}
    assert db.all_indices['b'] == {'y': 1}


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp(db):
    before = time.time()
    db.insert('p', 'r1', {'name': 'x'})
    after = time.time()
    row = db.dumpdb()[0]
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['name'] == 'x'
    assert before <= row['timestamp'] <= after


def test_insert_without_index_has_no_index_data(db):
    db.insert('p', 'r1', {'text': 'hello world'})
    assert 'index_data' not in db.dumpdb()[0]


def test_insert_stores_a_copy_of_item(db):
    item = {'text': 'hello'}
    db.insert('p', 'r1', item)
    item['text'] = 'changed'
    assert db.dumpdb()[0]['text'] == 'hello'


def test_insert_with_index_computes_weighted_index_data(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'hello world', 'body': 'hello hello'})
    row = db.get('p', 'r1')
    # title: hello=3, world=3 ; body: 'hello hello' -> 2 ; hello total 5
    assert row['index_data'] == [('hello', 5), ('world', 3)]


def test_insert_index_ignores_unindexed_fields(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha', 'body': 'beta gamma'})
    assert db.get('p', 'r1')['index_data'] == [('alpha', 1)]


def test_insert_index_counts_only_consecutive_repeats_per_group(db):
    db.textIndex('p', {'text': 1})
    db.insert('p', 'r1', {'text': '  a b a  '})
    data = dict(db.get('p', 'r1')['index_data'])
    assert data == {'a': 2, 'b': 1}


def test_insert_other_partition_not_indexed(db):
    db.textIndex('p', {'text': 1})
    db.insert('other', 'r1', {'text': 'hello'})
    assert 'index_data' not in db.get('other', 'r1')


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_for_empty_partition(db):
    assert db.textAnalysis('missing') is None


def test_text_analysis_returns_none_without_index_data(db):
    db.insert('p', 'r1', {'text': 'hello'})
    assert db.textAnalysis('p') is None


def test_text_analysis_returns_full_text_analysis(indexed_db):
    result = indexed_db.textAnalysis('p')
    assert type(result).__name__ == 'FullTextAnalysis'
    assert sorted(result.terms()) == ['apple', 'banana']


def test_text_analysis_sums_and_spread(indexed_db):
    data = indexed_db.textAnalysis('p').dump()
    assert data['apple']['sum'] == 4
    assert data['banana']['sum'] == 1
    spread = data['apple']['spread']
    assert set(spread.keys()) == {1, 2}
    assert spread[1]['value'] == 'r1'
    assert spread[1]['next']['value'] == 'r2'
    assert spread[2]['value'] == 'r3'


def test_text_analysis_only_includes_requested_partition(indexed_db):
    indexed_db.textIndex('q', {'text': 1})
    indexed_db.insert('q', 'x1', {'text': 'zebra'})
    assert indexed_db.textAnalysis('q').terms() == ['zebra']
    assert 'zebra' not in indexed_db.textAnalysis('p').terms()


# --------------------------------------------------------- FullTextAnalysis

def test_scores_sorted_descending_and_unknown_is_none(indexed_db):
    fta = indexed_db.textAnalysis('p')
    assert fta.scores('apple') == [2, 1]
    assert fta.scores('nothing') is None


def test_top_returns_highest_scoring_row_key(indexed_db):
    fta = indexed_db.textAnalysis('p')
    assert fta.top('apple') == 'r3'
    assert fta.top('banana') == 'r1'
    assert fta.top('nothing') is None


def test_all_returns_every_row_key_and_is_repeatable(indexed_db):
    fta = indexed_db.textAnalysis('p')
    assert fta.all('apple') == ['r1', 'r2', 'r3']
    assert fta.all('apple') == ['r1', 'r2', 'r3']
    assert fta.all('banana') == ['r1']


def test_filter_with_expression_and_non_callable(indexed_db):
    fta = indexed_db.textAnalysis('p')
    result = fta.filter(lambda t: t.startswith('a'))
    assert list(result.keys()) == ['apple']
    assert result['apple'] is fta.dump()['apple']
    assert fta.filter('not callable') == []
    assert fta.filter(lambda t: False) == {}


def test_serialize_orders_ids_by_score_without_duplicates(indexed_db):
    fta = indexed_db.textAnalysis('p')
    assert fta.serialize(fta.filter(lambda t: True)) == ['r3', 'r1', 'r2']
    assert fta.serialize(None) is None


def test_highlight_wraps_whole_word_only():
    fta = FullTextAnalysis({})
    out = fta.highlight('apple', 'I like apple pie and pineapple', '<b>{{_}}</b>')
    assert out == 'I like <b>apple</b> pie and pineapple'


def test_terms_and_dump_on_handbuilt_data():
    data = {'x': {'sum': 3, 'spread': {3: {'value': 'r9'}}}}
    fta = FullTextAnalysis(data)
    assert fta.dump() is data
    assert fta.terms() == ['x']
    assert fta.top('x') == 'r9'
