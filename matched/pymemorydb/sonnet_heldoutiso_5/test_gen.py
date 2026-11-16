import pytest

try:
    import memorydb as memorydb_module
except ImportError:  # pragma: no cover
    from pymemorydb import memorydb as memorydb_module

MemoryDb = memorydb_module.MemoryDb
text_analysis = memorydb_module.text_analysis
FullTextAnalysis = text_analysis.FullTextAnalysis


def make_db():
    """Database with partition 'p' indexed on 'title' (weight 3) and 'body'."""
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    return db


def make_simple_analysis():
    """r1 -> 'apple banana', r2 -> 'apple', all with weight 1."""
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 1}})
    db.insert('p', 'r1', {'title': 'apple banana'})
    db.insert('p', 'r2', {'title': 'apple'})
    return db.textAnalysis('p')


# ---------------------------------------------------------------- textIndex

def test_textIndex_registers_fields_for_partition():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 2}, 'body': 1})
    assert db.all_indices['p'] == {'title': {'weight': 2}, 'body': 1}


def test_textIndex_merges_and_overrides_on_repeated_calls():
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 2}})
    db.textIndex('p', {'body': 1, 'title': {'weight': 5}})
    assert db.all_indices['p'] == {'title': {'weight': 5}, 'body': 1}


def test_textIndex_partitions_are_independent():
    db = MemoryDb()
    db.textIndex('a', {'title': 1})
    db.textIndex('b', {'body': 1})
    assert set(db.all_indices.keys()) == {'a', 'b'}
    assert 'body' not in db.all_indices['a']
    assert 'title' not in db.all_indices['b']


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp_without_index():
    db = MemoryDb()
    item = {'title': 'hello'}
    db.insert('p', 'r1', item)
    assert len(db.table) == 1
    stored = db.table[0]
    assert stored['partition_key'] == 'p'
    assert stored['row_key'] == 'r1'
    assert isinstance(stored['timestamp'], float)
    assert stored['title'] == 'hello'
    assert 'index_data' not in stored


def test_insert_stores_a_copy_of_the_item():
    db = MemoryDb()
    item = {'title': 'hello'}
    db.insert('p', 'r1', item)
    assert db.table[0] is not item
    item['extra'] = 1
    assert 'extra' not in db.table[0]


def test_insert_does_not_index_other_partitions():
    db = make_db()
    db.insert('other', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_builds_weighted_index_data_sorted_descending():
    db = make_db()
    db.insert('p', 'r1', {'title': 'hello world', 'body': 'hello'})
    index_data = db.table[0]['index_data']
    # hello: 3 (title) + 1 (body) = 4, world: 3
    assert list(index_data) == [('hello', 4), ('world', 3)]


def test_insert_counts_repeated_words():
    db = MemoryDb()
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'a b a a'})
    assert dict(db.table[0]['index_data']) == {'a': 3, 'b': 1}
    assert db.table[0]['index_data'][0] == ('a', 3)


def test_insert_non_dict_index_description_uses_weight_one():
    db = MemoryDb()
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'x x y'})
    assert dict(db.table[0]['index_data']) == {'x': 2, 'y': 1}


def test_insert_ignores_unindexed_fields():
    db = MemoryDb()
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha', 'notes': 'beta gamma'})
    assert dict(db.table[0]['index_data']) == {'alpha': 1}


def test_insert_with_no_matching_fields_gives_empty_index_data():
    db = MemoryDb()
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'other': 'zzz'})
    assert db.table[0]['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_textAnalysis_returns_none_when_nothing_indexed():
    db = MemoryDb()
    db.insert('p', 'r1', {'title': 'hello'})
    assert db.textAnalysis('p') is None


def test_textAnalysis_returns_none_for_empty_index_data():
    db = MemoryDb()
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'other': 'zzz'})
    assert db.textAnalysis('p') is None


def test_textAnalysis_returns_fulltextanalysis_with_sums_and_spread():
    analysis = make_simple_analysis()
    assert isinstance(analysis, FullTextAnalysis)
    data = analysis.dump()
    assert set(data.keys()) == {'apple', 'banana'}
    assert data['apple']['sum'] == 2
    assert data['banana']['sum'] == 1
    assert data['apple']['spread'] == {1: {'value': 'r1', 'next': {'value': 'r2'}}}
    assert data['banana']['spread'] == {1: {'value': 'r1'}}


def test_textAnalysis_only_considers_requested_partition():
    db = MemoryDb()
    db.textIndex('p', {'title': 1})
    db.textIndex('q', {'title': 1})
    db.insert('p', 'r1', {'title': 'apple'})
    db.insert('q', 'r2', {'title': 'pear'})
    analysis = db.textAnalysis('p')
    assert analysis.terms() == ['apple']


def test_textAnalysis_sums_different_weights_for_same_term():
    db = make_db()
    db.insert('p', 'r1', {'title': 'apple'})  # weight 3
    db.insert('p', 'r2', {'body': 'apple'})   # weight 1
    analysis = db.textAnalysis('p')
    assert analysis.dump()['apple']['sum'] == 4
    assert analysis.scores('apple') == [3, 1]
    assert analysis.top('apple') == 'r1'


# --------------------------------------------------------- FullTextAnalysis

def test_fulltext_terms_scores_and_unknown_term():
    analysis = make_simple_analysis()
    assert sorted(analysis.terms()) == ['apple', 'banana']
    assert analysis.scores('apple') == [1]
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None


def test_fulltext_all_returns_every_row_key_for_term():
    analysis = make_simple_analysis()
    assert analysis.all('apple') == ['r1', 'r2']
    assert analysis.all('banana') == ['r1']
    with pytest.raises(KeyError):
        analysis.all('missing')


def test_fulltext_filter_and_serialize():
    analysis = make_simple_analysis()
    assert analysis.filter('not callable') == []
    filtered = analysis.filter(lambda term: term.startswith('b'))
    assert list(filtered.keys()) == ['banana']
    assert analysis.filter(lambda term: term == 'nothing') == {}
    assert analysis.serialize(None) is None
    assert analysis.serialize(filtered) == ['r1']
    everything = analysis.filter(lambda term: True)
    assert sorted(analysis.serialize(everything)) == ['r1', 'r2']
    assert analysis.serialize({}) == []


def test_fulltext_highlight_replaces_whole_words_only():
    analysis = FullTextAnalysis({})
    result = analysis.highlight('apple', 'I like apple and pineapple', '<b>{{_}}</b>')
    assert result == 'I like <b>apple</b> and pineapple'
