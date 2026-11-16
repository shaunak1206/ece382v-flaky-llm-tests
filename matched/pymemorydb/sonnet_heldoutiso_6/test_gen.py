import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


@pytest.fixture
def indexed_db():
    d = MemoryDb()
    d.textIndex('p', {'title': 1})
    return d


# ---------------------------------------------------------------- textIndex

def test_textindex_registers_fields_for_partition(db):
    db.textIndex('p', {'title': 1, 'body': {'weight': 3}})
    indices = db.dumpindices('p')
    assert indices == {'title': 1, 'body': {'weight': 3}}


def test_textindex_merges_and_overwrites_on_repeat_calls(db):
    db.textIndex('p', {'title': 1})
    db.textIndex('p', {'body': 2, 'title': {'weight': 5}})
    assert db.dumpindices('p') == {'title': {'weight': 5}, 'body': 2}


def test_textindex_partitions_are_independent(db):
    db.textIndex('a', {'x': 1})
    db.textIndex('b', {'y': 1})
    assert set(db.dumpindices().keys()) == {'a', 'b'}
    assert 'y' not in db.dumpindices('a')
    assert 'x' not in db.dumpindices('b')


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp(db):
    item = {'title': 'hello'}
    db.insert('p', 'r1', item)
    stored = db.get('p', 'r1')
    assert stored['partition_key'] == 'p'
    assert stored['row_key'] == 'r1'
    assert stored['title'] == 'hello'
    assert isinstance(stored['timestamp'], float)


def test_insert_without_index_has_no_index_data(db):
    db.insert('p', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.get('p', 'r1')


def test_insert_in_unindexed_partition_ignores_other_partition_index(db):
    db.textIndex('indexed', {'title': 1})
    db.insert('other', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.get('other', 'r1')


def test_insert_with_index_counts_words(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat dog cat'})
    data = indexed_db.get('p', 'r1')['index_data']
    # groupby only groups consecutive words, but counts are summed per term
    assert dict(data) == {'cat': 2, 'dog': 1}
    # sorted by score descending
    assert data[0] == ('cat', 2)


def test_insert_groups_consecutive_repeated_words(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'a a a b'})
    assert dict(indexed_db.get('p', 'r1')['index_data']) == {'a': 3, 'b': 1}


def test_insert_applies_field_weights(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'cat', 'body': 'cat dog'})
    data = db.get('p', 'r1')['index_data']
    assert data == [('cat', 4), ('dog', 1)]


def test_insert_strips_whitespace_and_skips_unindexed_fields(indexed_db):
    indexed_db.insert('p', 'r1', {'title': '  hello  ', 'other': 'ignored words'})
    data = indexed_db.get('p', 'r1')['index_data']
    assert data == [('hello', 1)]


def test_insert_indexed_partition_without_matching_fields_gives_empty_index(indexed_db):
    indexed_db.insert('p', 'r1', {'other': 'nothing'})
    assert indexed_db.get('p', 'r1')['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_when_no_indexed_items(db):
    assert db.textAnalysis('p') is None
    db.insert('p', 'r1', {'title': 'hello'})  # no index defined
    assert db.textAnalysis('p') is None


def test_textanalysis_returns_none_when_index_data_empty(indexed_db):
    indexed_db.insert('p', 'r1', {'other': 'nothing'})
    assert indexed_db.textAnalysis('p') is None


def test_textanalysis_builds_terms_sums_and_spread(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat dog'})
    indexed_db.insert('p', 'r2', {'title': 'cat'})
    analysis = indexed_db.textAnalysis('p')
    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ['cat', 'dog']
    dumped = analysis.dump()
    assert dumped['cat']['sum'] == 2
    assert dumped['dog']['sum'] == 1
    assert dumped['cat']['spread'][1]['value'] == 'r1'
    assert dumped['cat']['spread'][1]['next']['value'] == 'r2'


def test_textanalysis_ignores_other_partitions(db):
    db.textIndex('p', {'title': 1})
    db.textIndex('q', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha'})
    db.insert('q', 'r2', {'title': 'beta'})
    analysis = db.textAnalysis('p')
    assert analysis.terms() == ['alpha']


def test_textanalysis_scores_top_and_all(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat'})
    indexed_db.insert('p', 'r2', {'title': 'cat cat'})
    indexed_db.insert('p', 'r3', {'title': 'cat'})
    analysis = indexed_db.textAnalysis('p')
    assert analysis.scores('cat') == [2, 1]
    assert analysis.top('cat') == 'r2'
    assert sorted(analysis.all('cat')) == ['r1', 'r2', 'r3']
    assert analysis.dump()['cat']['sum'] == 4  # 2 + 1 + 1


def test_textanalysis_unknown_term_lookups(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat'})
    analysis = indexed_db.textAnalysis('p')
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None
    with pytest.raises(KeyError):
        analysis.all('missing')


# ----------------------------------------------------------- FullTextAnalysis

def test_fulltextanalysis_dump_and_terms_on_raw_data():
    data = {'a': {'sum': 1, 'spread': {1: {'value': 'r1'}}},
            'b': {'sum': 2, 'spread': {2: {'value': 'r2'}}}}
    fta = FullTextAnalysis(data)
    assert fta.dump() is data
    assert fta.terms() == ['a', 'b']
    assert fta.scores('b') == [2]
    assert fta.top('b') == 'r2'


def test_filter_and_serialize(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat'})
    indexed_db.insert('p', 'r2', {'title': 'cat cat'})
    indexed_db.insert('p', 'r3', {'title': 'dog'})
    analysis = indexed_db.textAnalysis('p')

    filtered = analysis.filter(lambda t: t == 'cat')
    assert list(filtered.keys()) == ['cat']
    assert filtered['cat'] is analysis.dump()['cat']
    # higher score first
    assert analysis.serialize(filtered) == ['r2', 'r1']

    everything = analysis.filter(lambda t: True)
    ids = analysis.serialize(everything)
    assert sorted(ids) == ['r1', 'r2', 'r3']
    assert len(ids) == len(set(ids))


def test_filter_non_callable_and_no_match(indexed_db):
    indexed_db.insert('p', 'r1', {'title': 'cat'})
    analysis = indexed_db.textAnalysis('p')
    assert analysis.filter('not callable') == []
    assert analysis.filter(lambda t: False) == {}
    assert analysis.serialize(None) is None


def test_highlight_wraps_whole_word_matches_only():
    fta = FullTextAnalysis({})
    result = fta.highlight('cat', 'the cat sat on concat', '<b>{{_}}</b>')
    assert result == 'the <b>cat</b> sat on concat'
    assert fta.highlight('cat', 'cat cat', '[{{_}}]') == '[cat] [cat]'
