import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # package-style layout
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


def _sample_analysis_data():
    return {
        'hello': {
            'sum': 3,
            'spread': {2: {'value': 'a'}, 1: {'value': 'b'}},
        },
        'world': {
            'sum': 1,
            'spread': {1: {'value': 'a'}},
        },
    }


# ---------------------------------------------------------------- textIndex

def test_text_index_registers_fields_for_partition(db):
    db.textIndex('docs', {'title': 1, 'body': 1})
    assert set(db.all_indices['docs'].keys()) == {'title', 'body'}


def test_text_index_merges_on_repeated_calls_and_overrides(db):
    db.textIndex('docs', {'title': 1})
    db.textIndex('docs', {'body': {'weight': 2}, 'title': {'weight': 5}})
    assert db.all_indices['docs']['body'] == {'weight': 2}
    assert db.all_indices['docs']['title'] == {'weight': 5}


def test_text_index_partitions_are_independent(db):
    db.textIndex('a', {'title': 1})
    db.textIndex('b', {'body': 1})
    assert 'body' not in db.all_indices['a']
    assert 'title' not in db.all_indices['b']


# ------------------------------------------------------------------- insert

def test_insert_sets_keys_and_timestamp_and_appends_copy(db):
    item = {'name': 'x'}
    result = db.insert('p', 'r1', item)
    assert result is None
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert isinstance(item['timestamp'], float)
    assert len(db.table) == 1
    stored = db.table[0]
    assert stored is not item
    assert stored['name'] == 'x'
    assert stored['row_key'] == 'r1'


def test_insert_without_index_has_no_index_data(db):
    db.insert('p', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_into_unindexed_partition_has_no_index_data(db):
    db.textIndex('indexed', {'title': 1})
    db.insert('other', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_counts_term_frequency_sorted_descending(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'hello world hello'})
    index_data = db.table[0]['index_data']
    assert index_data[0] == ('hello', 2)
    assert dict(index_data) == {'hello': 2, 'world': 1}


def test_insert_applies_field_weight_and_sums_across_fields(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'cat dog', 'body': 'cat'})
    index_data = db.table[0]['index_data']
    assert index_data == [('cat', 4), ('dog', 3)]


def test_insert_ignores_non_indexed_fields(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha', 'body': 'beta beta'})
    terms = dict(db.table[0]['index_data'])
    assert terms == {'alpha': 1}


def test_insert_strips_whitespace_before_tokenizing(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': '  hello  '})
    assert db.table[0]['index_data'] == [('hello', 1)]


def test_insert_with_no_indexed_fields_present_gives_empty_index_data(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'other': 'value'})
    assert db.table[0]['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_when_nothing_indexed(db):
    db.insert('p', 'r1', {'title': 'hello'})
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('missing') is None


def test_text_analysis_builds_sum_and_spread(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'a', {'title': 'hello world hello'})
    db.insert('p', 'b', {'title': 'hello'})
    analysis = db.textAnalysis('p')
    assert type(analysis).__name__ == 'FullTextAnalysis'
    data = analysis.dump()
    assert set(data.keys()) == {'hello', 'world'}
    assert data['hello']['sum'] == 3
    assert data['world']['sum'] == 1
    assert data['hello']['spread'][2]['value'] == 'a'
    assert data['hello']['spread'][1]['value'] == 'b'
    assert data['world']['spread'][1]['value'] == 'a'


def test_text_analysis_chains_rows_with_same_score(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'a', {'title': 'cat'})
    db.insert('p', 'b', {'title': 'cat'})
    db.insert('p', 'c', {'title': 'cat'})
    data = db.textAnalysis('p').dump()
    assert data['cat']['sum'] == 3
    node = data['cat']['spread'][1]
    assert node['value'] == 'a'
    assert node['next']['value'] == 'b'
    assert node['next']['next']['value'] == 'c'


def test_text_analysis_only_includes_requested_partition(db):
    db.textIndex('p', {'title': 1})
    db.textIndex('q', {'title': 1})
    db.insert('p', 'a', {'title': 'apple'})
    db.insert('q', 'b', {'title': 'banana'})
    assert db.textAnalysis('p').terms() == ['apple']
    assert db.textAnalysis('q').terms() == ['banana']


# --------------------------------------------------------- FullTextAnalysis

def test_full_text_analysis_dump_and_terms():
    data = _sample_analysis_data()
    fta = FullTextAnalysis(data)
    assert fta.dump() is data
    assert sorted(fta.terms()) == ['hello', 'world']


def test_full_text_analysis_scores_and_top():
    fta = FullTextAnalysis(_sample_analysis_data())
    assert fta.scores('hello') == [2, 1]
    assert fta.top('hello') == 'a'
    assert fta.scores('missing') is None
    assert fta.top('missing') is None


def test_full_text_analysis_all_follows_chains():
    data = {
        'cat': {
            'sum': 3,
            'spread': {1: {'value': 'a', 'next': {'value': 'b', 'next': {'value': 'c'}}}},
        },
        'dog': {
            'sum': 5,
            'spread': {5: {'value': 'z'}},
        },
    }
    fta = FullTextAnalysis(data)
    assert fta.all('cat') == ['a', 'b', 'c']
    assert fta.all('dog') == ['z']
    with pytest.raises(KeyError):
        fta.all('missing')


def test_full_text_analysis_filter_and_serialize():
    fta = FullTextAnalysis(_sample_analysis_data())
    assert fta.filter('not callable') == []
    assert fta.filter(lambda t: t == 'nothing') == {}

    filtered = fta.filter(lambda t: t.startswith('h'))
    assert list(filtered.keys()) == ['hello']
    assert filtered['hello']['sum'] == 3

    everything = fta.filter(lambda t: True)
    assert set(everything.keys()) == {'hello', 'world'}
    assert fta.serialize(everything) == ['a', 'b']
    assert fta.serialize(None) is None
    assert fta.serialize({}) == []


def test_full_text_analysis_highlight_whole_words_only():
    fta = FullTextAnalysis(_sample_analysis_data())
    result = fta.highlight('hello', 'hello world, hello there hellooo', '<b>{{_}}</b>')
    assert result == '<b>hello</b> world, <b>hello</b> there hellooo'
    assert fta.highlight('cat', 'a dog', '[{{_}}]') == 'a dog'
