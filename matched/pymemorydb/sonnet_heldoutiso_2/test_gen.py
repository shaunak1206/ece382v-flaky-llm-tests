import pytest

try:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis
except ImportError:  # pragma: no cover - package-style layout
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


@pytest.fixture
def sample_data():
    return {
        'apple': {
            'sum': 4,
            'spread': {
                2: {'value': 'a'},
                1: {'value': 'b', 'next': {'value': 'c'}},
            },
        },
        'banana': {
            'sum': 1,
            'spread': {1: {'value': 'd'}},
        },
    }


# ---------------------------------------------------------------- textIndex

def test_text_index_registers_fields_for_partition(db):
    db.textIndex('p', {'title': {'weight': 2}, 'body': 1})
    assert db.dumpindices('p') == {'title': {'weight': 2}, 'body': 1}


def test_text_index_merges_and_overrides_on_repeated_calls(db):
    db.textIndex('p', {'title': {'weight': 2}})
    db.textIndex('p', {'body': {'weight': 1}, 'title': {'weight': 5}})
    assert db.dumpindices('p') == {'title': {'weight': 5}, 'body': {'weight': 1}}


def test_text_index_partitions_are_independent(db):
    db.textIndex('p1', {'title': 1})
    db.textIndex('p2', {'body': 1})
    assert set(db.dumpindices('p1').keys()) == {'title'}
    assert set(db.dumpindices('p2').keys()) == {'body'}
    assert set(db.dumpindices().keys()) == {'p1', 'p2'}


# ------------------------------------------------------------------- insert

def test_insert_without_index_stores_basic_fields_only(db):
    db.insert('p', 'r1', {'title': 'hello world'})
    row = db.get('p', 'r1')
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['title'] == 'hello world'
    assert isinstance(row['timestamp'], float)
    assert 'index_data' not in row


def test_insert_mutates_passed_item_with_keys(db):
    item = {'title': 'x'}
    db.insert('p', 'r1', item)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert 'timestamp' in item


def test_insert_stores_shallow_copy(db):
    item = {'title': 'x'}
    db.insert('p', 'r1', item)
    item['title'] = 'changed'
    assert db.get('p', 'r1')['title'] == 'x'


def test_insert_with_index_builds_index_data_with_counts(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'hello world'})
    data = db.get('p', 'r1')['index_data']
    assert sorted(data) == [('hello', 1), ('world', 1)]


def test_insert_index_data_applies_weights_and_sorts_descending(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': {}})
    db.insert('p', 'r1', {'title': 'cat dog', 'body': 'dog'})
    data = db.get('p', 'r1')['index_data']
    assert data == [('dog', 4), ('cat', 3)]


def test_insert_counts_only_consecutive_repeats_per_run(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'a a b a'})
    data = dict(db.get('p', 'r1')['index_data'])
    assert data == {'a': 3, 'b': 1}


def test_insert_ignores_unindexed_fields(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'foo', 'other': 'bar baz'})
    data = dict(db.get('p', 'r1')['index_data'])
    assert data == {'foo': 1}


def test_insert_only_indexes_partition_with_index(db):
    db.textIndex('p1', {'title': 1})
    db.insert('p2', 'r1', {'title': 'foo'})
    assert 'index_data' not in db.get('p2', 'r1')


def test_insert_non_string_indexed_field_raises(db):
    db.textIndex('p', {'n': 1})
    with pytest.raises(AttributeError):
        db.insert('p', 'r1', {'n': 5})


# ------------------------------------------------------------- textAnalysis

def test_text_analysis_returns_none_when_no_indexed_items(db):
    assert db.textAnalysis('p') is None
    db.insert('p', 'r1', {'title': 'hello'})
    assert db.textAnalysis('p') is None


def test_text_analysis_skips_items_inserted_before_index(db):
    db.insert('p', 'old', {'title': 'legacy'})
    db.textIndex('p', {'title': 1})
    db.insert('p', 'new', {'title': 'fresh'})
    analysis = db.textAnalysis('p')
    assert isinstance(analysis, FullTextAnalysis)
    assert analysis.terms() == ['fresh']


def test_text_analysis_groups_rows_with_same_score_and_sums(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'apple banana'})
    db.insert('p', 'r2', {'title': 'apple'})
    analysis = db.textAnalysis('p')
    assert sorted(analysis.terms()) == ['apple', 'banana']
    assert analysis.dump()['apple']['sum'] == 2
    assert analysis.dump()['banana']['sum'] == 1
    assert analysis.scores('apple') == [1]
    assert analysis.top('apple') == 'r1'
    assert analysis.all('apple') == ['r1', 'r2']


def test_text_analysis_with_weights_ranks_rows(db):
    db.textIndex('p', {'title': {'weight': 5}, 'body': 1})
    db.insert('p', 'r1', {'title': 'python', 'body': 'python java'})
    db.insert('p', 'r2', {'title': 'java', 'body': 'x'})
    analysis = db.textAnalysis('p')
    assert analysis.scores('java') == [5, 1]
    assert analysis.top('java') == 'r2'
    assert analysis.top('python') == 'r1'
    assert analysis.dump()['java']['sum'] == 6


def test_text_analysis_excludes_other_partitions(db):
    db.textIndex('p1', {'title': 1})
    db.textIndex('p2', {'title': 1})
    db.insert('p1', 'r1', {'title': 'alpha'})
    db.insert('p2', 'r2', {'title': 'beta'})
    assert db.textAnalysis('p1').terms() == ['alpha']
    assert db.textAnalysis('p2').terms() == ['beta']


# --------------------------------------------------------- FullTextAnalysis

def test_full_text_analysis_dump_and_terms(sample_data):
    fta = FullTextAnalysis(sample_data)
    assert fta.dump() is sample_data
    assert fta.terms() == ['apple', 'banana']


def test_full_text_analysis_scores_and_top(sample_data):
    fta = FullTextAnalysis(sample_data)
    assert fta.scores('apple') == [2, 1]
    assert fta.scores('missing') is None
    assert fta.top('apple') == 'a'
    assert fta.top('missing') is None


def test_full_text_analysis_all_follows_chains_and_is_repeatable(sample_data):
    fta = FullTextAnalysis(sample_data)
    assert fta.all('apple') == ['a', 'b', 'c']
    assert fta.all('apple') == ['a', 'b', 'c']
    assert fta.all('banana') == ['d']


def test_full_text_analysis_filter(sample_data):
    fta = FullTextAnalysis(sample_data)
    assert fta.filter('not callable') == []
    result = fta.filter(lambda t: t.startswith('a'))
    assert list(result.keys()) == ['apple']
    assert result['apple'] is sample_data['apple']
    assert fta.filter(lambda t: False) == {}


def test_full_text_analysis_serialize(sample_data):
    fta = FullTextAnalysis(sample_data)
    assert fta.serialize(None) is None
    assert fta.serialize(fta.filter(lambda t: t == 'apple')) == ['a', 'b', 'c']
    ids = fta.serialize(fta.filter(lambda t: True))
    assert sorted(ids) == ['a', 'b', 'c', 'd']
    assert len(ids) == len(set(ids))


def test_full_text_analysis_highlight():
    fta = FullTextAnalysis({})
    assert fta.highlight('cat', 'the cat sat', '<b>{{_}}</b>') == 'the <b>cat</b> sat'
    # word boundaries are respected and matching is case-sensitive
    assert fta.highlight('cat', 'concat Cat cat', '[{{_}}]') == 'concat Cat [cat]'
