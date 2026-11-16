import pytest

from memorydb import MemoryDb
from text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


# ---------------------------------------------------------------------------
# MemoryDb.textIndex
# ---------------------------------------------------------------------------

def test_textindex_creates_partition_index_and_returns_none(db):
    result = db.textIndex('p', {'body': True})

    assert result is None
    assert db.all_indices == {'p': {'body': True}}


def test_textindex_merges_keys_and_overrides_existing_ones(db):
    db.textIndex('p', {'body': True, 'title': {'weight': 2}})
    db.textIndex('p', {'body': {'weight': 5}, 'tag': True})

    assert db.all_indices['p'] == {
        'body': {'weight': 5},
        'title': {'weight': 2},
        'tag': True,
    }


def test_textindex_is_scoped_per_partition(db):
    db.textIndex('a', {'body': True})
    db.textIndex('b', {'title': {'weight': 3}})

    assert db.all_indices['a'] == {'body': True}
    assert db.all_indices['b'] == {'title': {'weight': 3}}


# ---------------------------------------------------------------------------
# MemoryDb.insert
# ---------------------------------------------------------------------------

def test_insert_sets_keys_and_timestamp_on_stored_row_and_input(db):
    item = {'body': 'hello'}
    db.insert('p', 'r1', item)

    stored = db.table[0]
    assert stored['partition_key'] == 'p'
    assert stored['row_key'] == 'r1'
    assert isinstance(stored['timestamp'], float)
    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'


def test_insert_without_index_stores_no_index_data(db):
    db.insert('p', 'r1', {'body': 'hello world'})

    assert 'index_data' not in db.table[0]


def test_insert_stores_a_shallow_copy_of_the_item(db):
    item = {'body': 'alpha'}
    db.insert('p', 'r1', item)
    item['body'] = 'beta'

    assert db.table[0]['body'] == 'alpha'


def test_insert_builds_index_data_sorted_by_score(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'foo bar foo'})

    # consecutive grouping: foo(1), bar(1), foo(1) -> foo=2, bar=1
    assert db.table[0]['index_data'] == [('foo', 2), ('bar', 1)]


def test_insert_counts_consecutive_repeats_and_applies_weight(db):
    db.textIndex('p', {'body': {'weight': 3}})
    db.insert('p', 'r1', {'body': 'a a b'})

    # a: 2 * 3 = 6, b: 1 * 3 = 3
    assert db.table[0]['index_data'] == [('a', 6), ('b', 3)]


def test_insert_orders_index_data_by_descending_score(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'b c c c a a'})

    assert db.table[0]['index_data'] == [('c', 3), ('a', 2), ('b', 1)]


def test_insert_ignores_fields_that_are_not_indexed(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'red', 'other': 'blue'})

    assert db.table[0]['index_data'] == [('red', 1)]


def test_insert_stores_empty_index_data_when_no_indexed_field_present(db):
    db.textIndex('p', {'title': True})
    db.insert('p', 'r1', {'body': 'red'})

    assert db.table[0]['index_data'] == []


def test_insert_accumulates_scores_across_fields_with_weights(db):
    db.textIndex('p', {'title': True, 'body': {'weight': 2}})
    db.insert('p', 'r1', {'title': 'fox', 'body': 'fox dog'})

    # fox: title 1 + body 2 = 3, dog: body 2
    assert db.table[0]['index_data'] == [('fox', 3), ('dog', 2)]


# ---------------------------------------------------------------------------
# MemoryDb.textAnalysis
# ---------------------------------------------------------------------------

def test_textanalysis_returns_none_when_partition_has_no_index_data(db):
    db.insert('p', 'r1', {'body': 'hello'})

    assert db.textAnalysis('p') is None


def test_textanalysis_returns_none_for_empty_partition(db):
    db.textIndex('p', {'body': True})

    assert db.textAnalysis('p') is None


def test_textanalysis_returns_full_text_analysis_with_terms(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'fox dog'})
    db.insert('p', 'r2', {'body': 'dog cat'})

    analysis = db.textAnalysis('p')

    assert isinstance(analysis, FullTextAnalysis)
    assert sorted(analysis.terms()) == ['cat', 'dog', 'fox']


def test_textanalysis_ignores_rows_from_other_partitions(db):
    db.textIndex('p', {'body': True})
    db.textIndex('q', {'body': True})
    db.insert('p', 'r1', {'body': 'alpha'})
    db.insert('q', 'r2', {'body': 'beta'})

    analysis = db.textAnalysis('p')

    assert analysis.terms() == ['alpha']


def test_textanalysis_sums_scores_and_chains_rows_with_same_score(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'fox'})
    db.insert('p', 'r2', {'body': 'fox'})

    data = db.textAnalysis('p').dump()

    assert data['fox']['sum'] == 2
    assert list(data['fox']['spread'].keys()) == [1]
    head = data['fox']['spread'][1]
    assert head['value'] == 'r1'
    assert head['next']['value'] == 'r2'


def test_textanalysis_sum_weights_each_row_by_its_score(db):
    db.textIndex('p', {'body': True})
    db.insert('p', 'r1', {'body': 'fox fox fox'})
    db.insert('p', 'r2', {'body': 'fox'})

    analysis = db.textAnalysis('p')

    # r1 contributes score 3, r2 contributes score 1
    assert analysis.dump()['fox']['sum'] == 4
    assert analysis.scores('fox') == [3, 1]
    assert analysis.top('fox') == 'r1'


# ---------------------------------------------------------------------------
# FullTextAnalysis (text_analysis.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    return {
        'fox': {
            'sum': 3,
            'spread': {
                2: {'value': 'r1'},
                1: {'value': 'r2', 'next': {'value': 'r3'}},
            },
        },
        'dog': {
            'sum': 4,
            'spread': {
                3: {'value': 'r3'},
                1: {'value': 'r1'},
            },
        },
    }


def test_fulltextanalysis_scores_and_top(sample_data):
    analysis = FullTextAnalysis(sample_data)

    assert analysis.scores('fox') == [2, 1]
    assert analysis.scores('missing') is None
    assert analysis.top('fox') == 'r1'
    assert analysis.top('missing') is None


def test_fulltextanalysis_all_collects_every_row_in_chains(sample_data):
    analysis = FullTextAnalysis(sample_data)

    assert sorted(analysis.all('fox')) == ['r1', 'r2', 'r3']


def test_fulltextanalysis_serialize_returns_ids_in_score_order(sample_data):
    analysis = FullTextAnalysis(sample_data)

    assert analysis.serialize({'fox': sample_data['fox']}) == ['r1', 'r2', 'r3']


def test_fulltextanalysis_serialize_deduplicates_ids_across_terms(sample_data):
    analysis = FullTextAnalysis(sample_data)

    # r1 appears under both 'fox' and 'dog' but must be listed once
    ids = analysis.serialize(sample_data)

    assert ids == ['r3', 'r1', 'r2'] or sorted(ids) == ['r1', 'r2', 'r3']
    assert len(ids) == len(set(ids)) == 3


def test_fulltextanalysis_serialize_none_returns_none():
    analysis = FullTextAnalysis({})

    assert analysis.serialize(None) is None


def test_fulltextanalysis_filter_with_callable_returns_matching_terms(sample_data):
    analysis = FullTextAnalysis(sample_data)

    result = analysis.filter(lambda term: term.startswith('f'))

    assert result == {'fox': sample_data['fox']}


def test_fulltextanalysis_filter_with_non_callable_returns_empty_list(sample_data):
    analysis = FullTextAnalysis(sample_data)

    assert analysis.filter('fox') == []


def test_fulltextanalysis_highlight_wraps_whole_word_matches():
    analysis = FullTextAnalysis({})

    assert analysis.highlight('fox', 'the fox ran', '<b>{{_}}</b>') == 'the <b>fox</b> ran'
    assert analysis.highlight('fox', 'foxes and fox', '[{{_}}]') == 'foxes and [fox]'
