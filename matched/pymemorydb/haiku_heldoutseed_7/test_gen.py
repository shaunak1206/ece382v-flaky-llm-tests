import sys

import pytest

try:
    from pymemorydb.memorydb import MemoryDb
    from pymemorydb.text_analysis import FullTextAnalysis
except ImportError:
    from memorydb import MemoryDb
    from text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


# ---------------------------------------------------------------------------
# MemoryDb.textIndex
# ---------------------------------------------------------------------------

def test_text_index_creates_partition_entry(db):
    db.textIndex('posts', {'title': 1})
    assert db.all_indices == {'posts': {'title': 1}}


def test_text_index_merges_and_overwrites_existing_fields(db):
    db.textIndex('posts', {'title': 1})
    db.textIndex('posts', {'title': 2, 'body': {'weight': 3}})
    assert db.all_indices['posts'] == {'title': 2, 'body': {'weight': 3}}


def test_text_index_partitions_are_independent(db):
    db.textIndex('posts', {'title': 1})
    db.textIndex('users', {'name': 5})
    assert db.all_indices['posts'] == {'title': 1}
    assert db.all_indices['users'] == {'name': 5}


def test_text_index_stores_descriptor_values_by_reference(db):
    weight = {'weight': 3}
    db.textIndex('posts', {'title': weight})
    assert db.all_indices['posts']['title'] is weight


# ---------------------------------------------------------------------------
# MemoryDb.insert
# ---------------------------------------------------------------------------

def test_insert_sets_key_fields_and_timestamp(db, monkeypatch):
    module = sys.modules[MemoryDb.__module__]
    monkeypatch.setattr(module, 'time', lambda: 1234.5)
    item = {'name': 'x'}
    db.insert('users', 'u1', item)
    stored = db.table[0]
    assert stored['partition_key'] == 'users'
    assert stored['row_key'] == 'u1'
    assert stored['timestamp'] == 1234.5
    assert stored['name'] == 'x'
    # insert mutates the caller's dict before storing a copy
    assert item['partition_key'] == 'users'
    assert item['row_key'] == 'u1'


def test_insert_without_index_leaves_no_index_data(db):
    db.insert('posts', 'r1', {'title': 'apple'})
    assert 'index_data' not in db.table[0]


def test_insert_builds_index_data_sorted_by_score(db):
    db.textIndex('posts', {'title': 1})
    db.insert('posts', 'r1', {'title': 'apple apple banana'})
    assert db.table[0]['index_data'] == [('apple', 2), ('banana', 1)]


def test_insert_counts_non_consecutive_tokens_across_positions(db):
    db.textIndex('posts', {'title': 1})
    db.insert('posts', 'r1', {'title': 'apple banana apple'})
    assert dict(db.table[0]['index_data']) == {'apple': 2, 'banana': 1}


def test_insert_applies_weight_and_combines_fields(db):
    db.textIndex('posts', {'title': {'weight': 3}, 'body': 1})
    db.insert('posts', 'r1', {'title': 'apple apple banana', 'body': 'apple pie'})
    assert dict(db.table[0]['index_data']) == {'apple': 7, 'banana': 3, 'pie': 1}


def test_insert_ignores_unindexed_fields(db):
    db.textIndex('posts', {'title': 1})
    db.insert('posts', 'r1', {'title': 'apple', 'secret': 'hidden words'})
    tokens = [token for token, _ in db.table[0]['index_data']]
    assert tokens == ['apple']


def test_insert_excludes_row_key_from_index(db):
    db.textIndex('posts', {'title': 1, 'row_key': 1})
    db.insert('posts', 'r1', {'title': 'apple'})
    tokens = [token for token, _ in db.table[0]['index_data']]
    assert 'r1' not in tokens
    assert tokens == ['apple']


def test_insert_stores_copy_not_the_original_item(db):
    db.textIndex('posts', {'title': 1})
    item = {'title': 'apple'}
    db.insert('posts', 'r1', item)
    item['title'] = 'changed'
    assert db.table[0] is not item
    assert db.table[0]['title'] == 'apple'


# ---------------------------------------------------------------------------
# MemoryDb.textAnalysis
# ---------------------------------------------------------------------------

def test_text_analysis_aggregates_sum_and_spread(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'apple apple banana'})
    db.insert('p', 'r2', {'title': 'apple'})
    data = db.textAnalysis('p').dump()
    assert set(data.keys()) == {'apple', 'banana'}
    assert data['apple']['sum'] == 3
    assert data['banana']['sum'] == 1
    assert data['apple']['spread'] == {2: {'value': 'r1'}, 1: {'value': 'r2'}}
    assert data['banana']['spread'] == {1: {'value': 'r1'}}


def test_text_analysis_chains_rows_sharing_the_same_score(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'apple'})
    db.insert('p', 'r2', {'title': 'apple'})
    db.insert('p', 'r3', {'title': 'apple'})
    analysis = db.textAnalysis('p')
    spread = analysis.dump()['apple']['spread']
    assert list(spread.keys()) == [1]
    head = spread[1]
    assert head['value'] == 'r1'
    assert head['next']['value'] == 'r2'
    assert head['next']['next']['value'] == 'r3'
    assert analysis.dump()['apple']['sum'] == 3


def test_text_analysis_ignores_other_partitions(db):
    db.textIndex('p', {'title': 1})
    db.textIndex('q', {'title': 1})
    db.insert('p', 'r1', {'title': 'apple'})
    db.insert('q', 'r2', {'title': 'pear'})
    assert db.textAnalysis('p').terms() == ['apple']


def test_text_analysis_skips_rows_without_index_data(db):
    db.insert('p', 'old', {'title': 'apple'})  # inserted before indexing
    db.textIndex('p', {'title': 1})
    db.insert('p', 'new', {'title': 'banana'})
    data = db.textAnalysis('p').dump()
    assert list(data.keys()) == ['banana']


def test_text_analysis_returns_none_for_empty_partition(db):
    db.textIndex('p', {'title': 1})
    assert db.textAnalysis('p') is None


# ---------------------------------------------------------------------------
# text_analysis.FullTextAnalysis
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis():
    return FullTextAnalysis({
        'apple': {'sum': 3, 'spread': {2: {'value': 'r1'}, 1: {'value': 'r2'}}},
        'banana': {'sum': 1, 'spread': {1: {'value': 'r1'}}},
    })


def test_dump_returns_data_and_terms_lists_keys(analysis):
    assert analysis.dump() is analysis.data
    assert sorted(analysis.terms()) == ['apple', 'banana']


def test_scores_sorted_descending_and_none_for_unknown_term(analysis):
    assert analysis.scores('apple') == [2, 1]
    assert analysis.scores('cherry') is None


def test_top_returns_row_key_with_highest_score(analysis):
    assert analysis.top('apple') == 'r1'
    assert analysis.top('cherry') is None


def test_all_collects_row_keys_across_chains():
    chained = FullTextAnalysis({
        'apple': {
            'sum': 4,
            'spread': {
                1: {'value': 'r1', 'next': {'value': 'r2'}},
                3: {'value': 'r3'},
            },
        },
    })
    assert sorted(chained.all('apple')) == ['r1', 'r2', 'r3']


def test_filter_keeps_matching_terms_and_rejects_non_callable(analysis):
    result = analysis.filter(lambda term: term.startswith('a'))
    assert result == {'apple': analysis.data['apple']}
    assert analysis.filter(lambda term: False) == {}
    assert analysis.filter('not callable') == []


def test_serialize_returns_unique_ids_in_score_order_and_handles_edges():
    chained = FullTextAnalysis({
        'apple': {
            'sum': 4,
            'spread': {
                2: {'value': 'r1'},
                1: {'value': 'r2', 'next': {'value': 'r3'}},
            },
        },
        'banana': {'sum': 1, 'spread': {1: {'value': 'r1'}}},
    })
    assert chained.serialize(chained.data) == ['r1', 'r2', 'r3']
    assert chained.serialize(None) is None
    assert chained.serialize({}) == []


def test_highlight_wraps_whole_word_occurrences_only():
    analysis = FullTextAnalysis({})
    out = analysis.highlight('apple', 'pineapple and apple, apple!', '<b>{{_}}</b>')
    assert out == 'pineapple and <b>apple</b>, <b>apple</b>!'
