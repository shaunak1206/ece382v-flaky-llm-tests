import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.dirname(_HERE), os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from memorydb import MemoryDb
import text_analysis
from text_analysis import FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


def _indexed_db(partition='p', desc=None):
    database = MemoryDb()
    database.textIndex(partition, desc if desc is not None else {'body': {}})
    return database


def _sample_data():
    return {
        'hello': {
            'sum': 8,
            'spread': {
                3: {'value': 'C'},
                2: {'value': 'A', 'next': {'value': 'D'}},
                1: {'value': 'B'},
            },
        },
        'world': {
            'sum': 2,
            'spread': {2: {'value': 'A'}},
        },
    }


# ---------------------------------------------------------------- textIndex

def test_textindex_creates_partition_index_with_descriptors(db):
    db.textIndex('p', {'body': {'weight': 2}})
    assert db.all_indices == {'p': {'body': {'weight': 2}}}


def test_textindex_empty_descriptor_creates_empty_partition_index(db):
    db.textIndex('p', {})
    assert db.all_indices == {'p': {}}


def test_textindex_merges_into_existing_partition_and_overwrites_fields(db):
    db.textIndex('p', {'title': {'weight': 1}, 'body': {}})
    db.textIndex('p', {'title': {'weight': 5}, 'summary': {}})
    assert db.all_indices['p'] == {
        'title': {'weight': 5},
        'body': {},
        'summary': {},
    }


# ------------------------------------------------------------------- insert

def test_insert_sets_key_fields_and_timestamp(db):
    item = {'name': 'widget'}
    db.insert('p', 'r1', item)

    assert item['partition_key'] == 'p'
    assert item['row_key'] == 'r1'
    assert isinstance(item['timestamp'], float)

    assert len(db.table) == 1
    row = db.table[0]
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['name'] == 'widget'
    assert row['timestamp'] == item['timestamp']


def test_insert_stores_copy_not_original(db):
    item = {'name': 'x'}
    db.insert('p', 'r1', item)
    item['name'] = 'changed'

    assert db.table[0]['name'] == 'x'
    assert db.table[0] is not item


def test_insert_without_index_adds_no_index_data(db):
    db.insert('p', 'r1', {'body': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_computes_weighted_scores(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': {}})
    db.insert('p', 'r1', {'title': 'a a b', 'body': 'a'})

    # title: a -> 2 * 3 = 6, b -> 1 * 3 = 3; body: a -> 1; total a = 7
    assert db.table[0]['index_data'] == [('a', 7), ('b', 3)]


def test_insert_ignores_fields_that_are_not_indexed(db):
    db.textIndex('p', {'body': {}})
    db.insert('p', 'r1', {'body': 'x y', 'other': 'z z z'})

    assert db.table[0]['index_data'] == [('x', 1), ('y', 1)]


def test_insert_into_indexed_partition_without_matching_fields_stores_empty_index(db):
    db.textIndex('p', {'body': {}})
    db.insert('p', 'r1', {'other': 'nothing indexed here'})

    assert db.table[0]['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_without_indexed_rows(db):
    db.insert('p', 'r', {'body': 'x'})  # inserted before any index exists
    assert db.textAnalysis('p') is None
    assert db.textAnalysis('empty-partition') is None


def test_textanalysis_sums_scores_across_rows(db):
    db = _indexed_db()
    db.insert('p', 'A', {'body': 'hello hello world'})
    db.insert('p', 'B', {'body': 'hello'})
    db.insert('p', 'C', {'body': 'hello hello hello'})

    result = db.textAnalysis('p')
    assert isinstance(result, text_analysis.FullTextAnalysis)

    data = result.dump()
    assert data['hello']['sum'] == 6
    assert data['world']['sum'] == 1


def test_textanalysis_spread_groups_rows_by_score(db):
    db = _indexed_db()
    db.insert('p', 'A', {'body': 'hello hello world'})
    db.insert('p', 'B', {'body': 'hello'})
    db.insert('p', 'C', {'body': 'hello hello hello'})

    spread = db.textAnalysis('p').dump()['hello']['spread']
    assert set(spread.keys()) == {1, 2, 3}
    assert spread[3]['value'] == 'C'
    assert spread[2]['value'] == 'A'
    assert spread[1]['value'] == 'B'


def test_textanalysis_chains_rows_with_equal_score(db):
    db = _indexed_db()
    db.insert('p', 'A', {'body': 'x'})
    db.insert('p', 'B', {'body': 'x'})
    db.insert('p', 'C', {'body': 'x'})

    data = db.textAnalysis('p').dump()
    spread = data['x']['spread']

    assert list(spread.keys()) == [1]
    head = spread[1]
    assert head['value'] == 'A'
    assert head['next']['value'] == 'B'
    assert head['next']['next']['value'] == 'C'
    assert 'next' not in head['next']['next']
    assert data['x']['sum'] == 3


def test_textanalysis_applies_field_weights(db):
    db = MemoryDb()
    db.textIndex('p', {'title': {'weight': 4}})
    db.insert('p', 'r1', {'title': 'alpha beta'})

    data = db.textAnalysis('p').dump()
    assert data['alpha']['sum'] == 4
    assert data['beta']['sum'] == 4
    assert set(data['alpha']['spread'].keys()) == {4}


def test_textanalysis_skips_rows_inserted_before_indexing(db):
    db = MemoryDb()
    db.insert('p', 'old', {'body': 'word'})
    db.textIndex('p', {'body': {}})
    db.insert('p', 'new', {'body': 'word'})

    data = db.textAnalysis('p').dump()
    assert data['word']['sum'] == 1
    assert data['word']['spread'][1]['value'] == 'new'


# ------------------------------------------------------- FullTextAnalysis

def test_fulltextanalysis_terms_and_dump_return_data():
    data = _sample_data()
    fa = FullTextAnalysis(data)

    assert sorted(fa.terms()) == ['hello', 'world']
    assert fa.dump() is data


def test_scores_returns_descending_scores_or_none():
    fa = FullTextAnalysis(_sample_data())

    assert fa.scores('hello') == [3, 2, 1]
    assert fa.scores('missing') is None


def test_top_returns_row_key_with_highest_score():
    fa = FullTextAnalysis(_sample_data())

    assert fa.top('hello') == 'C'
    assert fa.top('missing') is None


def test_all_returns_every_row_key_including_chained_rows():
    fa = FullTextAnalysis(_sample_data())

    assert sorted(fa.all('hello')) == ['A', 'B', 'C', 'D']
    assert fa.all('world') == ['A']


def test_filter_keeps_matching_terms_with_original_values():
    data = {
        'apple': {'sum': 1, 'spread': {1: {'value': 'r1'}}},
        'apricot': {'sum': 1, 'spread': {1: {'value': 'r2'}}},
        'banana': {'sum': 1, 'spread': {1: {'value': 'r3'}}},
    }
    fa = FullTextAnalysis(data)

    result = fa.filter(lambda term: term.startswith('ap'))

    assert set(result.keys()) == {'apple', 'apricot'}
    assert result['apple'] is data['apple']


def test_filter_with_non_callable_returns_empty_list():
    fa = FullTextAnalysis(_sample_data())
    assert fa.filter('ap') == []


def test_serialize_returns_unique_row_keys_ordered_by_score():
    fa = FullTextAnalysis(_sample_data())

    assert fa.serialize(None) is None

    # filter with an always-true expression keeps term order: hello, world.
    # hello: score 3 -> C, score 2 -> A then chained D, score 1 -> B.
    # world: A is already present, so it is not repeated.
    serialized = fa.serialize(fa.filter(lambda term: True))
    assert serialized == ['C', 'A', 'D', 'B']


def test_highlight_wraps_whole_word_matches_only():
    fa = FullTextAnalysis({})
    result = fa.highlight('cat', 'the cat sat, cats!', '<b>{{_}}</b>')
    assert result == 'the <b>cat</b> sat, cats!'


def test_highlight_is_case_sensitive():
    fa = FullTextAnalysis({})
    result = fa.highlight('cat', 'Cat cat', '[{{_}}]')
    assert result == 'Cat [cat]'
