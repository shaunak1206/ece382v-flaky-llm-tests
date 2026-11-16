import os
import sys

import pytest

# Make the project root importable when pytest is run from the project root.
sys.path.insert(0, os.getcwd())

try:
    import memorydb as memorydb_module
except ImportError:
    from pymemorydb import memorydb as memorydb_module

MemoryDb = memorydb_module.MemoryDb
FullTextAnalysis = memorydb_module.text_analysis.FullTextAnalysis


def _chain(node):
    """Collect the 'value' entries of a linked list of postings."""
    values = []
    while node is not None:
        values.append(node['value'])
        node = node.get('next')
    return values


@pytest.fixture
def db():
    return MemoryDb()


# ---------------------------------------------------------------------------
# textIndex
# ---------------------------------------------------------------------------

def test_textIndex_stores_field_descriptions_and_returns_none(db):
    result = db.textIndex('p', {'title': {'weight': 2}})

    assert result is None
    assert db.all_indices == {'p': {'title': {'weight': 2}}}


def test_textIndex_merges_fields_across_calls(db):
    db.textIndex('p', {'title': {}})
    db.textIndex('p', {'body': {'weight': 4}})

    assert set(db.all_indices['p'].keys()) == {'title', 'body'}
    assert db.all_indices['p']['body'] == {'weight': 4}


def test_textIndex_does_not_affect_other_partitions(db):
    db.textIndex('a', {'title': {}})
    db.insert('b', 'r1', {'title': 'red'})

    assert 'b' not in db.all_indices
    assert 'index_data' not in db.table[0]


def test_textIndex_does_not_reindex_existing_rows(db):
    db.insert('p', 'r1', {'title': 'red'})
    db.textIndex('p', {'title': {}})

    assert 'index_data' not in db.table[0]


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_without_index_stores_row_without_index_data(db):
    db.insert('p', 'r1', {'title': 'red'})

    row = db.table[0]
    assert len(db.table) == 1
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert isinstance(row['timestamp'], float)
    assert 'index_data' not in row


def test_insert_tokenizes_indexed_field_with_counts(db):
    db.textIndex('p', {'title': {'weight': 1}})
    db.insert('p', 'r1', {'title': 'red red blue'})

    assert db.table[0]['index_data'] == [('red', 2), ('blue', 1)]


def test_insert_applies_field_weight(db):
    db.textIndex('p', {'title': {'weight': 3}})
    db.insert('p', 'r1', {'title': 'red blue'})

    assert dict(db.table[0]['index_data']) == {'red': 3, 'blue': 3}


def test_insert_dict_description_without_weight_defaults_to_one(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red red'})

    assert db.table[0]['index_data'] == [('red', 2)]


def test_insert_combines_scores_across_multiple_indexed_fields(db):
    db.textIndex('p', {'title': {'weight': 2}, 'body': {'weight': 1}})
    db.insert('p', 'r1', {'title': 'red', 'body': 'red blue'})

    index_data = db.table[0]['index_data']
    assert dict(index_data) == {'red': 3, 'blue': 1}
    assert index_data[0] == ('red', 3)


def test_insert_ignores_unindexed_fields(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red', 'body': 'blue'})

    tokens = [token for token, _ in db.table[0]['index_data']]
    assert tokens == ['red']


def test_insert_row_without_indexed_values_gets_empty_index_data(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'body': 'blue'})

    assert db.table[0]['index_data'] == []


def test_insert_stores_independent_copy_of_item(db):
    item = {'title': 'red'}
    db.insert('p', 'r1', item)
    item['title'] = 'changed'

    assert db.table[0]['title'] == 'red'


# ---------------------------------------------------------------------------
# textAnalysis
# ---------------------------------------------------------------------------

def test_textAnalysis_returns_none_without_indexed_rows(db):
    db.insert('p', 'r1', {'title': 'red'})

    assert db.textAnalysis('p') is None


def test_textAnalysis_computes_term_sums_and_terms(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red red blue'})
    db.insert('p', 'r2', {'title': 'red'})

    ta = db.textAnalysis('p')

    assert isinstance(ta, FullTextAnalysis)
    assert sorted(ta.terms()) == ['blue', 'red']
    data = ta.dump()
    assert data['red']['sum'] == 3
    assert data['blue']['sum'] == 1


def test_textAnalysis_groups_rows_by_score_in_spread(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red red blue'})
    db.insert('p', 'r2', {'title': 'red'})

    spread = db.textAnalysis('p').dump()['red']['spread']

    assert set(spread.keys()) == {1, 2}
    assert spread[2]['value'] == 'r1'
    assert spread[1]['value'] == 'r2'


def test_textAnalysis_chains_rows_sharing_term_and_score(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red'})
    db.insert('p', 'r2', {'title': 'red'})
    db.insert('p', 'r3', {'title': 'red'})

    data = db.textAnalysis('p').dump()

    assert list(data['red']['spread'].keys()) == [1]
    assert _chain(data['red']['spread'][1]) == ['r1', 'r2', 'r3']
    assert data['red']['sum'] == 3


def test_textAnalysis_is_limited_to_requested_partition(db):
    db.textIndex('p', {'title': {}})
    db.textIndex('q', {'title': {}})
    db.insert('p', 'r1', {'title': 'red'})
    db.insert('q', 'r2', {'title': 'red'})

    ta = db.textAnalysis('p')
    spread = ta.dump()['red']['spread']

    assert ta.dump()['red']['sum'] == 1
    assert _chain(spread[1]) == ['r1']


# ---------------------------------------------------------------------------
# FullTextAnalysis
# ---------------------------------------------------------------------------

def test_fulltextanalysis_scores_are_descending_and_none_for_missing():
    data = {'red': {'sum': 6, 'spread': {1: {'value': 'a'}, 3: {'value': 'b'}, 2: {'value': 'c'}}}}
    ta = FullTextAnalysis(data)

    assert ta.scores('red') == [3, 2, 1]
    assert ta.scores('missing') is None


def test_fulltextanalysis_top_returns_row_of_highest_score():
    data = {'red': {'sum': 6, 'spread': {1: {'value': 'a'}, 3: {'value': 'b'}, 2: {'value': 'c'}}}}
    ta = FullTextAnalysis(data)

    assert ta.top('red') == 'b'
    assert ta.top('missing') is None


def test_fulltextanalysis_all_walks_linked_postings():
    data = {
        'red': {
            'sum': 3,
            'spread': {
                2: {'value': 'a', 'next': {'value': 'b'}},
                1: {'value': 'c'},
            },
        }
    }
    ta = FullTextAnalysis(data)

    assert sorted(ta.all('red')) == ['a', 'b', 'c']


def test_fulltextanalysis_filter_with_callable_returns_matching_terms():
    data = {
        'red': {'sum': 3, 'spread': {}},
        'blue': {'sum': 1, 'spread': {}},
    }
    ta = FullTextAnalysis(data)

    result = ta.filter(lambda term: term.startswith('r'))

    assert result == {'red': data['red']}


def test_fulltextanalysis_filter_non_callable_returns_empty_list():
    ta = FullTextAnalysis({'red': {'sum': 1, 'spread': {}}})

    assert ta.filter('red') == []
    assert ta.filter(None) == []


def test_fulltextanalysis_serialize_orders_ids_by_score(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red red blue'})
    db.insert('p', 'r2', {'title': 'red'})
    ta = db.textAnalysis('p')

    assert ta.serialize(ta.filter(lambda term: term == 'red')) == ['r1', 'r2']
    assert ta.serialize(ta.filter(lambda term: term == 'blue')) == ['r1']


def test_fulltextanalysis_serialize_deduplicates_across_terms_and_handles_none(db):
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'red red blue'})
    db.insert('p', 'r2', {'title': 'red'})
    ta = db.textAnalysis('p')

    ids = ta.serialize(ta.filter(lambda term: True))

    assert sorted(ids) == ['r1', 'r2']
    assert len(ids) == 2
    assert ta.serialize(None) is None


def test_fulltextanalysis_highlight_wraps_whole_word_matches_only():
    ta = FullTextAnalysis({})

    result = ta.highlight('red', 'reddish red blue red', '[{{_}}]')

    assert result == 'reddish [red] blue [red]'
