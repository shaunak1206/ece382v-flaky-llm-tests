import os
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _path in (_ROOT, os.path.dirname(_ROOT), os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from pymemorydb import memorydb as _memorydb
except ImportError:
    import memorydb as _memorydb

MemoryDb = _memorydb.MemoryDb
FullTextAnalysis = _memorydb.text_analysis.FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


def make_analysis():
    data = {
        'cat': {
            'sum': 3,
            'spread': {
                2: {'value': 'r1'},
                1: {'value': 'r2', 'next': {'value': 'r3'}},
            },
        },
        'dog': {
            'sum': 1,
            'spread': {1: {'value': 'r4'}},
        },
    }
    return FullTextAnalysis(data), data


# ---------- textIndex ----------

def test_textindex_creates_partition_entry(db):
    db.textIndex('p', {'body': 1})
    assert db.all_indices['p'] == {'body': 1}


def test_textindex_merges_descriptors_for_same_partition(db):
    db.textIndex('p', {'body': 1})
    db.textIndex('p', {'title': {'weight': 2}})
    assert db.all_indices['p'] == {'body': 1, 'title': {'weight': 2}}


def test_textindex_overwrites_existing_field_descriptor(db):
    db.textIndex('p', {'body': 1})
    db.textIndex('p', {'body': 5})
    assert db.all_indices['p']['body'] == 5


def test_textindex_does_not_reindex_existing_rows(db):
    db.insert('p', 'r1', {'body': 'cat'})
    db.textIndex('p', {'body': 1})
    row = db.get('p', 'r1')
    assert 'index_data' not in row


# ---------- insert ----------

def test_insert_stores_row_with_keys_and_timestamp(db):
    db.insert('p', 'r1', {'body': 'hello'})
    row = db.get('p', 'r1')
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['body'] == 'hello'
    assert isinstance(row['timestamp'], float)


def test_insert_timestamp_is_within_call_window(db):
    before = time.time()
    db.insert('p', 'r1', {'body': 'hello'})
    after = time.time()
    row = db.get('p', 'r1')
    assert before <= row['timestamp'] <= after


def test_insert_without_index_has_no_index_data(db):
    db.insert('p', 'r1', {'body': 'hello world'})
    assert 'index_data' not in db.get('p', 'r1')


def test_insert_with_index_builds_term_scores(db):
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'a b a'})
    index_data = db.get('p', 'r1')['index_data']
    assert dict(index_data) == {'a': 2, 'b': 1}


def test_insert_index_data_is_sorted_by_score_descending(db):
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'x y y y z z'})
    scores = [s for _, s in db.get('p', 'r1')['index_data']]
    assert scores == sorted(scores, reverse=True)


def test_insert_applies_weight_from_descriptor(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'x y', 'body': 'z'})
    assert dict(db.get('p', 'r1')['index_data']) == {'x': 3, 'y': 3, 'z': 1}


def test_insert_sums_scores_across_indexed_fields(db):
    db.textIndex('p', {'title': {'weight': 2}, 'body': 1})
    db.insert('p', 'r1', {'title': 'cat', 'body': 'cat dog'})
    assert dict(db.get('p', 'r1')['index_data']) == {'cat': 3, 'dog': 1}


def test_insert_ignores_unindexed_fields_and_empty_index_when_none_match(db):
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'a', 'zebra': 'z x'})
    db.insert('p', 'r2', {'title': 'hello'})
    assert dict(db.get('p', 'r1')['index_data']) == {'a': 1}
    assert db.get('p', 'r2')['index_data'] == []


# ---------- textAnalysis ----------

def test_textanalysis_returns_none_without_index_data(db):
    db.insert('p', 'r1', {'body': 'cat'})
    assert db.textAnalysis('p') is None


def test_textanalysis_returns_fulltextanalysis_instance(db):
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'cat'})
    assert isinstance(db.textAnalysis('p'), FullTextAnalysis)


def test_textanalysis_chains_rows_sharing_term_and_score(db):
    db.textIndex('p', {'body': 1})
    db.insert('p', 'r1', {'body': 'cat dog'})
    db.insert('p', 'r2', {'body': 'cat'})
    ta = db.textAnalysis('p')
    assert sorted(ta.terms()) == ['cat', 'dog']
    cat = ta.dump()['cat']
    assert cat['sum'] == 2
    assert cat['spread'][1]['value'] == 'r1'
    assert cat['spread'][1]['next']['value'] == 'r2'
    assert ta.dump()['dog']['sum'] == 1


def test_textanalysis_sum_accumulates_scores_across_rows(db):
    db.textIndex('p', {'body': 2})
    for rk in ('r1', 'r2', 'r3'):
        db.insert('p', rk, {'body': 'cat'})
    ta = db.textAnalysis('p')
    assert ta.dump()['cat']['sum'] == 6
    assert ta.top('cat') == 'r1'


def test_textanalysis_only_includes_requested_partition(db):
    db.textIndex('p', {'body': 1})
    db.textIndex('q', {'body': 1})
    db.insert('p', 'r1', {'body': 'cat'})
    db.insert('q', 'r9', {'body': 'dog'})
    ta = db.textAnalysis('p')
    assert ta.terms() == ['cat']


# ---------- FullTextAnalysis ----------

def test_fta_terms_and_dump_return_data():
    fa, data = make_analysis()
    assert sorted(fa.terms()) == ['cat', 'dog']
    assert fa.dump() is data


def test_fta_scores_sorted_descending_and_none_for_missing():
    fa, _ = make_analysis()
    assert fa.scores('cat') == [2, 1]
    assert fa.scores('dog') == [1]
    assert fa.scores('bird') is None


def test_fta_top_returns_highest_score_row_or_none():
    fa, _ = make_analysis()
    assert fa.top('cat') == 'r1'
    assert fa.top('dog') == 'r4'
    assert fa.top('bird') is None


def test_fta_all_flattens_chain_in_order():
    fa, _ = make_analysis()
    assert fa.all('cat') == ['r1', 'r2', 'r3']


def test_fta_filter_returns_matching_terms_only():
    fa, data = make_analysis()
    result = fa.filter(lambda t: t.startswith('c'))
    assert list(result.keys()) == ['cat']
    assert result['cat'] is data['cat']


def test_fta_filter_non_callable_returns_empty_list():
    fa, _ = make_analysis()
    assert fa.filter('cat') == []


def test_fta_serialize_returns_unique_row_keys_and_handles_none():
    fa, data = make_analysis()
    assert fa.serialize(fa.dump()) == ['r1', 'r2', 'r3', 'r4']

    shared = {
        'a': {'sum': 1, 'spread': {1: {'value': 'x'}}},
        'b': {'sum': 1, 'spread': {1: {'value': 'x'}}},
    }
    assert FullTextAnalysis(shared).serialize(shared) == ['x']
    assert FullTextAnalysis(shared).serialize(None) is None


def test_fta_highlight_wraps_whole_words_only_and_all_occurrences():
    fa, _ = make_analysis()
    result = fa.highlight('cat', 'cat category the cat', '[{{_}}]')
    assert result == '[cat] category the [cat]'
