import os
import sys

import pytest

try:
    from pymemorydb import memorydb as _memorydb_module
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in (os.getcwd(), _here, os.path.dirname(_here)):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    try:
        from pymemorydb import memorydb as _memorydb_module
    except ImportError:
        import memorydb as _memorydb_module

MemoryDb = _memorydb_module.MemoryDb
FullTextAnalysis = _memorydb_module.text_analysis.FullTextAnalysis


@pytest.fixture
def db():
    return MemoryDb()


@pytest.fixture
def frozen_time(monkeypatch):
    monkeypatch.setattr(_memorydb_module, 'time', lambda: 1234.5)
    return 1234.5


@pytest.fixture
def analysis(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'd1', {'title': 'apple apple pie'})  # apple:2, pie:1
    db.insert('p', 'd2', {'title': 'apple'})            # apple:1
    db.insert('p', 'd3', {'title': 'apple apple'})      # apple:2
    return db.textAnalysis('p')


# ---------------------------------------------------------------- textIndex

def test_textindex_creates_partition_and_stores_descriptors(db):
    db.textIndex('books', {'title': {'weight': 2}, 'body': 1})
    assert 'books' in db.all_indices
    assert db.all_indices['books'] == {'title': {'weight': 2}, 'body': 1}


def test_textindex_merges_and_overwrites_fields(db):
    db.textIndex('books', {'title': 1, 'body': 1})
    db.textIndex('books', {'body': {'weight': 5}, 'author': 1})
    assert db.all_indices['books'] == {
        'title': 1,
        'body': {'weight': 5},
        'author': 1,
    }


def test_textindex_partitions_are_isolated(db):
    db.textIndex('a', {'title': 1})
    db.textIndex('b', {'name': 1})
    assert db.all_indices['a'] == {'title': 1}
    assert db.all_indices['b'] == {'name': 1}


# ------------------------------------------------------------------- insert

def test_insert_sets_key_fields_and_timestamp(db, frozen_time):
    db.insert('p', 'r1', {'name': 'x'})
    row = db.table[0]
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert row['timestamp'] == frozen_time
    assert row['name'] == 'x'


def test_insert_stores_shallow_copy_of_item(db):
    item = {'title': 'hello'}
    db.insert('p', 'r1', item)
    item['title'] = 'changed'
    assert db.table[0]['title'] == 'hello'
    assert db.table[0] is not item


def test_insert_without_textindex_stores_no_index_data(db):
    db.insert('p', 'r1', {'title': 'hello world'})
    assert 'index_data' not in db.table[0]


def test_insert_indexes_only_configured_fields(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'alpha', 'body': 'secret words'})
    assert dict(db.table[0]['index_data']) == {'alpha': 1}


def test_insert_accumulates_repeated_tokens(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'go go go stop go'})
    assert dict(db.table[0]['index_data']) == {'go': 4, 'stop': 1}


def test_insert_applies_field_weight(db):
    db.textIndex('p', {'title': {'weight': 3}, 'body': 1})
    db.insert('p', 'r1', {'title': 'alpha beta', 'body': 'beta'})
    assert dict(db.table[0]['index_data']) == {'alpha': 3, 'beta': 4}


def test_insert_sorts_index_data_by_score_descending(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'title': 'low high high mid mid mid'})
    assert db.table[0]['index_data'] == [('mid', 3), ('high', 2), ('low', 1)]


def test_insert_combines_tokens_across_indexed_fields(db):
    db.textIndex('p', {'title': 1, 'body': 1})
    db.insert('p', 'r1', {'title': 'red', 'body': 'red blue'})
    assert dict(db.table[0]['index_data']) == {'red': 2, 'blue': 1}


def test_insert_with_no_configured_field_present_gives_empty_index(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'r1', {'body': 'something'})
    assert db.table[0]['index_data'] == []


# ------------------------------------------------------------- textAnalysis

def test_textanalysis_returns_none_when_nothing_indexed(db):
    db.insert('p', 'r1', {'title': 'hello'})
    assert db.textAnalysis('p') is None


def test_textanalysis_sum_is_score_times_document_count(db):
    db.textIndex('p', {'title': 1})
    db.insert('p', 'd1', {'title': 'apple apple'})  # apple:2
    db.insert('p', 'd2', {'title': 'apple'})        # apple:1
    db.insert('p', 'd3', {'title': 'apple'})        # apple:1
    result = db.textAnalysis('p')
    assert isinstance(result, FullTextAnalysis)
    data = result.dump()
    # score 2 x 1 document + score 1 x 2 documents
    assert data['apple']['sum'] == 4
    assert set(data['apple']['spread'].keys()) == {1, 2}


def test_textanalysis_only_considers_requested_partition(db):
    db.textIndex('p1', {'title': 1})
    db.textIndex('p2', {'title': 1})
    db.insert('p1', 'a', {'title': 'alpha'})
    db.insert('p2', 'b', {'title': 'beta'})
    result = db.textAnalysis('p1')
    assert result.terms() == ['alpha']


def test_textanalysis_skips_items_inserted_before_textindex(db):
    db.insert('p', 'old', {'title': 'legacy'})
    db.textIndex('p', {'title': 1})
    db.insert('p', 'new', {'title': 'fresh'})
    result = db.textAnalysis('p')
    assert result.terms() == ['fresh']
    assert result.all('fresh') == ['new']


# ------------------------------------------------------- FullTextAnalysis

def test_fulltextanalysis_scores_and_top(analysis):
    assert analysis.scores('apple') == [2, 1]
    assert analysis.top('apple') == 'd1'
    assert analysis.scores('missing') is None
    assert analysis.top('missing') is None


def test_fulltextanalysis_all_returns_every_row_for_term(analysis):
    assert sorted(analysis.all('apple')) == ['d1', 'd2', 'd3']
    # Calling again must give the same result.
    assert sorted(analysis.all('apple')) == ['d1', 'd2', 'd3']


def test_fulltextanalysis_filter_returns_matching_terms(analysis):
    result = analysis.filter(lambda t: t.startswith('p'))
    assert list(result.keys()) == ['pie']
    assert result['pie'] is analysis.dump()['pie']


def test_fulltextanalysis_filter_non_callable_returns_empty_list(analysis):
    assert analysis.filter('apple') == []


def test_fulltextanalysis_serialize_orders_by_score_then_chain(analysis):
    filtered = analysis.filter(lambda t: t == 'apple')
    assert analysis.serialize(filtered) == ['d1', 'd3', 'd2']


def test_fulltextanalysis_serialize_none_returns_none(analysis):
    assert analysis.serialize(None) is None


def test_fulltextanalysis_highlight_replaces_whole_words_only(analysis):
    result = analysis.highlight('fox', 'the fox and foxes', '<b>{{_}}</b>')
    assert result == 'the <b>fox</b> and foxes'


def test_fulltextanalysis_highlight_is_case_sensitive(analysis):
    result = analysis.highlight('fox', 'Fox fox', '[{{_}}]')
    assert result == 'Fox [fox]'
