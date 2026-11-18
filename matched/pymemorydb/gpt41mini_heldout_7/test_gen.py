import pytest
from pymemorydb import memorydb, text_analysis
import time


@pytest.fixture
def db():
    return memorydb.MemoryDb()


def test_textIndex_creates_new_index_for_partition_key(db):
    desc = {'field1': {}, 'field2': {'weight': 2}}
    db.textIndex('partition1', desc)
    assert 'partition1' in db.all_indices
    assert db.all_indices['partition1'] == desc


def test_textIndex_updates_existing_index(db):
    db.all_indices['partition1'] = {'field1': {}}
    desc_update = {'field2': {'weight': 3}}
    db.textIndex('partition1', desc_update)
    expected = {'field1': {}, 'field2': {'weight': 3}}
    assert db.all_indices['partition1'] == expected


def test_insert_adds_item_with_partition_and_row_key_and_timestamp(db):
    item = {'name': 'test'}
    db.insert('part1', 'row1', item)
    assert any(i for i in db.table if i['partition_key'] == 'part1' and i['row_key'] == 'row1')
    stored = next(i for i in db.table if i['partition_key'] == 'part1' and i['row_key'] == 'row1')
    assert 'timestamp' in stored
    assert stored['name'] == 'test'


def test_insert_calls_indexItem_and_sets_index_data_if_index_exists(db):
    # Setup index
    desc = {'field': {}}
    db.textIndex('part1', desc)
    item = {'field': 'hello hello world', 'other': 'ignored'}
    db.insert('part1', 'rowA', item)
    stored = next(i for i in db.table if i['row_key'] == 'rowA')
    assert 'index_data' in stored
    # index_data should be list of tuples (word, weighted count)
    expected = [('hello', 2), ('world', 1)]
    assert stored['index_data'] == expected


def test_indexItem_returns_none_if_no_indices(db):
    item = {'field': 'something', 'partition_key': 'p1', 'row_key': 'r1'}
    result = db.indexItem(item)
    assert result is None


def test_indexItem_weights_and_groupby_correct(db):
    db.all_indices['p1'] = {'field': {'weight': 3}}
    item = {'partition_key': 'p1', 'row_key': 'r1', 'field': 'a a b'}
    result = db.indexItem(item)
    # 'a' appears twice * weight 3 = 6, 'b' once * 3 = 3
    assert result == [('a', 6), ('b', 3)]


def test_textAnalysis_returns_FullTextAnalysis_object(db):
    desc = {'text': {}}
    db.textIndex('p1', desc)
    item1 = {'text': 'foo bar bar', 'partition_key': 'p1', 'row_key': 'r1'}
    item2 = {'text': 'foo', 'partition_key': 'p1', 'row_key': 'r2'}
    db.insert('p1', 'r1', item1)
    db.insert('p1', 'r2', item2)
    analysis = db.textAnalysis('p1')
    assert isinstance(analysis, text_analysis.FullTextAnalysis)
    terms = analysis.terms()
    assert all(isinstance(t, str) for t in terms)  # keys are strings


def test_textAnalysis_returns_None_if_no_index_data(db):
    db.insert('p1', 'r1', {'field': 'value', 'partition_key': 'p1', 'row_key': 'r1'})
    result = db.textAnalysis('p1')
    assert result is None


def test_textAnalysis_handles_linked_list_in_tokenized_data(db):
    desc = {'words': {}}
    db.textIndex('p1', desc)
    item1 = {'words': 'a a a', 'partition_key': 'p1', 'row_key': 'r1'}
    item2 = {'words': 'a', 'partition_key': 'p1', 'row_key': 'r2'}
    db.insert('p1', 'r1', item1)
    db.insert('p1', 'r2', item2)
    analysis = db.textAnalysis('p1')
    assert 'words' in analysis.data
    spread = analysis.data['words']['spread']
    # The value with key 'a' in spread should have linked 'next' objects
    val = spread.get(1) or spread.get(3) or next(iter(spread.values()))
    linked_values = []
    while val:
        linked_values.append(val['value'])
        val = val.get('next')
    assert all(v in ['r1', 'r2'] for v in linked_values)


def test_FullTextAnalysis_dump_returns_data(db):
    fta = text_analysis.FullTextAnalysis({'key': {'sum': 1, 'spread': {}}})
    assert fta.dump() == {'key': {'sum': 1, 'spread': {}}}


def test_FullTextAnalysis_terms_returns_keys(db):
    data = {'term1': {}, 'term2': {}}
    fta = text_analysis.FullTextAnalysis(data)
    terms = fta.terms()
    assert set(terms) == set(['term1', 'term2'])


def test_FullTextAnalysis_scores_returns_sorted_keys(db):
    data = {'term': {'spread': {1: {'value': 'a'}, 3: {'value': 'b'}, 2: {'value': 'c'}}}}
    fta = text_analysis.FullTextAnalysis(data)
    scores = fta.scores('term')
    assert scores == [3, 2, 1]


def test_FullTextAnalysis_scores_returns_None_for_missing_term(db):
    fta = text_analysis.FullTextAnalysis({})
    assert fta.scores('missing') is None


def test_FullTextAnalysis_top_returns_value_of_highest_score(db):
    data = {'term': {'spread': {1: {'value': 'a'}, 5: {'value': 'best'}, 3: {'value': 'c'}}}}
    fta = text_analysis.FullTextAnalysis(data)
    top_val = fta.top('term')
    assert top_val == 'best'


def test_FullTextAnalysis_top_returns_None_for_missing_term(db):
    fta = text_analysis.FullTextAnalysis({})
    assert fta.top('nope') is None


def test_FullTextAnalysis_all_returns_all_values_in_chain(db):
    data = {'term': {'spread': {
        1: {'value': 'a', 'next': {'value': 'b', 'next': {'value': 'c'}}}
    }}}
    fta = text_analysis.FullTextAnalysis(data)
    all_vals = fta.all('term')
    assert all_vals == ['a', 'b', 'c']


def test_FullTextAnalysis_filter_returns_empty_list_if_not_callable(db):
    fta = text_analysis.FullTextAnalysis({'a': {}, 'b': {}})
    result = fta.filter("not_callable")
    assert result == []


def test_FullTextAnalysis_filter_returns_sorted_filtered_results(db):
    data = {
        'a': {'sum': 5, 'spread': {}},
        'b': {'sum': 10, 'spread': {}},
        'c': {'sum': 1, 'spread': {}}
    }
    fta = text_analysis.FullTextAnalysis(data)
    # filter returns only keys having sum >= 5
    result = fta.filter(lambda x: data[x]['sum'] >= 5)
    # keys in descending order by sum
    keys = list(result.keys())
    assert keys == ['b', 'a']


def test_FullTextAnalysis_serialize_returns_ids(db):
    # Setup data with linked next values
    data = {
        'term': {
            'sum': 10,
            'spread': {
                3: {'value': 'r1', 'next': {'value': 'r2'}},
                5: {'value': 'r3'}
            }
        }
    }
    fta = text_analysis.FullTextAnalysis(data)
    serialized = fta.serialize(data)
    # should contain all: r1, r2, r3 without duplicates
    assert set(serialized) == {'r1', 'r2', 'r3'}


def test_FullTextAnalysis_serialize_returns_None_for_None_input(db):
    fta = text_analysis.FullTextAnalysis({})
    assert fta.serialize(None) is None


def test_FullTextAnalysis_highlight_replaces_term(db):
    fta = text_analysis.FullTextAnalysis({})
    text = "This is a test. Testing the test."
    rendered = fta.highlight('test', text, '<b>{{_}}</b>')
    # Should replace 'test' exactly, case sensitive on whole word
    assert '<b>test</b>' in rendered
    assert 'Testing' in rendered  # Should not be changed


def test_FullTextAnalysis_highlight_case_sensitive_match(db):
    fta = text_analysis.FullTextAnalysis({})
    text = "test Test TEST test."
    rendered = fta.highlight('test', text, '[{{_}}]')
    # Only exact lowercase 'test' whole words replaced
    assert rendered.count('[test]') == 2
    # but because regex is default case-sensitive, 'Test' and 'TEST' remain
    assert 'Test' in rendered
    # Now test with re.I, but original code replaces with render.replace(..., re.I),
    # which is incorrect API usage, but we test as is
