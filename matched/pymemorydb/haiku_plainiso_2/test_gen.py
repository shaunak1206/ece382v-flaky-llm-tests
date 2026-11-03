import os
import sys

import pytest

# Make the flat pymemorydb source directory importable (memorydb.py,
# text_analysis.py, aggregate.py live side by side there).
_SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'matched_src', 'pymemorydb'))
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import memorydb as memorydb_module  # noqa: E402
from memorydb import MemoryDb  # noqa: E402

FullTextAnalysis = memorydb_module.text_analysis.FullTextAnalysis


def _docs_db():
    """Three rows indexed on title (weight 2) and body (weight 1).

    Resulting index_data:
      a -> [('cat', 3), ('dog', 2)]
      b -> [('dog', 4), ('bird', 1), ('cat', 1)]
      c -> [('fish', 2), ('dog', 1)]
    """
    db = MemoryDb()
    db.textIndex('docs', {'title': {'weight': 2}, 'body': {}})
    db.insert('docs', 'a', {'title': 'cat dog', 'body': 'cat'})
    db.insert('docs', 'b', {'title': 'dog dog', 'body': 'bird cat'})
    db.insert('docs', 'c', {'title': 'fish', 'body': 'dog'})
    return db


def _chain_db():
    """Rows x, y, z all share token 'red' with score 1, so they form one chain."""
    db = MemoryDb()
    db.textIndex('p', {'t': {}})
    db.insert('p', 'x', {'t': 'red'})
    db.insert('p', 'y', {'t': 'red'})
    db.insert('p', 'z', {'t': 'blue red'})
    return db


def test_insert_index_data_pairs():
    db = _docs_db()
    assert db.table[0]['index_data'] == [('cat', 3), ('dog', 2)]
    assert db.table[1]['index_data'] == [('dog', 4), ('bird', 1), ('cat', 1)]
    assert db.table[2]['index_data'] == [('fish', 2), ('dog', 1)]


def test_insert_sets_keys_and_timestamp():
    db = MemoryDb()
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r1', {'title': 'x', 'partition_key': 'other', 'row_key': 'zzz'})
    row = db.table[0]
    assert row['partition_key'] == 'p'
    assert row['row_key'] == 'r1'
    assert isinstance(row['timestamp'], float)


def test_insert_stores_copy_of_item():
    db = MemoryDb()
    db.textIndex('p', {'title': {}})
    item = {'title': 'cat'}
    db.insert('p', 'r', item)
    item['title'] = 'changed'
    assert db.table[0]['title'] == 'cat'
    assert db.table[0]['index_data'] == [('cat', 1)]


def test_insert_before_textIndex_has_no_index_data():
    db = MemoryDb()
    db.insert('p', 'early', {'t': 'red'})
    assert 'index_data' not in db.table[0]


def test_insert_without_indexed_field_gives_empty_list():
    db = MemoryDb()
    db.textIndex('p', {'title': {}})
    db.insert('p', 'r', {'body': 'hello'})
    assert db.table[0]['index_data'] == []


def test_insert_applies_field_weight():
    db = MemoryDb()
    db.textIndex('p', {'t': {'weight': 5}})
    db.insert('p', 'r', {'t': 'a a b'})
    assert db.table[0]['index_data'] == [('a', 10), ('b', 5)]


def test_insert_strips_and_sums_repeated_tokens():
    db = MemoryDb()
    db.textIndex('p', {'t': {}})
    db.insert('p', 'r', {'t': '  a b a  '})
    # Non-adjacent repeats of 'a' accumulate into one entry.
    assert db.table[0]['index_data'] == [('a', 2), ('b', 1)]


def test_insert_non_string_indexed_field_raises():
    db = MemoryDb()
    db.textIndex('p', {'t': {}})
    with pytest.raises(AttributeError):
        db.insert('p', 'r', {'t': 5})


def test_textIndex_non_dict_config_defaults_to_weight_one():
    db = MemoryDb()
    db.textIndex('p', {'t': True})
    db.insert('p', 'r', {'t': 'a a'})
    assert db.table[0]['index_data'] == [('a', 2)]


def test_textIndex_merges_and_overwrites_config():
    db = MemoryDb()
    db.textIndex('p', {'t': {'weight': 1}})
    db.textIndex('p', {'u': {}})
    assert db.dumpindices('p') == {'t': {'weight': 1}, 'u': {}}
    db.textIndex('p', {'t': {'weight': 4}})
    db.insert('p', 'r', {'t': 'x'})
    assert db.table[0]['index_data'] == [('x', 4)]
    # Indexing is scoped per partition.
    assert db.dumpindices('q') is None


def test_textAnalysis_none_without_indexed_rows():
    assert MemoryDb().textAnalysis('p') is None
    db = MemoryDb()
    db.insert('p', 'r', {'t': 'red'})
    assert db.textAnalysis('p') is None


def test_textAnalysis_sum_spread_and_type():
    result = _docs_db().textAnalysis('docs')
    assert isinstance(result, FullTextAnalysis)
    assert set(result.terms()) == {'cat', 'dog', 'bird', 'fish'}
    data = result.dump()
    assert data['cat']['sum'] == 4
    assert data['dog']['sum'] == 7
    assert data['bird']['sum'] == 1
    assert data['fish']['sum'] == 2
    assert sorted(data['dog']['spread'].keys()) == [1, 2, 4]


def test_textAnalysis_ignores_other_partitions():
    db = _docs_db()
    db.textIndex('other', {'title': {}})
    db.insert('other', 'z', {'title': 'unicorn'})
    assert 'unicorn' not in db.textAnalysis('docs').terms()
    assert db.textAnalysis('other').terms() == ['unicorn']


def test_textAnalysis_skips_rows_inserted_before_index():
    db = MemoryDb()
    db.insert('p', 'early', {'t': 'red'})
    db.textIndex('p', {'t': {}})
    db.insert('p', 'late', {'t': 'red'})
    result = db.textAnalysis('p')
    assert result.all('red') == ['late']
    assert result.dump()['red']['sum'] == 1


def test_textAnalysis_chain_for_shared_token_and_score():
    result = _chain_db().textAnalysis('p')
    data = result.dump()
    # Three rows at score 1 contribute 1 + 1 + 1.
    assert data['red']['sum'] == 3
    assert data['blue']['sum'] == 1
    assert result.all('red') == ['x', 'y', 'z']
    assert result.top('red') == 'x'


def test_top_returns_head_of_highest_score():
    result = _docs_db().textAnalysis('docs')
    assert result.top('dog') == 'b'
    assert result.top('cat') == 'a'
    assert result.top('fish') == 'c'
    assert result.top('nothing') is None


def test_scores_sorted_descending_and_none_for_missing():
    result = _docs_db().textAnalysis('docs')
    assert result.scores('dog') == [4, 2, 1]
    assert result.scores('cat') == [3, 1]
    assert result.scores('missing') is None


def test_all_returns_each_matching_row_once():
    result = _docs_db().textAnalysis('docs')
    dog_rows = result.all('dog')
    assert len(dog_rows) == 3
    assert set(dog_rows) == {'a', 'b', 'c'}
    assert set(result.all('cat')) == {'a', 'b'}
    assert result.all('bird') == ['b']


def test_serialize_orders_by_score_then_chain():
    result = _docs_db().textAnalysis('docs')
    assert result.serialize(result.filter(lambda t: t == 'dog')) == ['b', 'a', 'c']
    assert result.serialize(result.filter(lambda t: t == 'cat')) == ['a', 'b']
    assert result.serialize(None) is None


def test_serialize_follows_chain_for_shared_score():
    result = _chain_db().textAnalysis('p')
    assert result.serialize(result.filter(lambda t: t == 'red')) == ['x', 'y', 'z']


def test_filter_requires_callable():
    result = _docs_db().textAnalysis('docs')
    assert result.filter('dog') == []
    assert set(result.filter(lambda t: t.startswith('d')).keys()) == {'dog'}
    assert result.filter(lambda t: False) == {}


def test_highlight_whole_word_and_case_sensitive():
    fta = FullTextAnalysis({})
    assert fta.highlight('cat', 'concat cat', '*{{_}}*') == 'concat *cat*'
    assert fta.highlight('cat', 'Cat cat', '[{{_}}]') == 'Cat [cat]'


def test_direct_construction_top_and_scores():
    data = {'x': {'sum': 5, 'spread': {5: {'value': 'r5'}, 2: {'value': 'r2'}}}}
    fta = FullTextAnalysis(data)
    assert fta.dump() is data
    assert fta.terms() == ['x']
    assert fta.scores('x') == [5, 2]
    assert fta.top('x') == 'r5'
    assert fta.top('y') is None
