import pytest
import pandas as pd
import numpy as np
from lark import Tree
from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


# Helpers for constructing dummy parse tree nodes for tests
class DummyNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children


def make_tree(data, children):
    return Tree(data, children)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'a', 'c', 'b'],
        'col3': ['2020-01-01', '2020-02-02', '2020-03-03', None, '2020-05-05'],
        'col4': ['foo', 'bar', '', 'baz', 'qux'],
        'COL5': [5, 5, 5, 5, 5],
        'col6': [0, 1, 2, 3, 4],
        'col7': [None, 'a', 'b', 'c', None],
    })

@pytest.fixture
def qc(sample_df):
    return QualityChecker(sample_df.copy())


def test_validate_rejects_empty_dataframe():
    df = pd.DataFrame({'a': [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker._validate(df)


def test_comparator_to_fn():
    f_gt = QualityChecker._comparator_to_fn('>', 3)
    assert f_gt(4) is True
    assert f_gt(3) is False
    f_eq = QualityChecker._comparator_to_fn('==', 3)
    assert f_eq(3) is True
    assert f_eq(2) is False
    f_lt = QualityChecker._comparator_to_fn('<', 3)
    assert f_lt(2) is True
    assert f_lt(3) is False
    f_unknown = QualityChecker._comparator_to_fn('!=', 3)
    assert f_unknown(3) is False


def test_treat_column_name_case_sensitive(qc):
    qc.ignore_column_case = False
    assert qc._treat_column_name('Col1') == 'Col1'
    qc.ignore_column_case = True
    assert qc._treat_column_name('Col1') == 'col1'


def test_apply_quantile_true(qc):
    # Using quantile 0.5 of col1, comparator > 2
    node = DummyNode(
        'quantile',
        ['col1', '0.5', '>', '2']
    )
    result = qc._apply_quantile(node)
    assert result[0] == 'quantile'
    # median of 1..5 = 3, 3 > 2 is True
    assert result[1] is True

def test_apply_quantile_false(qc):
    # quantile 0.5 col1, comparator < 2
    node = DummyNode(
        'quantile',
        ['col1', '0.5', '<', '2']
    )
    name, res = qc._apply_quantile(node)
    assert name == "quantile"
    assert res is False  # median 3 < 2? no

def test_apply_date_validation_correct(qc):
    node = DummyNode(
        'is_date',
        ['col3']
    )
    name, res = qc._apply_date_validation(node)
    assert name == 'is_date'
    assert res is True

def test_apply_date_validation_error(qc):
    qc._obj['bad_date'] = ['x', 'y', 'z', 'a', 'b']
    node = DummyNode('is_date', ['bad_date'])
    name, res = qc._apply_date_validation(node)
    assert res is False
    assert "Date parse error" in name

def test_apply_has_one_of_no_ignore_case_true(qc):
    node = DummyNode(
        'has_one_of',
        ['col2', make_tree('list', [DummyNode('value', ['"a"']), DummyNode('value', ['"b"'])])]
    )
    name, res = qc._apply_has_one_of(node)
    assert name == 'has_one_of'
    # col2 unique vals are ['a','b','c'], c not in allowed -> False
    assert res is False

def test_apply_has_one_of_with_ignore_case_true(qc):
    # alter col2 to ['A', 'B', 'C'] to test ignore_case=True
    qc._obj['col2'] = ['A', 'B', 'A', 'C', 'B']
    node = DummyNode(
        'has_one_of',
        ['col2',
         make_tree('list', [DummyNode('value', ['"a"']), DummyNode('value', ['"b"'])]),
         'True']
    )
    name, res = qc._apply_has_one_of(node)
    assert name == 'has_one_of'
    # 'C' lowered to 'c', unique vals include 'c' not in allowed -> False
    assert res is False

def test_apply_value_length_ignore_nulls_false(qc):
    # All values length 1 or more in col4, comparator == 3 (false)
    node = DummyNode('value_length', ['col4', '==', '3'])
    name, res = qc._apply_value_length(node)
    assert name == 'value_length'
    assert res is False

def test_apply_value_length_ignore_nulls_true(qc):
    # Set col4 values so that ignoring nulls matters
    df = qc._obj.copy()
    df['col4'] = ['a', '', '', 'b', 'c']
    qc._obj = df
    node = DummyNode('value_length', ['col4', 'True', '==', '1'])
    name, res = qc._apply_value_length(node)
    assert name == 'value_length'
    assert res is True  # All non-null values length 1

def test_apply_percent_value_length_ignore_nulls_false(qc):
    # length( col4 ) == 3 for percent=40, rhs = 3, comparator >= percent
    # col4 length: ['foo'(3), 'bar'(3), ''(0), 'baz'(3), 'qux'(3)] -> 4/5 = 80% lengths ==3
    node = DummyNode('percent_value_length', ['col4', '40', 'False', '==', '3'])
    name, res = qc._apply_percent_value_length(node)
    assert name == 'percent_value_length'
    assert res is True

def test_apply_percent_value_length_ignore_nulls_true(qc):
    df = qc._obj.copy()
    df['col4'] = ['foo', '', '', 'baz', 'qux']
    qc._obj = df
    # With ignore_nulls True, drop empty string rows; length == 3 for 3/3 = 100%
    node = DummyNode('percent_value_length', ['col4', '90', 'True', '==', '3'])
    name, res = qc._apply_percent_value_length(node)
    assert name == 'percent_value_length'
    assert res is True

def test_apply_has_columns_case_sensitive(qc):
    node = DummyNode('has_columns', [
        make_tree('list', [DummyNode('value', ['"col1"']), DummyNode('value', ['"col2"'])])
    ])
    name, res = qc._apply_has_columns(node)
    assert name == 'has_columns'
    assert res is True

def test_apply_has_columns_case_insensitive(qc):
    node = DummyNode('has_columns', [
        make_tree('list', [DummyNode('value', ['"col1"']), DummyNode('value', ['"col2"'])]),
        'True'
    ])
    name, res = qc._apply_has_columns(node)
    assert name == 'has_columns'
    assert res is True

def test_apply_check_row_count(qc):
    node = DummyNode('row_count', ['>', '3'])
    result = qc._apply_check([node])
    assert result[0] == 'row_count'
    assert result[1] is True

def test_apply_check_has_columns(qc):
    node = DummyNode('has_columns', [
        make_tree('list', [DummyNode('value', ['"col1"'])])
    ])
    name, flag = qc._apply_check([node])
    assert name == 'has_columns'
    assert flag is True

def test_apply_check_is_unique_true(qc):
    # col5 is all 5 -> not unique
    node = DummyNode('is_unique', ['COL5'])
    name, res = qc._apply_check([node])
    assert name == 'is_unique'
    assert res is False  # Values repeated

def test_apply_check_not_null(qc):
    # col1 has no nulls
    node = DummyNode('not_null', ['col1'])
    name, res = qc._apply_check([node])
    assert name == 'not_null'
    assert res is True

def test_apply_check_not_null_false(qc):
    # col7 has nulls
    node = DummyNode('not_null', ['col7'])
    name, res = qc._apply_check([node])
    assert name == 'not_null'
    assert res is False

def test_apply_check_is_positive(qc):
    node = DummyNode('is_positive', ['col6'])
    name, res = qc._apply_check([node])
    assert name == 'is_positive'
    assert res is False  # col6 has zero which is not >0

def test_apply_check_has_one_of(qc):
    node = DummyNode('has_one_of', ['col2',
                                   make_tree('list', [DummyNode('value', ['"a"']), DummyNode('value', ['"b"'])])
                                  ])
    name, res = qc._apply_check([node])
    assert name == 'has_one_of'
    assert res is False

def test_apply_check_quantile(qc):
    node = DummyNode('quantile', ['col1', '0.5', '>', '2'])
    name, res = qc._apply_check([node])
    assert name == 'quantile'
    assert res is True

def test_apply_check_is_date(qc):
    node = DummyNode('is_date', ['col3'])
    name, res = qc._apply_check([node])
    assert name == 'is_date'
    assert res is True

def test_apply_check_value_length(qc):
    node = DummyNode('value_length', ['col4', '==', '3'])
    name, res = qc._apply_check([node])
    assert name == 'value_length'
    assert not res

def test_apply_check_percent_value_length(qc):
    node = DummyNode('percent_value_length', ['col4', '40', 'False', '==', '3'])
    name, res = qc._apply_check([node])
    assert name == 'percent_value_length'


def test_apply_check_unknown_raises(qc):
    node = DummyNode('nonexistent_check', ['col1'])
    with pytest.raises(GenieException):
        qc._apply_check([node])

def test_apply_check_key_error(qc):
    node = DummyNode('not_null', ['nonexistent_column'])
    name, res = qc._apply_check([node])
    assert res is False
    assert "Key error" in name

def test_apply_predicates_accepts_code_block(qc):
    check_node = DummyNode('row_count', ['>', '1'])
    predicate = DummyNode('code_block', [check_node])
    results = qc._apply_predicates([predicate])
    assert isinstance(results, list)
    assert results[0][0] == 'row_count'
    assert results[0][1] is True

def test_apply_predicates_unknown_instruction(qc):
    predicate = DummyNode('unknown', [])
    with pytest.raises(GenieException):
        qc._apply_predicates([predicate])

def test_run_single_predicates_node(qc):
    check_node = DummyNode('row_count', ['>', '1'])
    predicates = DummyNode('predicates', [DummyNode('code_block', [check_node])])
    result = qc._run(predicates)
    assert isinstance(result, list)
    assert result[0][0] == 'row_count'
    assert result[0][1] is True

def test_run_raises_for_multiple_children(qc):
    t = make_tree('root', [DummyNode('predicates', []), DummyNode('predicates', [])])
    with pytest.raises(GenieException):
        qc._run(t)

def test_run_raises_unknown_instruction(qc):
    t = make_tree('root', [DummyNode('unknown', [])])
    with pytest.raises(GenieException):
        qc._run(t)

def test_validate_spec_valid():
    spec = 'row_count > 0'
    msg, valid = QualityChecker.validate_spec(spec)
    assert valid is True
    assert msg is None

def test_validate_spec_invalid():
    spec = 'row_count >> 0'
    msg, valid = QualityChecker.validate_spec(spec)
    assert valid is False
    assert isinstance(msg, str)

def test_run_method_basic(sample_df):
    qc = QualityChecker(sample_df.copy())
    spec = 'row_count > 3'
    results = qc.run(spec)
    assert isinstance(results, list)
    assert results[0][0] == 'row_count'
    assert results[0][1] is True

def test_run_method_ignore_column_case(sample_df):
    df = sample_df.copy()
    df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    qc = QualityChecker(df)
    spec = 'has_columns ["a", "b"] True'
    results = qc.run(spec, ignore_column_case=True)
    assert any(result[0] == 'has_columns' for result in results)
    # Columns lowered so condition met
    assert all('a' in map(str.lower, df.columns))

