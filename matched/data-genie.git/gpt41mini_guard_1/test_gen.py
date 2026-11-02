import pytest
import numpy as np
import pandas as pd
from lark import Tree

from genie_pkg.dqc import QualityChecker, GenieException


def make_node(data, children):
    # Helper to make a minimal lark.Tree node usable for tests
    return Tree(data=data, children=children)


def test_validate_raises_on_empty_dataframe():
    df_empty = pd.DataFrame()
    with pytest.raises(AttributeError):
        QualityChecker._validate(df_empty)

    df_all_nan = pd.DataFrame({'a': [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker._validate(df_all_nan)


def test_comparator_to_fn():
    gt = QualityChecker._comparator_to_fn('>', 5)
    eq = QualityChecker._comparator_to_fn('==', 5)
    lt = QualityChecker._comparator_to_fn('<', 5)
    other = QualityChecker._comparator_to_fn('!=', 5)

    assert gt(6) is True
    assert gt(5) is False
    assert eq(5) is True
    assert eq(4) is False
    assert lt(4) is True
    assert lt(5) is False
    assert other(10) is False


def test_treat_column_name_respects_ignore_case():
    qc = QualityChecker(pd.DataFrame({'A': [1]}))
    qc.ignore_column_case = False
    assert qc._treat_column_name('XyZ') == 'XyZ'
    qc.ignore_column_case = True
    assert qc._treat_column_name('XyZ') == 'xyz'


def test_apply_quantile_basic():
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)

    # Make node with children: col name, quantile, comparator, rhs
    node = make_node('quantile', ['col1', '0.5', '>', '3'])
    key, result = qc._apply_quantile(node)
    assert key == 'quantile'
    # Median is 3, here 3>3 is False
    assert result is False

    node2 = make_node('quantile', ['col1', '0.5', '<', '4'])
    key2, result2 = qc._apply_quantile(node2)
    assert result2 is True


def test_apply_date_validation_valid_and_invalid():
    df = pd.DataFrame({'dates': ['2021-01-01', '2020-05-05']})
    qc = QualityChecker(df)
    node = make_node('is_date', ['dates'])
    key, result = qc._apply_date_validation(node)
    assert key == 'is_date'
    assert result is True

    df_bad = pd.DataFrame({'dates': ['bad-date', '2020-01-01']})
    qc_bad = QualityChecker(df_bad)
    node = make_node('is_date', ['dates'])
    key_bad, result_bad = qc_bad._apply_date_validation(node)
    # Because pd.to_datetime on good parseable strings works, but here we have bad string, 
    # in fact pd.to_datetime will raise exception if any value fails when called on Series with errors='raise'
    assert result_bad is False
    assert key_bad.startswith('Date parse error')


def test_apply_has_one_of_ignore_case_flag():
    df = pd.DataFrame({
        'col': ['A', 'b', 'C', 'a', 'B']
    })
    qc = QualityChecker(df)
    # Node children: column_name, Tree(children of allowed values), optional flag 'True' or 'False'
    allowed_values_tree = Tree('list', [Tree('value', ['"a"']), Tree('value', ['"b"'])])
    node = make_node('has_one_of', ['col', allowed_values_tree])
    key, res = qc._apply_has_one_of(node)
    assert key == 'has_one_of'
    # Unique values are A,B,C,a,b; not all are in {"a","b"} => False
    assert res is False

    node_ic = make_node('has_one_of', ['col', allowed_values_tree, 'True'])
    key2, res2 = qc._apply_has_one_of(node_ic)
    # Ignore case -> allowed_values = [a,b], unique_values lower: [a,b,c] => 'c' not in allowed_values => False
    assert res2 is False

    # Now test with only allowed values present
    df2 = pd.DataFrame({'col': ['a', 'A', 'b', 'B']})
    qc2 = QualityChecker(df2)
    node2 = make_node('has_one_of', ['col', allowed_values_tree])
    key3, res3 = qc2._apply_has_one_of(node2)
    assert res3 is False  # Because of case sensitive mismatch: 'A' not in ['a','b']

    node3 = make_node('has_one_of', ['col', allowed_values_tree, 'True'])
    key4, res4 = qc2._apply_has_one_of(node3)
    assert res4 is True


def test_apply_value_length_basic_ignore_nulls():
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'ccc']})
    qc = QualityChecker(df)

    # children: col, comparator, rhs
    node = make_node('value_length', ['col', '==', '3'])
    key, res = qc._apply_value_length(node)
    assert key == 'value_length'
    assert res is True

    # Lengths differ
    df2 = pd.DataFrame({'col': ['aa', 'bbb', 'ccc']})
    qc2 = QualityChecker(df2)
    node2 = make_node('value_length', ['col', '==', '3'])
    _, res2 = qc2._apply_value_length(node2)
    assert res2 is False

    # With ignore_nulls = True, and nulls are present as empty string or spaces
    df3 = pd.DataFrame({'col': ['aa', '   ', 'ccc']})
    qc3 = QualityChecker(df3)

    node3 = make_node('value_length', ['col', 'True', '==', '3'])
    _, res3 = qc3._apply_value_length(node3)
    # Non-null entries lengths are 2 and 3, so unique_lengths = [2,3] => fail
    assert res3 is False

    # If all real values same length ignoring nulls
    df4 = pd.DataFrame({'col': ['aaa', '   ', 'aaa']})
    qc4 = QualityChecker(df4)
    node4 = make_node('value_length', ['col', 'True', '==', '3'])
    _, res4 = qc4._apply_value_length(node4)
    assert res4 is True


def test_apply_percent_value_length_basic():
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'ccc', 'dd']})
    qc = QualityChecker(df)

    # children length 4 without ignore_nulls param: [col, percent, somevalue, rhs]
    node = make_node('percent_value_length', ['col', '75', 'foo', '3'])
    key, result = qc._apply_percent_value_length(node)
    assert key == 'percent_value_length'
    # lengths: 3,3,3,2; length==3 count=3; percent=3/4=75%
    assert result is True

    # children length 5 with ignore_nulls param: col, percent, ignore_nulls, somevalue, rhs
    df2 = pd.DataFrame({'col': ['aaa', '', 'ccc', 'dd']})
    qc2 = QualityChecker(df2)
    node2 = make_node('percent_value_length', ['col', '50', 'True', 'bar', '3'])
    key2, result2 = qc2._apply_percent_value_length(node2)
    # ignore_nulls True means dropna after converting empty string to NaN => data considered: 'aaa', 'ccc', 'dd'
    # lengths: 3,3,2; length==3 count=2; 2/4=50%, so passes the 50% threshold
    assert result2 is True

    # Check fails if percent is not met
    node3 = make_node('percent_value_length', ['col', '80', 'False', 'baz', '3'])
    _, res3 = qc2._apply_percent_value_length(node3)
    assert res3 is False


def test_apply_has_columns_ignore_case_various():
    df = pd.DataFrame(columns=['Name', 'Age', 'City'])
    qc = QualityChecker(df)

    allowed_tree = Tree('list', [Tree('value', ['"Name"']), Tree('value', ['"Age"'])])
    node = make_node('has_columns', [allowed_tree])
    key, result = qc._apply_has_columns(node)
    assert key == 'has_columns'
    assert result is True

    node2 = make_node('has_columns', [allowed_tree, 'True'])
    key2, result2 = qc._apply_has_columns(node2)
    assert result2 is True

    # Missing column
    allowed_tree_missing = Tree('list', [Tree('value', ['"Name"']), Tree('value', ['"Salary"'])])
    node3 = make_node('has_columns', [allowed_tree_missing])
    _, res3 = qc._apply_has_columns(node3)
    assert res3 is False

    node4 = make_node('has_columns', [allowed_tree_missing, 'True'])
    _, res4 = qc._apply_has_columns(node4)
    assert res4 is False


def test_apply_check_row_count_and_keyerror():
    df = pd.DataFrame({'a': [1, 2, 3]})
    qc = QualityChecker(df)

    # row_count check: children [comparator, quantity]
    c = make_node('row_count', ['>', '2'])
    key, result = qc._apply_check([c])
    assert key == 'row_count'
    assert result is True

    c_eq = make_node('row_count', ['==', '4'])
    key2, result2 = qc._apply_check([c_eq])
    assert result2 is False

    # check for KeyError in a column
    c_missing = make_node('is_unique', ['missing_col'])
    key3, res3 = qc._apply_check([c_missing])
    assert key3.startswith('Key error:')
    assert res3 is False


def test_apply_check_various_checks():
    df = pd.DataFrame({'col': [1, 2, 3], 'col2': [10, 10, 10], 'col3': [0, 1, 2], 'col4': ['a', 'a', 'a']})
    qc = QualityChecker(df)

    # is_unique true for col2 (all 10s) => False, actually col2.is_unique is False because values duplicated
    c_unique = make_node('is_unique', ['col'])
    key, res = qc._apply_check([c_unique])
    assert key == 'is_unique'
    assert res is True  # col: 1,2,3 unique values

    c_unique2 = make_node('is_unique', ['col2'])
    _, res2 = qc._apply_check([c_unique2])
    assert res2 is False

    # not_null: col no nulls
    c_not_null = make_node('not_null', ['col'])
    _, res3 = qc._apply_check([c_not_null])
    assert res3 is True

    # not_null with null
    df2 = pd.DataFrame({'col': [1, None, 3]})
    qc2 = QualityChecker(df2)
    c_not_null2 = make_node('not_null', ['col'])
    _, res4 = qc2._apply_check([c_not_null2])
    assert res4 is False

    # is_positive all >0 for col
    c_pos = make_node('is_positive', ['col'])
    _, res5 = qc._apply_check([c_pos])
    assert res5 is True

    # is_positive fails for col3 (0,1,2)
    c_pos2 = make_node('is_positive', ['col3'])
    _, res6 = qc._apply_check([c_pos2])
    assert res6 is False

    # has_one_of, quantile, is_date tested indirectly in other tests

    # unknown check type: raise GenieException
    c_unknown = make_node('unknown_check', [])
    with pytest.raises(GenieException):
        qc._apply_check([c_unknown])


def test_apply_predicates_code_block_and_exception():
    df = pd.DataFrame({'col': [1, 2, 3]})
    qc = QualityChecker(df)

    check1 = make_node('row_count', ['>', '2'])
    check2 = make_node('not_null', ['col'])
    code_block = make_node('code_block', [make_node('check', [check1]), make_node('check', [check2])])

    result = qc._apply_predicates([code_block])
    # Results are inserted at front, so last check first
    assert len(result) == 2
    assert result[0][0] == 'not_null'
    assert result[1][0] == 'row_count'

    # Predicate with unknown instruction
    unknown_pred = make_node('unknown', [])
    with pytest.raises(GenieException):
        qc._apply_predicates([unknown_pred])


def test_run_with_single_predicate_children():
    df = pd.DataFrame({'col': [1, 2]})
    qc = QualityChecker(df)

    check = make_node('row_count', ['>', '1'])
    check_wrapped = make_node('check', [check])
    predicate = make_node('predicates', [make_node('code_block', [check_wrapped])])
    tree = make_node('root', [predicate])

    # _run expects exactly one child with data "predicates"
    res = qc._run(predicate)
    assert isinstance(res, list)
    assert res[0][0] == 'row_count'

    # Wrong number of children => exception
    tree_multiple = make_node('root', [predicate, predicate])
    with pytest.raises(GenieException):
        qc._run(tree_multiple)

    # Unknown instruction in child
    bad_pred = make_node('bad', [])
    bad = make_node('root', [bad_pred])
    with pytest.raises(GenieException):
        qc._run(bad)


def test_parse_spec_and_validate_spec_valid_and_invalid():
    # We can only test with a known correct spec string present in grammar understanding
    # Minimal valid: we rely on QualityChecker.validate_spec giving no error on a simple spec
    valid_spec = "row_count > 0"
    err_msg, valid = QualityChecker.validate_spec(valid_spec)
    assert valid is True
    assert err_msg is None

    # Invalid spec string
    invalid_spec = "row_count >> 0"
    err_msg2, valid2 = QualityChecker.validate_spec(invalid_spec)
    assert valid2 is False
    assert err_msg2 is not None


def test_run_with_ignore_column_case_changes_columns():
    df = pd.DataFrame({'COL': [1, 2, 3], 'a': [4, 5, 6]})
    qc = QualityChecker(df.copy())

    spec = "row_count > 2"
    results = qc.run(spec, ignore_column_case=True)
    # After run, columns converted to lowercase
    assert all(isinstance(col, str) and col.islower() for col in qc._obj.columns)
    assert any(res[0] == 'row_count' for res in results)


def test_run_raises_on_invalid_spec():
    df = pd.DataFrame({'col': [1, 2]})
    qc = QualityChecker(df)

    with pytest.raises(Exception):
        qc.run("row_count >> 10")

