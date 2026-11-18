import pytest
import pandas as pd
import numpy as np
from lark import Tree, Token
from genie_pkg.dqc import QualityChecker, GenieException

# Helper functions to build parse trees roughly matching the expected nodes
def make_node(data, children):
    return Tree(data, children)

def make_token(value):
    return Token('TOKEN', value)

def make_check_node(data, children):
    # These check nodes have a "data" name and list of children as tokens or Trees
    return Tree(data, children)

# Sample DataFrames for testing
df_basic = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'DateCol': ['2020-01-01', '2021-06-15', '2023-03-04', '2019-12-31'],
    'Value': ['abc', 'defg', 'hi', 'jklmno'],
    'Category': ['dog', 'cat', 'dog', 'cat']
})

df_with_nulls = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [None, 20, 30, np.nan],
    'StrCol': ['abc', '', '   ', None],
})

df_empty_cols = pd.DataFrame({
    'empty1': [np.nan, np.nan],
    'empty2': [np.nan, np.nan]
})

df_case = pd.DataFrame({
    'a': [1,2,3],
    'B': [4,5,6],
    'Cat': ['dog', 'cat', 'dog']
})

# ------------------------------
# Tests
# ------------------------------

def test_validate_raises_on_empty_columns_df():
    with pytest.raises(AttributeError):
        QualityChecker(df_empty_cols)

def test_init_does_not_fail_on_normal_df():
    checker = QualityChecker(df_basic)
    assert isinstance(checker, QualityChecker)
    assert checker.ignore_column_case is False

def test_treat_column_name_respects_ignore_case_flag():
    checker = QualityChecker(df_case)
    checker.ignore_column_case = True
    assert checker._treat_column_name("COL") == "col"
    checker.ignore_column_case = False
    assert checker._treat_column_name("COL") == "COL"

@pytest.mark.parametrize("comp, lhs, rhs, expected", [
    (">", 5, 3, True),
    ("==", 4, 4, True),
    ("==", 2, 3, False),
    ("<", 1, 2, True),
    ("foo", 123, 123, False),  # unknown comparator
])
def test_comparator_to_fn(comp, lhs, rhs, expected):
    fn = QualityChecker._comparator_to_fn(comp, rhs)
    assert fn(lhs) == expected

def test_apply_quantile_true_and_false():
    checker = QualityChecker(df_basic)
    # Create a node resembling: quantile("A", 0.5, ">", 2)
    # quantile median of column A is 2.5
    node = make_check_node("quantile", [
        "A",
        "0.5",
        ">",
        "2"
    ])
    result_name, result_bool = checker._apply_quantile(node)
    assert result_name == "quantile"
    assert result_bool is True  # median is 2.5 > 2

    # comparator >
    node_false = make_check_node("quantile", [
        "A",
        "0.5",
        "<",
        "2"
    ])
    result_name2, result_bool2 = checker._apply_quantile(node_false)
    assert result_name2 == "quantile"
    assert result_bool2 is False  # median 2.5 < 2 -> False

def test_apply_date_validation_success_and_failure():
    checker = QualityChecker(df_basic)
    # Valid date column
    node = make_check_node("is_date", [
        "DateCol"
    ])
    result_name, result_bool = checker._apply_date_validation(node)
    assert result_name == "is_date"
    assert result_bool is True

    # Invalid date column (non-date strings)
    wrong_df = pd.DataFrame({'bad_date': ['hello', 'world']})
    checker2 = QualityChecker(wrong_df)
    node2 = make_check_node("is_date", ["bad_date"])
    result_name2, result_bool2 = checker2._apply_date_validation(node2)
    assert result_bool2 is False
    assert result_name2.startswith("Date parse error:")

def test_apply_has_one_of_without_ignore_case():
    checker = QualityChecker(df_basic)
    # node.children = [column_name, children subtree of allowed values]
    allowed_vals_subtree = make_node("children", [make_token('"dog"'), make_token('"cat"')])
    node = make_check_node("has_one_of", [
        "Category",
        allowed_vals_subtree,
    ])
    name, result = checker._apply_has_one_of(node)
    assert name == "has_one_of"
    assert result is True  # df_unique ['dog', 'cat'] are all in allowed list

def test_apply_has_one_of_with_ignore_case():
    checker = QualityChecker(df_basic)
    allowed_vals_subtree = make_node("children", [make_token('"DOG"'), make_token('"CAT"')])
    node = make_check_node("has_one_of", [
        "Category",
        allowed_vals_subtree,
        "True"  # ignore_case flag
    ])
    name, result = checker._apply_has_one_of(node)
    assert result is True

def test_apply_has_one_of_fails_when_unallowed_values_present():
    checker = QualityChecker(df_basic)
    allowed_vals_subtree = make_node("children", [make_token('"dog"')])
    node = make_check_node("has_one_of", [
        "Category",
        allowed_vals_subtree
    ])
    _, result = checker._apply_has_one_of(node)
    assert result is False  # 'cat' not in allowed_vals

def test_apply_value_length_without_ignore_nulls():
    df = pd.DataFrame({'col': ['aaaa', 'bbbb', 'cccc']})
    checker = QualityChecker(df)
    node = make_check_node("value_length", [
        "col",
        "==",
        "4"
    ])
    name, result = checker._apply_value_length(node)
    assert name == "value_length"
    assert result is True  # all length 4 and equal comparator 4

def test_apply_value_length_with_ignore_nulls_true():
    df = pd.DataFrame({'col': ['aaa', '', '   ', None, 'b']})
    checker = QualityChecker(df)
    # node with ignore_nulls='True', comparator='==', rhs=3
    node = make_check_node("value_length", [
        "col",
        "True",
        "==",
        "3"
    ])
    # After dropping null/empty, lengths are [3]
    name, result = checker._apply_value_length(node)
    assert name == "value_length"
    assert result is True

def test_apply_value_length_with_ignore_nulls_false_and_variable_lengths():
    df = pd.DataFrame({'col': ['aa', 'bbb', 'cccc']})
    checker = QualityChecker(df)
    node = make_check_node("value_length", [
        "col",
        "False",
        "==",
        "3"
    ])
    name, res = checker._apply_value_length(node)
    # unique lengths = {2,3,4} → len(unique_lengths)==3 → should be False
    assert name == "value_length"
    assert res is False

def test_apply_percent_value_length_basic_true():
    df = pd.DataFrame({'col': ['a'*3, 'b'*3, 'c'*3, None]})
    checker = QualityChecker(df)
    # Without ignore_nulls, percent 75, rhs=3 length
    node = make_check_node("percent_value_length", [
        "col",
        "75",
        "False",
        "3"
    ])
    name, res = checker._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert res is True  # 3 out of 4 rows have length 3, 75% met

def test_apply_percent_value_length_with_ignore_nulls_true():
    df = pd.DataFrame({'col': ['a'*3, 'b'*3, 'c'*3, None]})
    checker = QualityChecker(df)
    # With ignore_nulls True, percent 100, rhs=3 length
    node = make_check_node("percent_value_length", [
        "col",
        "100",
        "True",
        "False",
        "3"
    ])
    name, res = checker._apply_percent_value_length(node)
    assert name == "percent_value_length"
    # only consider non-null rows, all have length 3 -> 100% pass
    assert res is True

def test_apply_has_columns_exact_match_and_ignore_case():
    df = pd.DataFrame(columns=['col1','col2','col3'])
    checker = QualityChecker(df)
    cols_subtree = make_node("children", [make_token('"col1"'), make_token('"col3"')])
    node = make_check_node("has_columns", [cols_subtree])
    name, res = checker._apply_has_columns(node)
    assert name == "has_columns"
    assert res is True

    # Now with ignore_case = True but column names upper case in node
    cols_subtree2 = make_node("children", [make_token('"COL1"'), make_token('"COL3"')])
    node2 = make_check_node("has_columns", [cols_subtree2, "True"])
    name2, res2 = checker._apply_has_columns(node2)
    assert res2 is True

def test_apply_check_row_count_success_and_failure():
    df = pd.DataFrame({'x':[1,2,3]})
    checker = QualityChecker(df)
    # row_count > 2 (3 rows > 2)
    node = make_check_node("row_count", [" >", "2"])
    result_name, result_bool = checker._apply_check([node])
    assert result_name == "row_count"
    assert result_bool is True

    # row_count < 2 (3 rows < 2) should be False
    node2 = make_check_node("row_count", ["<", "2"])
    res_name2, res_bool2 = checker._apply_check([node2])
    assert res_name2 == "row_count"
    assert res_bool2 is False

def test_apply_check_not_null_and_is_unique():
    df = pd.DataFrame({'a':[1,2,3], 'b':[1,1,2]})
    checker = QualityChecker(df)
    node_unique = make_check_node("is_unique", ["a"])
    name, res = checker._apply_check([node_unique])
    assert name == "is_unique"
    assert res is True

    node_not_null = make_check_node("not_null", ["a"])
    name2, res2 = checker._apply_check([node_not_null])
    assert name2 == "not_null"
    assert res2 is True

    node_not_null_with_nan = make_check_node("not_null", ["b"])
    df_nan = pd.DataFrame({'b':[1, None, 2]})
    checker_nan = QualityChecker(df_nan)
    name3, res3 = checker_nan._apply_check([node_not_null_with_nan])
    assert name3 == "not_null"
    assert res3 is False

def test_apply_check_is_positive_true_and_false():
    df = pd.DataFrame({'a':[1,2,3], 'b':[-1,0,1]})
    checker = QualityChecker(df)
    node_pos = make_check_node("is_positive", ["a"])
    name, res = checker._apply_check([node_pos])
    assert name == "is_positive"
    assert res is True

    node_pos2 = make_check_node("is_positive", ["b"])
    name2, res2 = checker._apply_check([node_pos2])
    assert name2 == "is_positive"
    assert res2 is False

def test_apply_check_unsupported_raises():
    checker = QualityChecker(df_basic)
    node = make_check_node("unknown_check", ["A"])
    with pytest.raises(GenieException):
        checker._apply_check([node])

def test_apply_check_with_missing_column_returns_false():
    df = pd.DataFrame({'a':[1,2]})
    checker = QualityChecker(df)
    node = make_check_node("not_null", ["missing_col"])
    name, res = checker._apply_check([node])
    assert res is False
    assert name.startswith("Key error")

def test_apply_predicates_runs_code_block():
    checker = QualityChecker(df_basic)
    # predicate.data == "code_block"
    # It runs checks (list of checks) in predicate.children: a list of Trees, each with .children list
    check_node1 = make_check_node("row_count", [">", "0"])
    check_node2 = make_check_node("not_null", ["A"])
    code_block = make_node("code_block", [make_node("check", [check_node1]), make_node("check", [check_node2])])
    results = checker._apply_predicates([code_block])
    # results inserted in reverse order (insert(0, v))
    assert isinstance(results, list)
    assert any(r[0] == "row_count" for r in results)
    assert any(r[0] == "not_null" for r in results)

def test_apply_predicates_raises_unknown_instruction():
    checker = QualityChecker(df_basic)
    unknown_predicate = make_node("unknown", [])
    with pytest.raises(GenieException):
        checker._apply_predicates([unknown_predicate])

def test__run_accepts_predicates_and_raises_other():
    checker = QualityChecker(df_basic)
    # _run expects Tree with exactly one child

    # Normal case
    check_node1 = make_check_node("row_count", [">", "0"])
    code_block = make_node("code_block", [make_node("check", [check_node1])])
    predicates = make_node("predicates", [code_block])
    pt = make_node("start", [predicates])  # root

    results = checker._run(pt)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"

    # Wrong number of children
    pt2 = make_node("start", [predicates, predicates])
    with pytest.raises(GenieException):
        checker._run(pt2)

    # Unknown instruction in root child
    wrong_inst = make_node("bad_instruction", [])
    pt3 = make_node("start", [wrong_inst])
    with pytest.raises(GenieException):
        checker._run(pt3)

def test_validate_spec_returns_true_for_valid_and_false_for_invalid():
    good_spec = 'row_count > 0'
    err, valid = QualityChecker.validate_spec(good_spec)
    assert valid is True
    assert err is None

    bad_spec = 'row_count >>> 0'
    err2, valid2 = QualityChecker.validate_spec(bad_spec)
    assert valid2 is False
    assert isinstance(err2, str)

def test_run_changes_ignore_case_and_runs_checks(monkeypatch):
    df = pd.DataFrame({
        'Col': [1, 1, 1]
    })
    checker = QualityChecker(df)
    spec = 'is_unique("col")'  # lowercase col but actual col is 'Col'

    # monkeypatch _parse_spec to a dummy returning a Tree with 'is_unique' check
    # Structure needed for _run: Tree with one child 'predicates' containing a code_block whose children contain checks
    check_node = make_check_node("is_unique", ["col"])
    check_tree = make_node("check", [check_node])
    code_block = make_node("code_block", [check_tree])
    predicates = make_node("predicates", [code_block])
    pt = make_node("start", [predicates])

    monkeypatch.setattr(QualityChecker, "_parse_spec", lambda self, spec_: pt)
    result = checker.run(spec, ignore_column_case=True)
    # The column case is ignored, so should pick up 'Col' as 'col'
    assert isinstance(result, list)
    assert result[0][0] == "is_unique"
    assert result[0][1] is True
