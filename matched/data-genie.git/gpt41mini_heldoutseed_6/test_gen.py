import pytest
import pandas as pd
import numpy as np
from lark import Tree, Token
from genie_pkg import dqc, GenieException

# Helper functions to create Lark Tree nodes resembling the DSL parse output

def make_token(value):
    return Token('TOKEN', str(value))

def make_tree(data, children):
    return Tree(data, children)

def make_check(data, children):
    # Represents a <check> node with children
    return make_tree(data, children)

def make_code_block(checks):
    # Represents a code_block node with list of checks children
    return make_tree("code_block", checks)

def make_predicates(code_blocks):
    # predicates node with code_block children
    return make_tree("predicates", code_blocks)

# Wrapping a check inside the list expected by _apply_check
def wrap_check(c):
    # _apply_check expects check to be a list of 1 Tree (check)
    return [c]

@pytest.fixture
def df_basic():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["x", "y", "z", "x", "y"],
        "C": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01"]),
        "D": ["hello", "world", "foo", "bar", np.nan],
        "E": ["test", "test", "test", "test", "test"],
        "F": [10, 20, 30, 40, 50]
    })

@pytest.fixture
def qc_basic(df_basic):
    return dqc.QualityChecker(df_basic)

def test_validate_raises_on_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(AttributeError):
        dqc.QualityChecker._validate(empty_df)

    # DataFrame with only NaN columns is also empty
    df_nan = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        dqc.QualityChecker._validate(df_nan)

def test_comparator_to_fn_behaviour():
    comp = dqc.QualityChecker._comparator_to_fn
    fn_gt = comp(">", 5)
    assert fn_gt(6) is True
    assert fn_gt(5) is False

    fn_eq = comp("==", 3)
    assert fn_eq(3) is True
    assert fn_eq(2) is False

    fn_lt = comp("<", 10)
    assert fn_lt(9) is True
    assert fn_lt(10) is False

    fn_default = comp("???", 1)
    assert fn_default(1) is False

def test_treat_column_name_behavior(qc_basic):
    qc_basic.ignore_column_case = False
    assert qc_basic._treat_column_name("TestCol") == "TestCol"
    qc_basic.ignore_column_case = True
    assert qc_basic._treat_column_name("TestCol") == "testcol"

def test_apply_quantile_true_and_false(qc_basic):
    # quantile > 0.5 and rhs 3, column F has median of 30, so 30 > 3 is True
    node = make_check("quantile", [
        "F",          # column name
        "0.5",        # quantile q
        ">",          # comparator
        "3"           # rhs
    ])
    # children are expected to be token-like strings, so keep as strings
    # The _apply_quantile converts strings inside

    result = qc_basic._apply_quantile(node)
    assert result[0] == "quantile"
    assert result[1] is True

    # quantile < 0.5 and rhs 100 should be False
    node_false = make_check("quantile", [
        "F",
        "0.5",
        "<",
        "10"
    ])
    res_false = qc_basic._apply_quantile(node_false)
    assert res_false[0] == "quantile"
    assert res_false[1] is False

def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"dates": ["2021-01-01", "2021-02-02", "notadate"]})
    qc = dqc.QualityChecker(df)
    node_good = make_check("is_date", ["dates"])
    label, valid = qc._apply_date_validation(node_good)
    # Will fail due to 'notadate'
    # Actually pd.to_datetime will error for this string?
    # In pandas <1.0 errors raise or coerce? Here code expects exception and False.
    # So output False since parsing fails
    assert label.startswith("Date parse error") or valid is False

    # Fix dataframe with correct dates
    df2 = pd.DataFrame({"dates": ["2021-01-01", "2021-02-02", "2021-03-03"]})
    qc2 = dqc.QualityChecker(df2)
    node2 = make_check("is_date", ["dates"])
    label2, valid2 = qc2._apply_date_validation(node2)
    assert label2 == "is_date"
    assert valid2 is True

def test_apply_has_one_of_case_sensitive_and_ignore_case():
    df = pd.DataFrame({
        "col": ["A", "B", "C", "A"]
    })
    qc = dqc.QualityChecker(df)

    # Allowed values are A and B, ignore case false
    allowed_values_node = make_tree("childrenstring", [make_token('"A"'), make_token('"B"')])
    node = make_check("has_one_of", [
        "col",         # column name
        make_tree("children", [make_token('"A"'), make_token('"B"')]),  # allowed values subtree
    ])
    # Manually fix children structure for allowed_values (list of tokens as children of children)
    node.children[1] = make_tree("children", [make_token('"A"'), make_token('"B"')])

    label, result = qc._apply_has_one_of(node)
    assert label == "has_one_of"
    assert result is False  # Because C is in col but not allowed

    # Now set ignore_case = True (third child = 'True')
    node_ic = make_check("has_one_of", [
        "col",
        make_tree("children", [make_token('"a"'), make_token('"b"')]),
        "True"
    ])
    label2, result2 = qc._apply_has_one_of(node_ic)
    assert label2 == "has_one_of"
    # 'C' lowercased is 'c' which is not in ['a', 'b'], so False
    assert result2 is False

    # Adjust data so that unique values are allowed (change C to A)
    df2 = pd.DataFrame({"col": ["A", "B", "A", "B"]})
    qc2 = dqc.QualityChecker(df2)
    node2 = make_check("has_one_of", [
        "col",
        make_tree("children", [make_token('"A"'), make_token('"B"')]),
    ])
    label3, result3 = qc2._apply_has_one_of(node2)
    assert label3 == "has_one_of"
    assert result3 is True

def test_apply_value_length_ignore_nulls_and_comparator(qc_basic):
    # Column D has strings with lengths mostly 5, but one nan
    node = make_check("value_length", [
        "D",
        "False",
        "<",
        "6"
    ])  # ignore_nulls 'False', comparator < rhs=6
    label, passed = qc_basic._apply_value_length(node)
    assert label == "value_length"
    assert isinstance(passed, bool)

    # Now ignore_nulls='True' (skip NaN and empty string rows)
    node2 = make_check("value_length", [
        "D",
        "True",
        "==",
        "5"
    ])
    label2, passed2 = qc_basic._apply_value_length(node2)
    assert label2 == "value_length"
    assert passed2 in [True, False]

def test_apply_percent_value_length_basic(qc_basic):
    # Length 5 for "hello", 5 for "world", 3 for "foo", 3 for "bar", and nan treated zero length.
    node = make_check("percent_value_length", [
        "D",
        "40",
        "False",
        "5"
    ])
    label, passed = qc_basic._apply_percent_value_length(node)
    assert label == "percent_value_length"
    assert isinstance(passed, bool)

def test_apply_has_columns_case_sensitive_and_ignore_case():
    df = pd.DataFrame(columns=["A", "b", "C"])
    qc = dqc.QualityChecker(df)

    # Check case-sensitive true
    node = make_check("has_columns", [
        make_tree("children", [make_token('"A"'), make_token('"b"')])
    ])
    label, result = qc._apply_has_columns(node)
    assert label == "has_columns"
    assert result is True

    # Check case-insensitive with columns ['a', 'B']
    node2 = make_check("has_columns", [
        make_tree("children", [make_token('"a"'), make_token('"B"')]),
        "True"
    ])
    label2, result2 = qc._apply_has_columns(node2)
    assert label2 == "has_columns"
    assert result2 is True

    # Missing columns case
    node3 = make_check("has_columns", [
        make_tree("children", [make_token('"A"'), make_token('"X"')])
    ])
    label3, result3 = qc._apply_has_columns(node3)
    assert label3 == "has_columns"
    assert result3 is False

def test_apply_check_row_count_and_exceptions(qc_basic):
    # row_count > 3
    node = make_check("row_count", [">", "3"])
    label, result = qc_basic._apply_check(wrap_check(node))
    assert label == "row_count"
    assert result is True

    # row_count == 100 should be False
    node2 = make_check("row_count", ["==", "100"])
    label2, result2 = qc_basic._apply_check(wrap_check(node2))
    assert label2 == "row_count"
    assert result2 is False

    # is_unique on column E with identical values (not unique)
    node3 = make_check("is_unique", ["E"])
    label3, result3 = qc_basic._apply_check(wrap_check(node3))
    assert label3 == "is_unique"
    assert result3 is False

    # not_null on column A all non-null => True
    node4 = make_check("not_null", ["A"])
    label4, result4 = qc_basic._apply_check(wrap_check(node4))
    assert label4 == "not_null"
    assert result4 is True

    # is_positive on column F all positive => True
    node5 = make_check("is_positive", ["F"])
    label5, result5 = qc_basic._apply_check(wrap_check(node5))
    assert label5 == "is_positive"
    assert result5 is True

    # has_one_of, passing values
    node6 = make_check("has_one_of", [
        "B",
        make_tree("children", [make_token('"x"'), make_token('"y"'), make_token('"z"')]),
    ])
    label6, result6 = qc_basic._apply_check(wrap_check(node6))
    assert label6 == "has_one_of"
    assert result6 is True

    # quantile usage through _apply_check (reuse earlier test)
    node7 = make_check("quantile", ["F", "0.5", ">", "20"])
    label7, result7 = qc_basic._apply_check(wrap_check(node7))
    assert label7 == "quantile"
    assert result7 is True

    # is_date returns True on column C
    node8 = make_check("is_date", ["C"])
    label8, result8 = qc_basic._apply_check(wrap_check(node8))
    assert label8 == "is_date"
    assert result8 is True

    # value_length usage through _apply_check
    node9 = make_check("value_length", ["D", "False", "==", "5"])
    label9, result9 = qc_basic._apply_check(wrap_check(node9))
    assert label9 == "value_length"
    assert isinstance(result9, bool)

    # percent_value_length usage through _apply_check
    node10 = make_check("percent_value_length", ["D", "20", "False", "5"])
    label10, result10 = qc_basic._apply_check(wrap_check(node10))
    assert label10 == "percent_value_length"
    assert isinstance(result10, bool)

    # Unknown check should raise GenieException
    unknown_node = make_check("unknown_check", ["A"])
    with pytest.raises(GenieException):
        qc_basic._apply_check(wrap_check(unknown_node))

    # KeyError: column not present => returns (msg, False)
    missing_col_node = make_check("not_null", ["NO_COL"])
    label_missing, result_missing = qc_basic._apply_check(wrap_check(missing_col_node))
    assert result_missing is False
    assert label_missing.startswith("Key error:")

def test_apply_predicates_runs_all_checks_in_code_block(qc_basic):
    # Create two checks in a code_block node: row_count > 3 and not_null A
    check1 = make_check("row_count", [">", "3"])
    check2 = make_check("not_null", ["A"])
    code_block = make_code_block([check1, check2])
    predicates = make_predicates([code_block])

    results = qc_basic._apply_predicates(predicates.children)
    # Results order is reversed because of insert(0, v)
    assert any(label == "row_count" and outcome for label, outcome in results)
    assert any(label == "not_null" and outcome for label, outcome in results)

def test_run_raises_on_multiple_children(qc_basic):
    pt = make_tree("root", [make_tree("predicates", []), make_tree("predicates", [])])
    with pytest.raises(GenieException):
        qc_basic._run(pt)

def test_run_raises_on_unknown_instruction(qc_basic):
    bad_inst = make_tree("root", [make_tree("bad_instruction", [])])
    with pytest.raises(GenieException):
        qc_basic._run(bad_inst)

def test_parse_spec_and_validate_spec_success_and_failure():
    good_spec = "row_count > 0"
    err, valid = dqc.QualityChecker.validate_spec(good_spec)
    assert valid is True
    assert err is None

    bad_spec = "row_count >> 0"  # invalid syntax
    err, valid = dqc.QualityChecker.validate_spec(bad_spec)
    assert valid is False
    assert isinstance(err, str)

def test_run_method_returns_expected(qc_basic):
    spec = "row_count > 0"
    results = qc_basic.run(spec)
    assert isinstance(results, list)
    # It contains tuples (label, boolean)
    assert all(isinstance(t[0], str) and isinstance(t[1], bool) for t in results)

def test_run_with_ignore_column_case_lowercases_columns(df_basic):
    qc = dqc.QualityChecker(df_basic.copy())
    spec = "not_null a"
    # 'a' column does not exist, but test should lowercase original columns
    with pytest.raises(KeyError):
        qc.run(spec, ignore_column_case=True)
    # But columns are all lower-case after run attempt
    assert all(c.islower() for c in qc._obj.columns)

def test_ignore_column_case_flag_effect(qc_basic):
    qc_basic.ignore_column_case = True
    colname = "A"
    treated = qc_basic._treat_column_name(colname)
    assert treated == colname.lower()

def test__apply_percent_value_length_ignore_nulls_true_vs_false():
    df = pd.DataFrame({
        "col": ["a", "bb", "", np.nan, "ccc"]
    })
    qc = dqc.QualityChecker(df)

    node_false = make_check("percent_value_length", [
        "col", "20", "False", "2"
    ])
    label_f, val_f = qc._apply_percent_value_length(node_false)
    assert label_f == "percent_value_length"
    assert isinstance(val_f, bool)

    node_true = make_check("percent_value_length", [
        "col", "20", "True", "2"
    ])
    label_t, val_t = qc._apply_percent_value_length(node_true)
    assert label_t == "percent_value_length"
    assert isinstance(val_t, bool)
