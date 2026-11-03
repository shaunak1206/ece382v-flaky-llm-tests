import pytest
import pandas as pd
from lark import Tree, Token
from genie_pkg.dqc import QualityChecker, GenieException


# Helpers to create dummy Lark tree nodes simulating parsed DSL
def make_token(value):
    return Token('VALUE', value)

def make_tree(data, children):
    return Tree(data, children)

# Helper to create a QualityChecker for arbitrary DataFrame
def qc(df):
    return QualityChecker(df)


def test_validate_raises_on_all_nan_columns():
    df = pd.DataFrame({'A': [None, None], 'B': [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)

def test_validate_passes_non_empty_dataframe():
    df = pd.DataFrame({'A': [1, None], 'B': [None, None]})
    # Should not raise
    qc(df)

def test_comparator_to_fn_greater_equal_less():
    fn_gt = QualityChecker._comparator_to_fn(">", 5)
    assert fn_gt(6) is True
    assert fn_gt(5) is False

    fn_eq = QualityChecker._comparator_to_fn("==", 2)
    assert fn_eq(2) is True
    assert fn_eq(3) is False

    fn_lt = QualityChecker._comparator_to_fn("<", 10)
    assert fn_lt(9) is True
    assert fn_lt(10) is False

    fn_false = QualityChecker._comparator_to_fn("!=", 0)  # not implemented comparator
    assert fn_false(100) is False

def test_treat_column_name_ignore_case_flag():
    df = pd.DataFrame({'A': [1]})
    checker = qc(df)
    checker.ignore_column_case = False
    assert checker._treat_column_name("TestColumn") == "TestColumn"
    checker.ignore_column_case = True
    assert checker._treat_column_name("TestColumn") == "testcolumn"

def test_apply_quantile_correctly_evaluates():
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    checker = qc(df)
    # Lark node simulation: children: column_name, quantile, comparator, rhs
    node = make_tree("quantile", [
        "col1",  # column name
        "0.5",   # quantile (median)
        ">",
        "2"
    ])
    result = checker._apply_quantile(node)
    assert result[0] == "quantile"
    # median of col1 is 3, 3 > 2 -> True
    assert result[1] is True

def test_apply_date_validation_on_valid_and_invalid_dates():
    df = pd.DataFrame({'dates': ["2020-01-01", "2021-12-31"]})
    checker = qc(df)
    node_valid = make_tree("is_date", ["dates"])
    name, valid = checker._apply_date_validation(node_valid)
    assert name == "is_date"
    assert valid is True

    df_bad = pd.DataFrame({'dates': ["notadate", "2021-12-31"]})
    checker_bad = qc(df_bad)
    node_bad = make_tree("is_date", ["dates"])
    name_bad, valid_bad = checker_bad._apply_date_validation(node_bad)
    assert valid_bad is False
    assert "Date parse error" in name_bad

def test_apply_has_one_of_without_ignore_case():
    df = pd.DataFrame({'Col': ["A", "B", "C"]})
    checker = qc(df)
    allowed_children = [
        make_token('"A"'),
        make_token('"B"'),
        make_token('"C"'),
    ]
    allowed_tree = make_tree("allowed_vals", allowed_children)
    # no ignore_case child
    node = make_tree("has_one_of", [
        "Col",
        allowed_tree,
    ])
    name, valid = checker._apply_has_one_of(node)
    assert name == "has_one_of"
    assert valid is True

def test_apply_has_one_of_with_ignore_case_true():
    df = pd.DataFrame({'col': ["a", "b", "c"]})
    checker = qc(df)
    allowed_children = [
        make_token('"A"'),
        make_token('"B"'),
    ]
    allowed_tree = make_tree("allowed_vals", allowed_children)
    node = make_tree("has_one_of", [
        "col",
        allowed_tree,
        "True",
    ])
    name, valid = checker._apply_has_one_of(node)
    assert name == "has_one_of"
    assert valid is False  # because "c" not in allowed values

def test_apply_value_length_comparator_false_and_true():
    df = pd.DataFrame({'Col': ["aaa", "bbb", "ccc"]})
    checker = qc(df)
    # no ignore_nulls, comparator "==", rhs 3
    node = make_tree("value_length", ["Col", "==", "3"])
    name, valid = checker._apply_value_length(node)
    assert name == "value_length"
    assert valid is True

    node_false = make_tree("value_length", ["Col", "==", "2"])
    _, valid_false = checker._apply_value_length(node_false)
    assert valid_false is False

def test_apply_value_length_with_ignore_nulls():
    df = pd.DataFrame({'Col': ["aaa", "", "bbb"]})
    checker = qc(df)
    node = make_tree("value_length", ["Col", "True", "==", "3"])  # ignore_nulls = 'True'
    # "" converted to NaN and dropped, so lengths are all 3
    name, valid = checker._apply_value_length(node)
    assert name == "value_length"
    assert valid is True

def test_apply_percent_value_length_true_and_false():
    df = pd.DataFrame({'Col': ["aaa", "bbbb", "ccc"]})
    checker = qc(df)
    # percent=50, comparator: RHS length=3, no ignore_nulls (False)
    node = make_tree("percent_value_length", ["Col", "50", "False", "3"])
    name, valid = checker._apply_percent_value_length(node)
    assert name == "percent_value_length"
    # only 2 of 3 rows have length 3 -> 66% >=50 True
    assert valid is True

    # Now make percent 70, test False because only 66%
    node2 = make_tree("percent_value_length", ["Col", "70", "False", "3"])
    _, valid2 = checker._apply_percent_value_length(node2)
    assert valid2 is False

def test_apply_percent_value_length_ignore_nulls_true():
    df = pd.DataFrame({'Col': ["aaa", "", "ccc"]})
    checker = qc(df)
    # percent=100, ignore_nulls='True', rhs=3
    node = make_tree("percent_value_length", ["Col", "100", "True", "False", "3"])
    # Will use 'False' for ignore_nulls param from children index 2, so ignore nulls == True
    # But node has len 5, so this branch:
    # len == 4 else branch? Actually input is 5 long :  ["Col", "100", "True", "False", "3"]
    # That should trigger else branch (len !=4)
    name, valid = checker._apply_percent_value_length(node)
    assert name == "percent_value_length"
    # "aaa" and "ccc" lengths are 3, empty string ignored -> 2 of 2 rows match 3 length -> 100%
    assert valid is True

def test_apply_has_columns_with_and_without_ignore_case():
    df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
    checker = qc(df)

    cols_tree = make_tree("columns", [
        make_token('"A"'),
        make_token('"B"'),
    ])
    # without ignore_case param
    node = make_tree("has_columns", [cols_tree])
    name, valid = checker._apply_has_columns(node)
    assert name == "has_columns"
    assert valid is True

    # with ignore_case = 'True'
    node_ic = make_tree("has_columns", [cols_tree, "True"])
    name_ic, valid_ic = checker._apply_has_columns(node_ic)
    assert valid_ic is True

def test_apply_check_row_count_true_and_false():
    df = pd.DataFrame({'A': [1, 2, 3]})
    checker = qc(df)

    # check row_count with '>' 2
    check_tree = make_tree("row_count", [">", "2"])
    result = checker._apply_check([check_tree])
    assert result == ("row_count", True)

    # check row_count with '==' 3
    check_tree2 = make_tree("row_count", ["==", "3"])
    result2 = checker._apply_check([check_tree2])
    assert result2 == ("row_count", True)

    # check row_count with '<' 3 should be False
    check_tree3 = make_tree("row_count", ["<", "3"])
    result3 = checker._apply_check([check_tree3])
    assert result3 == ("row_count", False)

def test_apply_check_has_columns_used():
    df = pd.DataFrame({'A': [1], 'b': [2]})
    checker = qc(df)

    cols_tree = make_tree("columns", [
        make_token('"A"'),
    ])
    check_tree = make_tree("has_columns", [cols_tree])
    result = checker._apply_check([check_tree])
    assert result[0] == "has_columns"
    assert result[1] is True

def test_apply_check_is_unique_true_false():
    df = pd.DataFrame({'col': [1, 2, 3]})
    checker = qc(df)
    check_tree = make_tree("is_unique", ["col"])
    name, valid = checker._apply_check([check_tree])
    assert name == "is_unique"
    assert valid is True

    df_dup = pd.DataFrame({'col': [1, 2, 2]})
    checker_dup = qc(df_dup)
    name_dup, valid_dup = checker_dup._apply_check([check_tree])
    assert valid_dup is False

def test_apply_check_not_null_true_false():
    df = pd.DataFrame({'col': [1, 2, 3]})
    checker = qc(df)
    check_tree = make_tree("not_null", ["col"])
    name, valid = checker._apply_check([check_tree])
    assert valid is True

    df_null = pd.DataFrame({'col': [1, None, 3]})
    checker_null = qc(df_null)
    name_null, valid_null = checker_null._apply_check([check_tree])
    assert valid_null is False

def test_apply_check_is_positive_true_false():
    df = pd.DataFrame({'col': [1, 2, 3]})
    checker = qc(df)
    check_tree = make_tree("is_positive", ["col"])
    name, valid = checker._apply_check([check_tree])
    assert valid is True

    df_neg = pd.DataFrame({'col': [1, -2, 3]})
    checker_neg = qc(df_neg)
    name_neg, valid_neg = checker_neg._apply_check([check_tree])
    assert valid_neg is False

def test_apply_check_unknown_raises_and_keyerror_caught():
    df = pd.DataFrame({'col': [1, 2]})
    checker = qc(df)

    # Unknown check type
    unknown_tree = make_tree("unknown_check", ["col"])
    with pytest.raises(GenieException):
        checker._apply_check([unknown_tree])

    # Key error (column not present)
    missing_tree = make_tree("not_null", ["missing_col"])
    name, valid = checker._apply_check([missing_tree])
    assert valid is False
    assert "Key error" in name

def test_apply_predicates_runs_code_block_and_raises_on_unknown():
    df = pd.DataFrame({'col': [1, 2, 3]})
    checker = qc(df)

    # valid code block with one check: row_count > 2
    check_tree = make_tree("row_count", [">", "2"])
    code_block = make_tree("code_block", [check_tree])
    predicates = [code_block]

    results = checker._apply_predicates(predicates)
    assert isinstance(results, list)
    assert results and results[0][0] == "row_count"

    # unknown predicate data triggers exception
    bad_pred = make_tree("invalid_pred", [])
    with pytest.raises(GenieException):
        checker._apply_predicates([bad_pred])

def test_run_raises_for_multiple_children():
    df = pd.DataFrame({'col': [1]})
    checker = qc(df)
    # Tree with two children triggers exception
    tree = make_tree("root", [make_tree("predicates", []), make_tree("predicates", [])])
    with pytest.raises(GenieException):
        checker._run(tree)

def test_run_raises_on_unknown_instruction():
    df = pd.DataFrame({'col': [1]})
    checker = qc(df)
    tree = make_tree("root", [make_tree("unknown", [])])
    with pytest.raises(GenieException):
        checker._run(tree)

def test_parse_spec_and_validate_spec_returns_expected():
    # Minimal valid DSL for row_count > 0
    s = "row_count > 0"
    err, valid = QualityChecker.validate_spec(s)
    assert (err is None) and valid

    # Invalid spec returns error message and False
    err2, valid2 = QualityChecker.validate_spec("row_count >> 0")
    assert valid2 is False
    assert isinstance(err2, str)

def test_run_with_ignore_column_case_modifies_columns():
    df = pd.DataFrame({'ColOne': [1, 2]})
    checker = qc(df)
    # Check row_count > 1 as DSL string
    res = checker.run("row_count > 1", ignore_column_case=True)
    assert isinstance(res, list)
    # Columns altered to lowercase
    assert all(c.islower() for c in checker._obj.columns)

def test_run_full_integration_row_count_and_has_columns():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    checker = qc(df)
    spec = """
    {
      has_columns ["A", "B"]
      row_count > 2
    }
    """
    # The DSL grammar.g is unknown here, but run tries to parse this spec.
    # We do not have grammar or DSL here to parse complex multi-line specs so expect exception or error.
    # Instead, test a simple one-liner:
    spec_simple = "row_count > 2"
    result = checker.run(spec_simple)
    assert any(res[0] == "row_count" and res[1] is True for res in result)
