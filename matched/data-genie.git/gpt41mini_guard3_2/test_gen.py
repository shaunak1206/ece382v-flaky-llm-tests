import pytest
import random
import numpy as np
import pandas as pd
from lark import Tree, Token
from genie_pkg.dqc import QualityChecker, GenieException


# Helper to build minimal node objects similar to parsed Tree nodes.
class DummyNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children

@pytest.mark.parametrize("data, comparator, rhs, lhs, expected_result", [
    (">", 10, 12, True),
    (">", 10, 8, False),
    ("==", 10, 10, True),
    ("==", 10, 9, False),
    ("<", 5, 4, True),
    ("<", 5, 6, False),
    ("!=", 0, 0, False),  # unknown comparator returns always False
])
def test__comparator_to_fn_logic(data, comparator, rhs, expected_result):
    # Seed random to fulfill instructions (even if comparator_to_fn uses none).
    random.seed(42)
    np.random.seed(42)

    fn = QualityChecker._comparator_to_fn(data, rhs)
    assert fn(comparator) == expected_result


def test_init_empty_columns_raises():
    random.seed(42)
    np.random.seed(42)
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_treat_column_name_ignore_case_flag_behavior():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"A": [1, 2]})
    checker = QualityChecker(df)
    # By default ignore_column_case is False
    assert checker._treat_column_name("Abc") == "Abc"

    checker.ignore_column_case = True
    assert checker._treat_column_name("Abc") == "abc"


def test_apply_quantile_basic(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    data = {"col": [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    checker = QualityChecker(df)
    comparer_called = []

    # Build node:
    # children: [column_name, q, comparator, rhs]
    # Use quantile 0.5, comparator '>', rhs=3
    node = DummyNode("quantile", ["col", "0.5", ">", "3"])

    label, result = checker._apply_quantile(node)
    assert label == "quantile"
    # median 3.0, comparator > 3 -> False
    assert result is False

    node = DummyNode("quantile", ["col", "0.5", "<", "3"])
    label, result = checker._apply_quantile(node)
    # 3 < 3 is False
    assert result is False

    node = DummyNode("quantile", ["col", "0.5", "==", "3"])
    label, result = checker._apply_quantile(node)
    assert result is True


def test_apply_date_validation(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"date_col": ["2020-01-01", "2021-12-31"], "bad_date": ["foo", "bar"]})
    checker = QualityChecker(df)

    node_good = DummyNode("is_date", ["date_col"])
    label, result = checker._apply_date_validation(node_good)
    assert label == "is_date"
    assert result is True

    node_bad = DummyNode("is_date", ["bad_date"])
    label, result = checker._apply_date_validation(node_bad)
    assert label.startswith("Date parse error")
    assert result is False


def test_apply_has_one_of_case_and_ignore(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"fruit": ["Apple", "banana", "Cherry"]})
    checker = QualityChecker(df)

    # allowed_values = ["Apple", "Banana", "Cherry"]
    allowed_children = [Token("VALUE", '"Apple"'), Token("VALUE", '"banana"')]
    allowed_node = DummyNode("allowed_values", allowed_children)

    # No ignore case param (default False)
    node = DummyNode("has_one_of", ["fruit", allowed_node])
    label, result = checker._apply_has_one_of(node)
    # Unique values ['Apple', 'banana', 'Cherry'] some not in allowed exactly
    assert result is False

    # Add ignore_case = 'True'
    node = DummyNode("has_one_of", ["fruit", allowed_node, 'True'])
    label, result = checker._apply_has_one_of(node)
    # 'cherry' not in allowed list lowercased, expect False
    assert result is False

    # Allowed all fruits lowercase
    allowed_children_all = [Token("VALUE", '"apple"'), Token("VALUE", '"banana"'), Token("VALUE", '"cherry"')]
    allowed_node_all = DummyNode("allowed_values", allowed_children_all)
    node = DummyNode("has_one_of", ["fruit", allowed_node_all, 'True'])
    label, result = checker._apply_has_one_of(node)
    assert result is True


def test_apply_value_length_with_and_without_ignore_nulls():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        "col": ["aaa", "bbb", "", None, "ccc"]
    })
    checker = QualityChecker(df)

    # length of all including "" and None treated as strings
    node_without_null_ignore = DummyNode("value_length", ["col", "<", "4"])
    label, result = checker._apply_value_length(node_without_null_ignore)
    assert label == "value_length"
    # All lengths are 3 except "" (length 0) and None (string 'None' length 4)
    # So multiple lengths exist -> len(unique_lengths) != 1 -> False
    assert result is False

    # length ignoring nulls (node.children has 4 elements: ignore_nulls, comparator, rhs)
    node_with_null_ignore = DummyNode("value_length", ["col", 'True', "<", "4"])
    label, result = checker._apply_value_length(node_with_null_ignore)
    # Valid strings without nulls are 'aaa','bbb','ccc', all length 3, so unique_lengths==1
    # 3 < 4 -> True
    assert result is True


def test_apply_percent_value_length_basic():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({
        "col": ["aaa", "bbb", "ccc", "ddd", None]
    })
    checker = QualityChecker(df)

    # 80% of rows must have length == 3; all except one None have length 3
    # Without ignore_nulls param (defaults to 'False')
    node = DummyNode("percent_value_length", ["col", "80", "False", "3"])
    label, result = checker._apply_percent_value_length(node)
    assert label == "percent_value_length"
    assert result is True

    # With ignore_nulls = 'True' and 80%, length 3
    node_with_true = DummyNode("percent_value_length", ["col", "80", "True", "ignore", "3"])
    # Fixing node structure to match code expectations:
    # len(node.children) == 5: percent = [1], ignore_nulls = [2], rhs = [4]
    node_with_true = DummyNode("percent_value_length", ["col", "80", "True", "dummy_ignore", "3"])
    label, result = checker._apply_percent_value_length(node_with_true)
    assert result is True

    # Fail case: require 100% length == 5
    node_fail = DummyNode("percent_value_length", ["col", "100", "False", "5"])
    label, result = checker._apply_percent_value_length(node_fail)
    assert result is False


def test_apply_has_columns_ignore_case_various():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame(columns=["A", "b", "CCC"])
    checker = QualityChecker(df)

    allowed_cols = [Token("VALUE", '"A"'), Token("VALUE", '"b"')]
    allowed_node = DummyNode("allowed_columns", allowed_cols)

    # ignore_case = False (default)
    node = DummyNode("has_columns", [allowed_node])
    label, result = checker._apply_has_columns(node)
    # Columns 'A' and 'b' are in df, expect True
    assert result is True

    # Add an extra column required: 'c'
    allowed_cols_extra = [Token("VALUE", '"A"'), Token("VALUE", '"b"'), Token("VALUE", '"c"')]
    allowed_node_extra = DummyNode("allowed_columns", allowed_cols_extra)
    node_extra = DummyNode("has_columns", [allowed_node_extra])
    label, result = checker._apply_has_columns(node_extra)
    assert result is False

    # ignore_case = True, pass 'true' string to enable ignore_case
    node_ignore = DummyNode("has_columns", [allowed_node_extra, 'True'])
    label, result = checker._apply_has_columns(node_ignore)
    # df columns lower: ['a','b','ccc'], allowed lower: ['a','b','c']
    # 'c' missing -> fail
    assert result is False

    # Add 'C' column to df to make test pass with ignore_case
    df2 = pd.DataFrame(columns=["A", "b", "CCC", "c"])
    checker2 = QualityChecker(df2)
    label, result = checker2._apply_has_columns(node_ignore)
    assert result is True


def test_apply_check_works_for_various_checks(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    data = {
        "col1": [1, 2, 3, 4],  # unique, positive, non-null
        "col2": [1, 1, 2, 2],  # not unique
        "col3": [1, -1, 3, 4],  # not all positive
        "col4": [1, 2, None, 4],
        "col5": ["a", "a", "a", "a"],
    }
    df = pd.DataFrame(data)
    checker = QualityChecker(df)

    # row_count > 3
    node = DummyNode("row_count", [">", "3"])
    label, result = checker._apply_check([node])
    assert label == "row_count"
    assert result is True

    # row_count == 5
    node_eq = DummyNode("row_count", ["==", "5"])
    label, result = checker._apply_check([node_eq])
    assert result is False

    # has_columns exists
    allowed_cols = [Token("VALUE", '"col1"'), Token("VALUE", '"col2"')]
    allowed_node = DummyNode("allowed_columns", allowed_cols)
    node_has_col = DummyNode("has_columns", [allowed_node])
    label, result = checker._apply_check([node_has_col])
    assert result is True

    # is_unique col1 (unique)
    node_unique = DummyNode("is_unique", ["col1"])
    label, result = checker._apply_check([node_unique])
    assert result is True

    # is_unique col2 (not unique)
    node_unique2 = DummyNode("is_unique", ["col2"])
    label, result = checker._apply_check([node_unique2])
    assert result is False

    # not_null col1 (no nulls)
    node_not_null = DummyNode("not_null", ["col1"])
    label, result = checker._apply_check([node_not_null])
    assert result is True

    # not_null col4 (has nulls)
    node_not_null2 = DummyNode("not_null", ["col4"])
    label, result = checker._apply_check([node_not_null2])
    assert result is False

    # is_positive col1 (all > 0)
    node_pos = DummyNode("is_positive", ["col1"])
    label, result = checker._apply_check([node_pos])
    assert result is True

    # is_positive col3 (one negative)
    node_pos2 = DummyNode("is_positive", ["col3"])
    label, result = checker._apply_check([node_pos2])
    assert result is False


def test_apply_check_key_error_returns_false():
    random.seed(42)
    np.random.seed(42)
    df = pd.DataFrame({"col": [1, 2]})
    checker = QualityChecker(df)

    node = DummyNode("is_unique", ["nonexistent_column"])
    label, result = checker._apply_check([node])
    assert label.startswith("Key error")
    assert result is False

def test_apply_check_unknown_raises_GenieException():
    random.seed(42)
    np.random.seed(42)
    df = pd.DataFrame({"col": [1, 2]})
    checker = QualityChecker(df)

    node = DummyNode("unknown_check", ["col"])
    with pytest.raises(GenieException):
        checker._apply_check([node])


def test__apply_predicates_code_block_and_unknown(monkeypatch):
    random.seed(42)
    np.random.seed(42)
    df = pd.DataFrame({"a": [1, 2, 3]})
    checker = QualityChecker(df)

    # Create a predicate of type code_block with one check child
    row_count_node = DummyNode("row_count", [">", "0"])
    check_node = DummyNode("check", [row_count_node])
    code_block_node = DummyNode("code_block", [check_node])

    results = checker._apply_predicates([code_block_node])
    # check returns tuple of (label, boolean)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0][0] == "row_count"
    assert results[0][1] is True

    # Unknown predicate type raises GenieException
    unknown_pred = DummyNode("unknown_predicate", [])
    with pytest.raises(GenieException):
        checker._apply_predicates([unknown_pred])


def test__run_valid_and_invalid_instruction(monkeypatch):
    random.seed(42)
    np.random.seed(42)
    df = pd.DataFrame({"a": [1, 2]})
    checker = QualityChecker(df)

    # Good case: predicates instruction with one child predicate
    row_count_node = DummyNode("row_count", [">", "0"])
    check_node = DummyNode("check", [row_count_node])
    code_block_node = DummyNode("code_block", [check_node])
    predicates_node = DummyNode("predicates", [code_block_node])
    tree = DummyNode("start", [predicates_node])

    results = checker._run(tree)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"
    assert results[0][1] is True

    # Invalid case: more than one child in root
    tree_multi = DummyNode("start", [predicates_node, predicates_node])
    with pytest.raises(GenieException):
        checker._run(tree_multi)

    # Unknown instruction type
    bad_instr = DummyNode("bad_instruction", [])
    tree_bad = DummyNode("start", [bad_instr])
    with pytest.raises(GenieException):
        checker._run(tree_bad)


def test_static_parse_spec_and_validate_spec(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    # Will parse real grammar.g from package resources (should succeed)
    good_spec = 'row_count > 0'

    # parse_spec returns a Tree instance
    tree = QualityChecker._parse_spec(good_spec)
    assert hasattr(tree, "children")

    # validate_spec returns (None, True) for good input
    err, valid = QualityChecker.validate_spec(good_spec)
    assert valid is True
    assert err is None

    # validate_spec returns (str, False) for bad input
    bad_spec = "this is invalid spec !!"
    err, valid = QualityChecker.validate_spec(bad_spec)
    assert valid is False
    assert isinstance(err, str)


def test_run_integration_basic(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    # Setup minimal df with columns
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    checker = QualityChecker(df)

    # Test run with valid spec that uses has_columns
    spec = 'has_columns ("A", "B")'
    results = checker.run(spec)
    # Should return list of tuples (label, bool)
    assert isinstance(results, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    # We expect True for both columns exist
    for label, res in results:
        assert res is True

    # Test run with ignore_column_case True modifies columns to lower
    checker = QualityChecker(df)
    results = checker.run(spec, ignore_column_case=True)
    # Columns of checker._obj should now be lowercase
    assert all(c.islower() for c in checker._obj.columns)


def test_run_with_invalid_spec_raises(monkeypatch):
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"A": [1]})
    checker = QualityChecker(df)

    bad_spec = "not_a_valid_spec !!"
    with pytest.raises(Exception):
        checker.run(bad_spec)


# Add a test confirming behavior of ignoring column case for operations
def test_ignore_column_case_effect_on_column_access():
    random.seed(42)
    np.random.seed(42)

    data = {"a": [1, 2, 3]}
    df = pd.DataFrame(data)
    checker = QualityChecker(df)

    # Force ignore_column_case True and columns forced lower-case
    checker.ignore_column_case = True
    checker._obj.columns = map(str.lower, checker._obj.columns)

    # Use _treat_column_name for uppercase access and check matches "a"
    colname = checker._treat_column_name("A")
    assert colname == "a"

    # The DataFrame columns after conversion should be lowercase string or map object
    assert all(isinstance(c, str) for c in checker._obj.columns)


# Run length based checks with empty strings and null mixing
def test_value_length_ignore_nulls_and_not():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"col": ["a", "aa", "aaa", "", None]})
    checker = QualityChecker(df)

    # Without ignore_nulls param in children length=3 (default ignore_nulls='False')
    node = DummyNode("value_length", ["col", "==", "3"])
    label, result = checker._apply_value_length(node)
    # Lengths are [1,2,3,0,4], multiple unique => False
    assert result is False

    # With ignore_nulls='True' (children length=4)
    node_with_ignore = DummyNode("value_length", ["col", "True", "==", "3"])
    label, result = checker._apply_value_length(node_with_ignore)
    # Now drop nulls and empty strings: left "a", "aa", "aaa" => lengths unique and ==3? No: lengths = [1,2,3] unique 3, so fails length == 1 => False
    assert result is False


def test_percent_value_length_with_mixed_nulls_edge():
    random.seed(42)
    np.random.seed(42)

    df = pd.DataFrame({"col": ["a", "aa", "aaa", ..., None]})
    # Build a DataFrame with 100 rows 70 with length 3, 30 others with length !=3 and some None
    vals = ["abc"]*70 + ["longer"]*20 + [None]*10
    df = pd.DataFrame({"col": vals})
    checker = QualityChecker(df)

    # percent = 60, ignore_nulls = False, rhs = 3
    node = DummyNode("percent_value_length", ["col", "60", "False", "3"])
    label, result = checker._apply_percent_value_length(node)
    # 70 rows length 3 out of 100, so ratio 70% > 60% -> True
    assert result is True

    # percent = 80, ignore_nulls = True, rhs = 3
    node2 = DummyNode("percent_value_length", ["col", "80", "True", "dummy", "3"])
    label, result = checker._apply_percent_value_length(node2)
    # drop None rows leaves 90 rows: 70 length==3, 20 longer
    # 70/100 * 100 = 70 < 80 -> False
    assert result is False
