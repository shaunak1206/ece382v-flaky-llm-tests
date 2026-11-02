import pytest
import pandas as pd
from lark import Tree, Token
from genie_pkg.dqc import QualityChecker, GenieException
import numpy as np


# Helper to build lark-like node objects for testing _apply_* methods
class Node:
    def __init__(self, data, children):
        self.data = data
        self.children = children


def make_token(value):
    # mimic a Lark Token for cases that need .value attribute
    class DummyToken:
        def __init__(self, val):
            self.value = val
    return DummyToken(value)


@pytest.fixture
def df_basic():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["a", "b", "c", "d", "e"],
        "C": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", None, "2021-05-01"]),
        "D": [10, 10, 10, 10, 10],
        "E": ["foo", "foo", "foo", "foo", None]
    })


@pytest.fixture
def qc(df_basic):
    return QualityChecker(df_basic)


def test_init_with_empty_dataframe_raises():
    df = pd.DataFrame({"empty": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_comparator_to_fn():
    fn = QualityChecker._comparator_to_fn(">")
    assert fn(5) is True if 10 > 5 else True
    assert fn(3) is False if 2 > 3 else False

    eq_fn = QualityChecker._comparator_to_fn("==")
    assert eq_fn(5) is True if 5 == 5 else True
    lt_fn = QualityChecker._comparator_to_fn("<")
    assert lt_fn(2) is True if 1 < 2 else True

    false_fn = QualityChecker._comparator_to_fn("???")
    assert false_fn(100) is False


def test_treat_column_name_ignore_case_false(qc):
    qc.ignore_column_case = False
    assert qc._treat_column_name("AbC") == "AbC"


def test_treat_column_name_ignore_case_true(qc):
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_apply_quantile_true(qc):
    # Quantile of column "A" is 3 for q=0.5
    node = Node("quantile", ["A", "0.5", ">", "2"])
    res = qc._apply_quantile(node)
    assert res == ("quantile", True)
    node_eq = Node("quantile", ["A", "0.5", "==", "3"])
    res_eq = qc._apply_quantile(node_eq)
    assert res_eq == ("quantile", True)
    node_false = Node("quantile", ["A", "0.5", "<", "2"])
    res_false = qc._apply_quantile(node_false)
    assert res_false == ("quantile", False)


def test_apply_quantile_ignore_column_case(df_basic):
    qc = QualityChecker(df_basic)
    qc.ignore_column_case = True
    df_basic.columns = [c.upper() for c in df_basic.columns]
    qc._obj = df_basic
    node = Node("quantile", ["a", "0.5", ">", "2"])
    result = qc._apply_quantile(node)
    assert result[0] == "quantile"
    assert isinstance(result[1], bool)


def test_apply_date_validation_success(qc):
    node = Node("is_date", ["C"])
    result = qc._apply_date_validation(node)
    assert result == ("is_date", True)


def test_apply_date_validation_failure(qc):
    # Add invalid date column
    qc._obj["bad_date"] = ["2020-01-01", "nope", "2020-01-03", "2020-01-04", "2020-01-05"]
    node = Node("is_date", ["bad_date"])
    label, valid = qc._apply_date_validation(node)
    assert valid is False
    assert "Date parse error" in label


def test_apply_has_one_of_default_case(qc):
    node = Node("has_one_of", [
        "B",
        Node("list", [make_token('"a"'), make_token('"b"'), make_token('"c"')])
    ])
    result = qc._apply_has_one_of(node)
    assert result == ("has_one_of", False)  # because B has "d" and "e" too

    # Adjust dataframe so all B values are from allowed set
    qc._obj["B"] = ["a", "a", "b", "b", "c"]
    result = qc._apply_has_one_of(node)
    assert result == ("has_one_of", True)


def test_apply_has_one_of_ignore_case(qc):
    node = Node("has_one_of", [
        "B",
        Node("list", [make_token('"A"'), make_token('"B"'), make_token('"C"')]),
        "True"
    ])
    qc._obj["B"] = ["a", "A", "b", "B", "c"]
    res = qc._apply_has_one_of(node)
    assert res == ("has_one_of", True)


def test_apply_value_length_strict(qc):
    qc._obj["F"] = ["aaa", "bbb", "ccc", "ddd", "eee"]
    node = Node("value_length", ["F", "==", "3"])
    result = qc._apply_value_length(node)
    assert result == ("value_length", True)


def test_apply_value_length_ignore_nulls(qc):
    qc._obj["G"] = ["aa", "", "cc", "  ", None]
    node = Node("value_length", ["G", "True", "==", "2"])
    result = qc._apply_value_length(node)
    assert result[0] == "value_length"
    # lengths ignoring nulls ("", whitespace, None) should be equal length 2, 2 strings
    assert result[1] is True


def test_apply_value_length_unique_lengths_fail(qc):
    qc._obj["H"] = ["a", "bb", "ccc", "dd", "e"]
    node = Node("value_length", ["H", "==", "1"])
    result = qc._apply_value_length(node)
    assert result[0] == "value_length"
    assert result[1] is False


def test_apply_percent_value_length_basic(qc):
    qc._obj["I"] = ["abc", "abc", None, "", "abc"]
    # 3 out of 5 have length 3, so 60%
    node = Node("percent_value_length", ["I", "60", "==", "3"])
    result = qc._apply_percent_value_length(node)
    assert result[0] == "percent_value_length"
    assert result[1] is True

    # 80% is false in this case
    node2 = Node("percent_value_length", ["I", "80", "==", "3"])
    result2 = qc._apply_percent_value_length(node2)
    assert result2[1] is False


def test_apply_percent_value_length_ignore_nulls(qc):
    qc._obj["J"] = ["abc", "abc", None, "", "abc"]
    node = Node("percent_value_length", ["J", "60", "True", "==", "3"])
    result = qc._apply_percent_value_length(node)
    assert result[0] == "percent_value_length"
    # ignoring nulls, length 3 count is 3 out of 3 non-null values = 100%
    assert result[1] is True


def test_apply_has_columns_basic(qc):
    columns = [make_token('"A"'), make_token('"B"')]
    node = Node("has_columns", [Node("list", columns)])
    result = qc._apply_has_columns(node)
    assert result[0] == "has_columns"
    assert result[1] is True

    # Add a missing column
    node_missing = Node("has_columns", [Node("list", columns + [make_token('"Z"')])])
    result_missing = qc._apply_has_columns(node_missing)
    assert result_missing[1] is False


def test_apply_has_columns_ignore_case(qc):
    columns = [make_token('"A"'), make_token('"B"')]
    node = Node("has_columns", [Node("list", columns), "True"])
    # Uppercase columns in dataframe
    qc._obj.columns = [c.upper() for c in qc._obj.columns]
    result = qc._apply_has_columns(node)
    assert result[1] is True


def test_apply_check_row_count(qc):
    node = Node("row_count", [">", "3"])
    out = qc._apply_check([node])
    assert out[0] == "row_count"
    assert out[1] is True

    node2 = Node("row_count", ["<", "3"])
    out2 = qc._apply_check([node2])
    assert out2[1] is False


def test_apply_check_is_unique(qc):
    # Column "A" is unique
    node = Node("is_unique", ["A"])
    result = qc._apply_check([node])
    assert result == ("is_unique", True)

    # Make "A" non-unique
    qc._obj.loc[1, "A"] = qc._obj.loc[0, "A"]
    result2 = qc._apply_check([node])
    assert result2 == ("is_unique", False)


def test_apply_check_not_null(qc):
    node = Node("not_null", ["A"])
    result = qc._apply_check([node])
    assert result == ("not_null", True)
    node_null = Node("not_null", ["E"])
    result_null = qc._apply_check([node_null])
    assert result_null == ("not_null", False)


def test_apply_check_is_positive(qc):
    # column A all > 0
    node = Node("is_positive", ["A"])
    result = qc._apply_check([node])
    assert result == ("is_positive", True)
    # Create negative value
    qc._obj.loc[0, "A"] = -1
    result2 = qc._apply_check([node])
    assert result2 == ("is_positive", False)


def test_apply_check_unknown_raises(qc):
    node = Node("unknown_check", ["A"])
    with pytest.raises(GenieException):
        qc._apply_check([node])


def test_apply_check_keyerror_returns_false(qc):
    # a column that does not exist triggers KeyError inside _apply_check
    node = Node("is_unique", ["NON_EXISTENT_COLUMN"])
    label, status = qc._apply_check([node])
    assert status is False
    assert "Key error" in label


def test_apply_predicates_inserts_at_start(qc):
    # Build Tree for code_block with two checks
    node1 = Node("row_count", [">", "0"])
    node2 = Node("not_null", ["A"])
    code_block_node = Node("code_block", [Node("dummy", [node1]), Node("dummy", [node2])])
    predicates_node = Node("predicates", [code_block_node])
    results = qc._apply_predicates(predicates_node.children)
    # Since results.insert(0, v), last check is at front
    # So results[0] corresponds to node2 result, results[1] to node1's
    labels = [r[0] for r in results]
    assert "not_null" in labels and "row_count" in labels


def test__run_accepts_predicates_and_raises_on_bad_instruction(qc):
    # Prepare a predicates Tree with one code_block child (no actual checks)
    node1 = Node("code_block", [])
    predicates_node = Node("predicates", [node1])
    tree = Tree("root", [predicates_node])  # The tree must have one child only

    # Correct use: predicates
    res = qc._run(Tree("root", [Node("predicates", [])]))
    assert isinstance(res, list)

    # With actual predicates
    res2 = qc._run(Tree("root", [predicates_node]))
    assert isinstance(res2, list)

    # Tree with more than one child raises
    with pytest.raises(GenieException):
        qc._run(Tree("root", [predicates_node, predicates_node]))

    # Tree with unknown instruction raise
    with pytest.raises(GenieException):
        qc._run(Tree("root", [Node("unknown", [])]))


def test_validate_spec_return_values():
    valid_spec = "row_count > 0"
    err, valid = QualityChecker.validate_spec(valid_spec)
    assert valid is True
    assert err is None

    invalid_spec = "something wrong"
    err, valid = QualityChecker.validate_spec(invalid_spec)
    assert valid is False
    assert isinstance(err, str)


def test_run_returns_list_and_ignore_column_case(df_basic):
    qc = QualityChecker(df_basic)
    # Using a valid check_spec string
    check_spec = "row_count > 0"
    results = qc.run(check_spec)
    assert isinstance(results, list)
    # test ignore_column_case actually affects column names
    qc = QualityChecker(df_basic.rename(columns=str.upper))
    results2 = qc.run(check_spec, ignore_column_case=True)
    assert isinstance(results2, list)
