import pytest
import pandas as pd
from genie_pkg.dqc import QualityChecker
from lark import Tree
import numpy as np


# Helper to create a Tree node like structure for testing internal methods
class FakeChild:
    def __init__(self, value):
        self.value = value


class FakeNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children


@pytest.fixture
def df_simple():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })


@pytest.fixture
def df_with_nan():
    return pd.DataFrame({
        "col1": ["abc", "def", None, "ghi", ""],
        "col2": ["", None, None, None, None],
        "date": ["2020-01-01", "2020-02-02", "invalid", "2020-04-04", None],
        "num": [1, 2, 3, 4, 5]
    })


@pytest.fixture
def checker_simple(df_simple):
    return QualityChecker(df_simple)


@pytest.fixture
def checker_with_nan(df_with_nan):
    return QualityChecker(df_with_nan)


def test_validate_empty_df_raises():
    empty_df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(empty_df)


@pytest.mark.parametrize("comparator, rhs, lhs, expected", [
    (">", 5, 6, True),
    (">", 5, 4, False),
    ("==", 3, 3, True),
    ("==", 3, 2, False),
    ("<", 10, 9, True),
    ("<", 10, 11, False),
    ("unknown", 10, 1, False),
])
def test_comparator_to_fn_behavior(comparator, rhs, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert fn(lhs) == expected


def test_treat_column_name_respects_ignore_case(checker_simple):
    checker_simple.ignore_column_case = True
    assert checker_simple._treat_column_name("ColName") == "colname"
    checker_simple.ignore_column_case = False
    assert checker_simple._treat_column_name("ColName") == "ColName"


def test_apply_quantile_true_and_false():
    # Setup DataFrame with predictable quantiles
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="quantile",
        children=["val", 0.5, ">", 2.5]
    )
    # Median is 3, 3 > 2.5 == True
    label, result = checker._apply_quantile(node)
    assert label == "quantile"
    assert result is True

    node = FakeNode(
        data="quantile",
        children=["val", 0.5, "<", 2.5]
    )
    label, result = checker._apply_quantile(node)
    # Median is 3, 3 < 2.5 == False
    assert not result


def test_apply_date_validation_success_and_failure(checker_with_nan):
    node = FakeNode(
        data="is_date",
        children=["date"]
    )
    label, result = checker_with_nan._apply_date_validation(node)
    assert label == "is_date"
    assert result is False  # column contains invalid date "invalid"

    # Fix date col to valid dates only
    checker_with_nan._obj["date"] = pd.to_datetime(
        ["2020-01-01", "2020-02-02", "2020-03-03", "2020-04-04", "2020-05-05"]
    )
    label, result = checker_with_nan._apply_date_validation(node)
    assert label == "is_date"
    assert result is True


def test_apply_has_one_of_basic_and_ignore_case(checker_simple):
    # Setup node with values
    node = FakeNode(
        data="has_one_of",
        children=[
            "A",
            FakeNode(children=[FakeChild('"1"'), FakeChild('"2"')]),
        ]
    )
    # Column A has unique values [1, 2, 3], 3 not in ["1","2"] (strings not ints)
    label, result = checker_simple._apply_has_one_of(node)
    assert label == "has_one_of"
    assert result is False

    # Values as strings to match unique values exactly
    df = pd.DataFrame({"A": ["1", "2", "1"]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="has_one_of",
        children=[
            "A",
            FakeNode(children=[FakeChild('"1"'), FakeChild('"2"')]),
        ]
    )
    label, result = checker._apply_has_one_of(node)
    assert result is True

    # Test with ignore_case=True (3rd child 'True'), values that differ by case
    df = pd.DataFrame({"A": ["a", "B", "A"]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="has_one_of",
        children=[
            "A",
            FakeNode(children=[FakeChild('"a"'), FakeChild('"b"')]),
            "True"
        ]
    )
    label, result = checker._apply_has_one_of(node)
    assert result is True


@pytest.mark.parametrize("ignore_nulls,val_map", [("False", [3]), ("True", [3])])
def test_apply_value_length_single_length(ignore_nulls, val_map):
    df = pd.DataFrame({"col": ["aaa", "bbb", "ccc"]})
    checker = QualityChecker(df)

    if ignore_nulls == "False":
        # length of each string is 3, unique_lengths = [3]
        node = FakeNode(
            data="value_length",
            children=["col", "==", "3"]
        )
    else:
        # Length ignore_nulls = True passed as 2nd arg
        node = FakeNode(
            data="value_length",
            children=["col", "True", "==", "3"]
        )

    label, result = checker._apply_value_length(node)
    assert label == "value_length"
    assert result is True


def test_apply_value_length_detects_multiple_lengths():
    df = pd.DataFrame({"col": ["aa", "bbb", "cccc"]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="value_length",
        children=["col", "==", "3"]
    )
    label, result = checker._apply_value_length(node)
    assert label == "value_length"
    assert result is False


def test_apply_percent_value_length_basic():
    df = pd.DataFrame({"col": ["aa", "bb", "ccc", "ddd", None]})
    checker = QualityChecker(df)

    # 40 percent of rows have length == 3
    node = FakeNode(
        data="percent_value_length",
        children=["col", "40", "==", "3"]
    )
    label, result = checker._apply_percent_value_length(node)
    assert label == "percent_value_length"
    assert result is True

    # 80 percent threshold should fail
    node = FakeNode(
        data="percent_value_length",
        children=["col", "80", "==", "3"]
    )
    label, result = checker._apply_percent_value_length(node)
    assert label == "percent_value_length"
    assert not result


def test_apply_percent_value_length_ignore_nulls():
    df = pd.DataFrame({"col": ["aa", "bb", "ccc", "ddd", None, ""]})
    checker = QualityChecker(df)

    # with ignore_nulls=True rows with None or "" are ignored
    node = FakeNode(
        data="percent_value_length",
        children=["col", "80", "True", "==", "3"]
    )
    label, result = checker._apply_percent_value_length(node)
    assert label == "percent_value_length"
    assert result is True


def test_apply_has_columns_basic_and_ignore_case():
    df = pd.DataFrame(columns=["A", "b", "C"])
    checker = QualityChecker(df)
    children_nodes = [FakeChild('"A"'), FakeChild('"b"')]
    node = FakeNode(
        data="has_columns",
        children=[FakeNode(children=children_nodes)]
    )
    label, result = checker._apply_has_columns(node)
    assert label == "has_columns"
    assert result is True

    # Case insensitive
    node = FakeNode(
        data="has_columns",
        children=[FakeNode(children=children_nodes), 'True']
    )
    checker.ignore_column_case = False
    label, result = checker._apply_has_columns(node)
    assert result is True

    # Columns missing
    node = FakeNode(
        data="has_columns",
        children=[FakeNode(children=[FakeChild('"A"'), FakeChild('"x"')])]
    )
    label, result = checker._apply_has_columns(node)
    assert not result


def test_apply_check_row_count():
    df = pd.DataFrame({"A": range(5)})
    checker = QualityChecker(df)
    node = FakeNode(
        data="row_count",
        children=[">", "3"]
    )
    label, result = checker._apply_check([node])
    assert label == "row_count"
    assert result is True

    node = FakeNode(
        data="row_count",
        children=["<", "3"]
    )
    label, result = checker._apply_check([node])
    assert label == "row_count"
    assert not result


def test_apply_check_is_unique():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 5, 5]})
    checker = QualityChecker(df)

    node = FakeNode(
        data="is_unique",
        children=["A"]
    )
    label, result = checker._apply_check([node])
    assert label == "is_unique"
    assert result is True

    node = FakeNode(
        data="is_unique",
        children=["B"]
    )
    label, result = checker._apply_check([node])
    assert label == "is_unique"
    assert not result


def test_apply_check_not_null():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, None, 3]})
    checker = QualityChecker(df)

    node = FakeNode(
        data="not_null",
        children=["A"]
    )
    label, result = checker._apply_check([node])
    assert label == "not_null"
    assert result is True

    node = FakeNode(
        data="not_null",
        children=["B"]
    )
    label, result = checker._apply_check([node])
    assert label == "not_null"
    assert not result


def test_apply_check_is_positive():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, -1, 3]})
    checker = QualityChecker(df)

    node = FakeNode(
        data="is_positive",
        children=["A"]
    )
    label, result = checker._apply_check([node])
    assert label == "is_positive"
    assert result is True

    node = FakeNode(
        data="is_positive",
        children=["B"]
    )
    label, result = checker._apply_check([node])
    assert label == "is_positive"
    assert not result


def test_apply_check_has_one_of():
    df = pd.DataFrame({"col": ["a", "b", "c"]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="has_one_of",
        children=[
            "col",
            FakeNode(children=[FakeChild('"a"'), FakeChild('"b"')])
        ]
    )
    label, result = checker._apply_check([node])
    assert label == "has_one_of"
    assert not result

    node.children.append("True")  # ignore_case = True
    label, result = checker._apply_check([node])
    assert label == "has_one_of"
    # Still False because 'c' missing


def test_apply_check_unsupported_raises():
    df = pd.DataFrame({"col": [1, 2]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="unsupported_check",
        children=["col"]
    )
    with pytest.raises(Exception):
        checker._apply_check([node])


def test_apply_check_key_error_handling():
    df = pd.DataFrame({"col": [1, 2]})
    checker = QualityChecker(df)
    node = FakeNode(
        data="not_null",
        children=["missing_col"]
    )
    label, result = checker._apply_check([node])
    assert result is False
    assert "Key error" in label


def test_apply_predicates_basic():
    df = pd.DataFrame({"col": [1, 2, 3]})
    checker = QualityChecker(df)
    check_node = FakeNode(
        data="not_null",
        children=["col"]
    )
    code_block_node = FakeNode(
        data="code_block",
        children=[check_node]
    )
    predicates = [code_block_node]
    results = checker._apply_predicates(predicates)
    assert isinstance(results, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in results)
    assert results[0][0] == "not_null"
    assert results[0][1] is True


def test_apply_predicates_unknown_raises():
    checker = QualityChecker(pd.DataFrame({"a": [1]}))
    with pytest.raises(Exception):
        checker._apply_predicates([FakeNode(data="unknown", children=[])])


def test_run_with_predicates_success(monkeypatch):
    df = pd.DataFrame({"col": [1, 1, 1]})
    checker = QualityChecker(df)

    # Construct AST: predicates -> code_block -> some check children
    node_check = FakeNode(
        data="not_null",
        children=["col"]
    )
    code_block = FakeNode(data="code_block", children=[node_check])
    predicates = FakeNode(data="predicates", children=[code_block])
    ast = FakeNode(data='root', children=[predicates])

    monkeypatch.setattr(checker, "_run", lambda pt: checker._apply_predicates(pt.children[0].children))
    monkeypatch.setattr(checker, "_parse_spec", lambda cs: ast)

    result = checker.run("not_null(col)")
    assert isinstance(result, list)
    assert any(r[0] == "not_null" for r in result)


def test_run_raises_on_multiple_children():
    df = pd.DataFrame({"a": [1]})
    checker = QualityChecker(df)

    ast = FakeNode(data="root", children=[FakeNode(data="predicates", children=[]), FakeNode(data="predicates", children=[])])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(checker, "_parse_spec", lambda cs: ast)
    with pytest.raises(Exception):
        checker.run("")


def test_validate_spec_success_and_fail():
    valid_spec = "row_count > 0"
    err, ok = QualityChecker.validate_spec(valid_spec)
    assert ok is True
    assert err is None

    invalid_spec = "row_count >> 0"  # invalid comparator
    err, ok = QualityChecker.validate_spec(invalid_spec)
    assert ok is False
    assert err is not None

