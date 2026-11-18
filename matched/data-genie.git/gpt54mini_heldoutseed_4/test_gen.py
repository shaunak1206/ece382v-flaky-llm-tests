import pytest
import pandas as pd

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


class Node:
    def __init__(self, data, children=None):
        self.data = data
        self.children = list(children or [])


def make_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "x"],
            "C": ["2020-01-01", "not-a-date", "2020-03-01"],
            "D": ["aa", "bbb", "ccc"],
            "E": [10, 20, 30],
        }
    )


def test_validate_empty_dataframe_raises():
    df = pd.DataFrame({"empty": [None, None]})
    with pytest.raises(AttributeError):
        _ = df.dqc


def test_comparator_to_fn_greater_equal_less_and_default():
    gt = QualityChecker._comparator_to_fn(">", 2)
    eq = QualityChecker._comparator_to_fn("==", 2)
    lt = QualityChecker._comparator_to_fn("<", 2)
    other = QualityChecker._comparator_to_fn("!=", 2)

    assert gt(3) is True
    assert gt(2) is False
    assert eq(2) is True
    assert eq(3) is False
    assert lt(1) is True
    assert lt(2) is False
    assert other(100) is False


def test_treat_column_name_respects_ignore_column_case():
    qc = make_df().dqc
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_apply_quantile():
    qc = make_df().dqc
    node = Node("quantile", ["E", "0.5", ">", "15"])
    name, result = qc._apply_quantile(node)
    assert name == "quantile"
    assert result is False

    node2 = Node("quantile", ["E", "0.5", ">", "10"])
    _, result2 = qc._apply_quantile(node2)
    assert result2 is True


def test_apply_date_validation_success_and_failure():
    qc = make_df().dqc
    ok_node = Node("is_date", ["C"])
    name, result = qc._apply_date_validation(ok_node)
    assert name == "is_date"
    assert result is False  # because one value is invalid

    valid_df = pd.DataFrame({"C": ["2020-01-01", "2020-03-01"]})
    valid_qc = valid_df.dqc
    name2, result2 = valid_qc._apply_date_validation(ok_node)
    assert name2 == "is_date"
    assert result2 is True


def test_apply_has_one_of_without_ignore_case():
    qc = make_df().dqc
    allowed = Node("allowed_values", [Node("token", ['"x"']), Node("token", ['"y"'])])
    node = Node("has_one_of", ["B", allowed])
    name, result = qc._apply_has_one_of(node)
    assert name == "has_one_of"
    assert result is True

    bad_allowed = Node("allowed_values", [Node("token", ['"x"'])])
    node2 = Node("has_one_of", ["B", bad_allowed])
    _, result2 = qc._apply_has_one_of(node2)
    assert result2 is False


def test_apply_has_one_of_with_ignore_case():
    df = pd.DataFrame({"B": ["X", "y"]})
    qc = df.dqc
    allowed = Node("allowed_values", [Node("token", ['"x"']), Node("token", ['"Y"'])])
    node = Node("has_one_of", ["B", allowed, "True"])
    _, result = qc._apply_has_one_of(node)
    assert result is True


def test_apply_value_length_without_ignoring_nulls():
    df = pd.DataFrame({"D": ["aa", "bb", "cc"]})
    qc = df.dqc
    node = Node("value_length", ["D", "==", "2"])
    name, result = qc._apply_value_length(node)
    assert name == "value_length"
    assert result is True


def test_apply_value_length_with_ignore_nulls():
    df = pd.DataFrame({"D": ["aa", "", "  ", None]})
    qc = df.dqc
    node = Node("value_length", ["D", "True", "==", "2"])
    _, result = qc._apply_value_length(node)
    assert result is True


def test_apply_percent_value_length_without_ignore_nulls():
    df = pd.DataFrame({"D": ["aa", "bb", "cc", "dd"]})
    qc = df.dqc
    node = Node("percent_value_length", ["D", "50", "==", "2"])
    name, result = qc._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert result is True
    assert "length" in qc._obj.columns


def test_apply_percent_value_length_with_ignore_nulls():
    df = pd.DataFrame({"D": ["aa", "bb", "", None]})
    qc = df.dqc
    node = Node("percent_value_length", ["D", "50", "True", "==", "2"])
    _, result = qc._apply_percent_value_length(node)
    assert result is True


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = df.dqc
    cols = Node("columns", [Node("token", ['"A"']), Node("token", ['"B"'])])
    node = Node("has_columns", [cols])
    _, result = qc._apply_has_columns(node)
    assert result is True

    cols2 = Node("columns", [Node("token", ['"a"']), Node("token", ['"b"'])])
    node2 = Node("has_columns", [cols2, "True"])
    _, result2 = qc._apply_has_columns(node2)
    assert result2 is True


def test_apply_check_row_count_is_unique_not_null_is_positive_and_unknown():
    df = make_df()
    qc = df.dqc

    row_count = Node("row_count", [">", "2"])
    assert qc._apply_check([row_count]) == ("row_count", True)

    is_unique = Node("is_unique", ["A"])
    assert qc._apply_check([is_unique]) == ("is_unique", True)

    not_null = Node("not_null", ["A"])
    assert qc._apply_check([not_null]) == ("not_null", True)

    is_positive = Node("is_positive", ["A"])
    assert qc._apply_check([is_positive]) == ("is_positive", True)

    with pytest.raises(GenieException):
        qc._apply_check([Node("does_not_exist", [])])


def test_apply_check_keyerror_returns_false():
    df = make_df()
    qc = df.dqc
    missing = Node("not_null", ["missing_column"])
    name, result = qc._apply_check([missing])
    assert name == "Key error: not_null for missing_column"
    assert result is False


def test_apply_predicates_prepends_results_in_reverse_order():
    df = make_df()
    qc = df.dqc

    check1 = Node("check", [Node("row_count", [">", "1"])])
    check2 = Node("check", [Node("is_positive", ["A"])])
    predicates = [Node("code_block", [check1, check2])]

    results = qc._apply_predicates(predicates)
    assert results == [("is_positive", True), ("row_count", True)]


def test_run_and_validate_spec_invalid_spec():
    df = make_df()
    qc = df.dqc

    msg, ok = QualityChecker.validate_spec("this is not a valid spec")
    assert ok is False
    assert isinstance(msg, str) and msg

    with pytest.raises(GenieException):
        qc._run(Node("root", [Node("not_predicates", [])]))
