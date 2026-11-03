import pytest
import pandas as pd
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


class DummyToken:
    def __init__(self, value):
        self.value = value


class DummyNode:
    def __init__(self, data, children=None):
        self.data = data
        self.children = list(children or [])


def make_qc(df):
    return QualityChecker(df.copy())


def test_validate_rejects_empty_dataframe():
    df = pd.DataFrame({"a": [None], "b": [None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        QualityChecker(df)


def test_validate_allows_dataframe_with_non_empty_column():
    df = pd.DataFrame({"a": [None, 1], "b": [None, None]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)


def test_comparator_to_fn_gt_eq_lt_and_unknown():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    other = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(2) is False
    assert eq(3) is True
    assert eq(4) is False
    assert lt(2) is True
    assert lt(3) is False
    assert other(999) is False


def test_treat_column_name_respects_ignore_column_case():
    qc = make_qc(pd.DataFrame({"Name": ["A"]}))
    assert qc._treat_column_name("Name") == "Name"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Name") == "name"


def test_apply_date_validation_success_and_failure():
    qc = make_qc(pd.DataFrame({"dt": ["2024-01-01", "2024-02-02"]}))
    ok_node = DummyNode("is_date", ["dt"])
    assert qc._apply_date_validation(ok_node) == ("is_date", True)

    bad_qc = make_qc(pd.DataFrame({"dt": ["not-a-date"]}))
    bad_node = DummyNode("is_date", ["dt"])
    label, result = bad_qc._apply_date_validation(bad_node)
    assert label == "Date parse error: is_date for dt"
    assert result is False


def test_apply_has_one_of_without_ignore_case():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    qc = make_qc(df)
    node = DummyNode(
        "has_one_of",
        ["color", DummyNode("values", [DummyToken('"red"'), DummyToken('"blue"')])],
    )
    assert qc._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_has_one_of_with_ignore_case():
    df = pd.DataFrame({"color": ["Red", "BLUE", "red"]})
    qc = make_qc(df)
    node = DummyNode(
        "has_one_of",
        ["color", DummyNode("values", [DummyToken('"red"'), DummyToken('"blue"')]), "True"],
    )
    assert qc._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_has_one_of_detects_unexpected_values():
    df = pd.DataFrame({"color": ["red", "green"]})
    qc = make_qc(df)
    node = DummyNode(
        "has_one_of",
        ["color", DummyNode("values", [DummyToken('"red"'), DummyToken('"blue"')])],
    )
    assert qc._apply_has_one_of(node) == ("has_one_of", False)


def test_apply_value_length_without_ignore_nulls():
    df = pd.DataFrame({"code": ["AA", "BB", "CC"]})
    qc = make_qc(df)
    node = DummyNode("value_length", ["code", "==", "2"])
    assert qc._apply_value_length(node) == ("value_length", True)


def test_apply_value_length_with_ignore_nulls_and_mixed_lengths():
    df = pd.DataFrame({"code": ["AA", " ", None, "BB"]})
    qc = make_qc(df)
    node = DummyNode("value_length", ["code", "True", "==", "2"])
    label, result = qc._apply_value_length(node)
    assert label == "value_length"
    assert result is True


def test_apply_percent_value_length_without_ignore_nulls():
    df = pd.DataFrame({"code": ["AA", "AA", "BB", "C"]})
    qc = make_qc(df)
    node = DummyNode("percent_value_length", ["code", "75", "==", "2"])
    assert qc._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"Name": [1], "Age": [2]})
    qc = make_qc(df)

    node_sensitive = DummyNode(
        "has_columns",
        [DummyNode("columns", [DummyToken('"Name"'), DummyToken('"Age"')])],
    )
    assert qc._apply_has_columns(node_sensitive) == ("has_columns", True)

    node_insensitive = DummyNode(
        "has_columns",
        [DummyNode("columns", [DummyToken('"name"'), DummyToken('"age"')]), "True"],
    )
    assert qc._apply_has_columns(node_insensitive) == ("has_columns", True)


def test_apply_check_row_count_and_not_null_and_is_positive():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    qc = make_qc(df)

    row_count_check = [Tree("row_count", [">", "2"])]
    assert qc._apply_check(row_count_check) == ("row_count", True)

    not_null_check = [Tree("not_null", ["x"])]
    assert qc._apply_check(not_null_check) == ("not_null", True)

    is_positive_check = [Tree("is_positive", ["y"])]
    assert qc._apply_check(is_positive_check) == ("is_positive", True)


def test_apply_check_unknown_instruction_raises_genie_exception():
    qc = make_qc(pd.DataFrame({"x": [1]}))
    with pytest.raises(GenieException, match="not implemented"):
        qc._apply_check([Tree("some_new_check", ["x"])])


def test_apply_check_keyerror_returns_failure_message():
    qc = make_qc(pd.DataFrame({"x": [1]}))
    label, result = qc._apply_check([Tree("not_null", ["missing"])])
    assert label == "Key error: not_null for missing"
    assert result is False


def test_apply_predicates_inserts_results_in_reverse_order():
    qc = make_qc(pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]}))

    p1 = Tree("code_block", [Tree("check", [Tree("row_count", [">", "1"])])])
    p2 = Tree("code_block", [Tree("check", [Tree("not_null", ["y"])])])

    results = qc._apply_predicates([p1, p2])
    assert results == [("not_null", True), ("row_count", True)]


def test_run_with_invalid_parse_spec_returns_false_in_validate_spec():
    msg, ok = QualityChecker.validate_spec("this is not valid dsl !!!")
    assert ok is False
    assert isinstance(msg, str)


def test_run_unknown_instruction_raises_genie_exception():
    qc = make_qc(pd.DataFrame({"x": [1]}))
    ast = Tree("start", [Tree("unknown", [])])
    with pytest.raises(GenieException, match="Unknown instruction"):
        qc._run(ast)
