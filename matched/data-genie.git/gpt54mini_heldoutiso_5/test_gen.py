import pytest
import pandas as pd
import numpy as np
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_qc(df):
    return QualityChecker(df.copy())


def leaf_check(node):
    return [node]


def wrapped_check(node):
    return Tree("check", [node])


def code_block(*check_nodes):
    return Tree("code_block", [wrapped_check(n) for n in check_nodes])


def predicates(*blocks):
    return Tree("predicates", list(blocks))


def program(*blocks):
    return Tree("program", [predicates(*blocks)])


def test_validate_raises_on_all_empty_dataframe():
    df = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
    with pytest.raises(AttributeError, match="DataFrame is empty!!!"):
        QualityChecker(df)


def test_validate_allows_dataframe_with_one_non_null_column():
    df = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)


def test_comparator_to_fn_supported_operators():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)

    assert gt(4) is True and gt(2) is False
    assert eq(3) is True and eq(2) is False
    assert lt(2) is True and lt(4) is False


def test_comparator_to_fn_unknown_operator_returns_false():
    fn = QualityChecker._comparator_to_fn("!=", 3)
    assert fn(999) is False


def test_treat_column_name_respects_ignore_column_case():
    qc = make_qc(pd.DataFrame({"Name": ["a"]}))
    assert qc._treat_column_name("Name") == "Name"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Name") == "name"


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "b": [2]})
    qc = make_qc(df)

    node = Tree("has_columns", [Tree("cols", [Tree("col", ['"A"']), Tree("col", ['"b"'])])])
    name, ok = qc._apply_has_columns(node)
    assert name == "has_columns"
    assert ok is True

    node2 = Tree("has_columns", [Tree("cols", [Tree("col", ['"a"']), Tree("col", ['"B"'])]), "True"])
    name2, ok2 = qc._apply_has_columns(node2)
    assert name2 == "has_columns"
    assert ok2 is True


def test_apply_has_columns_returns_false_when_missing_column():
    df = pd.DataFrame({"A": [1], "b": [2]})
    qc = make_qc(df)
    node = Tree("has_columns", [Tree("cols", [Tree("col", ['"A"']), Tree("col", ['"missing"'])])])
    _, ok = qc._apply_has_columns(node)
    assert ok is False


def test_apply_has_one_of_without_ignore_case():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    qc = make_qc(df)
    node = Tree("has_one_of", [("color"), Tree("vals", [Tree("v", ['"red"']), Tree("v", ['"blue"'])])])
    # node.children[0] is the column name string, matching the implementation's expectations
    _, ok = qc._apply_has_one_of(node)
    assert ok is True


def test_apply_has_one_of_with_ignore_case():
    df = pd.DataFrame({"color": ["Red", "BLUE", "red"]})
    qc = make_qc(df)
    node = Tree("has_one_of", ["color", Tree("vals", [Tree("v", ['"red"']), Tree("v", ['"blue"'])]), "True"])
    _, ok = qc._apply_has_one_of(node)
    assert ok is True


def test_apply_value_length_ignore_nulls_false():
    df = pd.DataFrame({"code": ["aa", "bb", None]})
    qc = make_qc(df)
    node = Tree("value_length", ["code", "==", 2])
    _, ok = qc._apply_value_length(node)
    assert ok is False  # includes "None" string length, so lengths are not uniform


def test_apply_value_length_ignore_nulls_true():
    df = pd.DataFrame({"code": ["aa", "bb", None, "  "]})
    qc = make_qc(df)
    node = Tree("value_length", ["code", "True", "==", 2])
    _, ok = qc._apply_value_length(node)
    assert ok is True


def test_apply_percent_value_length_default_and_ignore_nulls():
    df = pd.DataFrame({"code": ["aa", "bb", "cc", None]})
    qc = make_qc(df)

    node = Tree("percent_value_length", ["code", 75, "==", 2])
    _, ok = qc._apply_percent_value_length(node)
    assert ok is True

    node2 = Tree("percent_value_length", ["code", 100, "True", "==", 2])
    _, ok2 = qc._apply_percent_value_length(node2)
    assert ok2 is True


def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"d": ["2020-01-01", "2021-12-31"]})
    qc = make_qc(df)
    node = Tree("is_date", ["d"])
    _, ok = qc._apply_date_validation(node)
    assert ok is True

    df_bad = pd.DataFrame({"d": ["2020-01-01", "not-a-date"]})
    qc_bad = make_qc(df_bad)
    node_bad = Tree("is_date", ["d"])
    msg, ok_bad = qc_bad._apply_date_validation(node_bad)
    assert ok_bad is False
    assert msg.startswith("Date parse error:")


def test_apply_check_row_count_and_not_null():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]})
    qc = make_qc(df)

    row_count = Tree("row_count", [">", 2])
    name, ok = qc._apply_check([row_count])
    assert name == "row_count" and ok is True

    not_null = Tree("not_null", ["a"])
    name2, ok2 = qc._apply_check([not_null])
    assert name2 == "not_null" and ok2 is True


def test_apply_check_unknown_check_raises_genie_exception():
    df = pd.DataFrame({"a": [1]})
    qc = make_qc(df)
    node = Tree("unsupported_rule", ["a"])
    with pytest.raises(GenieException, match="not implemented in the DSL"):
        qc._apply_check([node])


def test_apply_check_key_error_returns_message_and_false():
    df = pd.DataFrame({"a": [1]})
    qc = make_qc(df)
    node = Tree("not_null", ["missing"])
    msg, ok = qc._apply_check([node])
    assert ok is False
    assert msg == "Key error: not_null for missing"


def test_apply_predicates_inserts_results_in_reverse_order():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    qc = make_qc(df)

    block = code_block(
        Tree("row_count", [">", 2]),
        Tree("not_null", ["a"]),
        Tree("is_unique", ["b"]),
    )
    results = qc._apply_predicates([block])

    assert [r[0] for r in results] == ["is_unique", "not_null", "row_count"]
    assert all(r[1] is True for r in results)


def test_run_requires_single_child_and_unknown_instruction():
    df = pd.DataFrame({"a": [1]})
    qc = make_qc(df)

    with pytest.raises(GenieException, match="How is this possible"):
        qc._run(Tree("program", [predicates(), predicates()]))

    with pytest.raises(GenieException, match="Unknown instruction"):
        qc._run(Tree("program", [Tree("something_else", [])]))
