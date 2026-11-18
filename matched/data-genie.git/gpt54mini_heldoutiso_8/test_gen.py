import pandas as pd
import pytest
from types import SimpleNamespace
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_node(data, *children):
    return SimpleNamespace(data=data, children=list(children))


def make_check_node(node):
    return SimpleNamespace(children=[node])


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError, match="DataFrame is empty!!!"):
        _ = df.dqc


def test_comparator_to_fn_gt_eq_lt_and_default():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    other = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(3) is False
    assert eq(3) is True
    assert eq(2) is False
    assert lt(2) is True
    assert lt(3) is False
    assert other(999) is False


def test_treat_column_name_respects_ignore_case_flag():
    df = pd.DataFrame({"A": [1]})
    qc = df.dqc

    assert qc._treat_column_name("A") == "A"
    qc.ignore_column_case = True
    assert qc._treat_column_name("A") == "a"


def test_apply_check_row_count_and_not_null_and_positive():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 3]})
    qc = df.dqc

    row_count = make_check_node(make_node("row_count", ">", "2"))
    not_null = make_check_node(make_node("not_null", "x"))
    positive = make_check_node(make_node("is_positive", "x"))

    assert qc._apply_check(row_count) == ("row_count", True)
    assert qc._apply_check(not_null) == ("not_null", True)
    assert qc._apply_check(positive) == ("is_positive", True)


def test_apply_check_is_unique_false_for_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2]})
    qc = df.dqc
    node = make_check_node(make_node("is_unique", "x"))

    assert qc._apply_check(node) == ("is_unique", False)


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "b": [2]})
    qc = df.dqc

    cols = SimpleNamespace(children=[SimpleNamespace(value='"A"'), SimpleNamespace(value='"b"')])
    node = make_node("has_columns", cols)
    assert qc._apply_has_columns(node) == ("has_columns", True)

    missing_case = make_node(
        "has_columns",
        cols,
        "False",
    )
    # Same expectation as case-sensitive; included to exercise parsing of 2-child branch
    assert qc._apply_has_columns(missing_case) == ("has_columns", True)

    insensitive_cols = SimpleNamespace(
        children=[SimpleNamespace(value='"a"'), SimpleNamespace(value='"B"')]
    )
    insensitive_node = make_node("has_columns", insensitive_cols, "True")
    assert qc._apply_has_columns(insensitive_node) == ("has_columns", True)


def test_apply_has_one_of_with_default_case_sensitive_and_ignore_case():
    df = pd.DataFrame({"color": ["Red", "Blue", "Red"]})
    qc = df.dqc

    allowed = SimpleNamespace(
        children=[SimpleNamespace(value='"Red"'), SimpleNamespace(value='"Blue"')]
    )
    node = make_node("has_one_of", "color", allowed)
    assert qc._apply_has_one_of(node) == ("has_one_of", True)

    allowed_mixed = SimpleNamespace(
        children=[SimpleNamespace(value='"red"'), SimpleNamespace(value='"blue"')]
    )
    insensitive_node = make_node("has_one_of", "color", allowed_mixed, "True")
    assert qc._apply_has_one_of(insensitive_node) == ("has_one_of", True)


def test_apply_has_one_of_false_when_unexpected_values_exist():
    df = pd.DataFrame({"color": ["Red", "Green"]})
    qc = df.dqc

    allowed = SimpleNamespace(children=[SimpleNamespace(value='"Red"')])
    node = make_node("has_one_of", "color", allowed)
    assert qc._apply_has_one_of(node) == ("has_one_of", False)


def test_apply_quantile_greater_than_threshold():
    df = pd.DataFrame({"score": [1, 2, 3, 4, 100]})
    qc = df.dqc
    node = make_node("quantile", "score", "0.5", ">", "3")

    assert qc._apply_quantile(node) == ("quantile", True)


def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"dt": ["2021-01-01", "2022-02-02"]})
    qc = df.dqc

    ok = make_node("is_date", "dt")
    assert qc._apply_date_validation(ok) == ("is_date", True)

    bad_df = pd.DataFrame({"dt": ["2021-01-01", "not-a-date"]})
    bad_qc = bad_df.dqc
    bad = make_node("is_date", "dt")
    result = bad_qc._apply_date_validation(bad)
    assert result[0].startswith("Date parse error: is_date for dt")
    assert result[1] is False


def test_apply_value_length_ignoring_and_not_ignoring_nulls():
    df = pd.DataFrame({"name": ["aa", "bb", None]})
    qc = df.dqc

    node_no_ignore = make_node("value_length", "name", "==", "2")
    assert qc._apply_value_length(node_no_ignore) == ("value_length", False)

    node_ignore = make_node("value_length", "name", "True", "==", "2")
    assert qc._apply_value_length(node_ignore) == ("value_length", True)


def test_apply_percent_value_length_passes_with_expected_percentage():
    df = pd.DataFrame({"code": ["xx", "yy", "zz", "a"]})
    qc = df.dqc

    node = make_node("percent_value_length", "code", "75", "False", "2")
    assert qc._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_percent_value_length_with_ignore_nulls_drops_blank_rows():
    df = pd.DataFrame({"code": ["xx", "  ", None, "yy"]})
    qc = df.dqc

    node = make_node("percent_value_length", "code", "50", "True", "2")
    result = qc._apply_percent_value_length(node)
    assert result == ("percent_value_length", True)
    assert "length" in qc._obj.columns


def test_apply_check_unknown_instruction_raises_genie_exception():
    df = pd.DataFrame({"x": [1]})
    qc = df.dqc
    node = make_check_node(make_node("something_new", "x"))

    with pytest.raises(GenieException, match="something_new seems to be not implemented"):
        qc._apply_check(node)


def test_apply_check_key_error_returns_false_tuple():
    df = pd.DataFrame({"x": [1]})
    qc = df.dqc
    node = make_check_node(make_node("not_null", "missing"))

    assert qc._apply_check(node) == ("Key error: not_null for missing", False)


def test_apply_predicates_preserves_reverse_insertion_order():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    qc = df.dqc

    p1 = make_node("code_block", make_check_node(make_node("row_count", ">", "0")))
    p2 = make_node("code_block", make_check_node(make_node("not_null", "x")))
    results = qc._apply_predicates([p1, p2])

    assert results == [("not_null", True), ("row_count", True)]


def test_run_uses_parser_and_lowercases_columns_when_requested(monkeypatch):
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = df.dqc

    dummy_ast = Tree("root", [Tree("predicates", [])])
    seen = {}

    def fake_parse_spec(spec):
        seen["spec"] = spec
        return dummy_ast

    def fake_run(ast):
        seen["ast"] = ast
        return [("ok", True)]

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(fake_parse_spec))
    monkeypatch.setattr(QualityChecker, "_run", fake_run)

    result = qc.run("dummy spec", ignore_column_case=True)

    assert result == [("ok", True)]
    assert seen["spec"] == "dummy spec"
    assert seen["ast"] == dummy_ast
    assert list(qc._obj.columns) == ["a", "b"]
