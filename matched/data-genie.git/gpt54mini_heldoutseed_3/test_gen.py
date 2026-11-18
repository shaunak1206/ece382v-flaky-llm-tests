import pandas as pd
import pytest
from lark import Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_node(data, *children):
    return Tree(data, list(children))


def make_check_node(data, *children):
    return [Tree(data, list(children))]


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        QualityChecker(df)


def test_validate_allows_dataframe_with_one_non_empty_column():
    df = pd.DataFrame({"a": [None, None], "b": [1, None]})
    qc = QualityChecker(df)
    assert qc._obj is df


def test_comparator_to_fn_gt_eq_lt_and_default():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    other = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(2) is False
    assert eq(3) is True
    assert eq(2) is False
    assert lt(2) is True
    assert lt(4) is False
    assert other(999) is False


def test_treat_column_name_respects_ignore_column_case():
    df = pd.DataFrame({"MixedCase": [1]})
    qc = QualityChecker(df)

    assert qc._treat_column_name("MixedCase") == "MixedCase"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MixedCase") == "mixedcase"


def test_apply_date_validation_success():
    df = pd.DataFrame({"date_col": ["2020-01-01", "2021-02-03"]})
    qc = QualityChecker(df)
    node = make_node("is_date", "date_col")

    name, ok = qc._apply_date_validation(node)
    assert name == "is_date"
    assert ok is True


def test_apply_date_validation_failure_returns_descriptive_message():
    df = pd.DataFrame({"date_col": ["2020-01-01", "not-a-date"]})
    qc = QualityChecker(df)
    node = make_node("is_date", "date_col")

    name, ok = qc._apply_date_validation(node)
    assert ok is False
    assert name == "Date parse error: is_date for date_col"


def test_apply_has_one_of_case_sensitive_and_insensitive():
    df = pd.DataFrame({"color": ["Red", "Blue"]})
    qc = QualityChecker(df)

    allowed = make_node("values", Tree("token", ['"Red"']), Tree("token", ['"Blue"']))
    node_sensitive = make_node("has_one_of", "color", allowed)
    name, ok = qc._apply_has_one_of(node_sensitive)
    assert name == "has_one_of"
    assert ok is True

    node_insensitive = make_node("has_one_of", "color", allowed, "True")
    name, ok = qc._apply_has_one_of(node_insensitive)
    assert name == "has_one_of"
    assert ok is True

    df2 = pd.DataFrame({"color": ["Red", "Green"]})
    qc2 = QualityChecker(df2)
    node_fail = make_node("has_one_of", "color", allowed)
    _, ok = qc2._apply_has_one_of(node_fail)
    assert ok is False


def test_apply_value_length_without_ignoring_nulls():
    df = pd.DataFrame({"col": ["ab", "cd", "ef"]})
    qc = QualityChecker(df)
    node = make_node("value_length", "col", "==", "2")

    name, ok = qc._apply_value_length(node)
    assert name == "value_length"
    assert ok is True


def test_apply_value_length_ignoring_nulls_and_spaces():
    df = pd.DataFrame({"col": ["ab", "  ", None, "cd"]})
    qc = QualityChecker(df)
    node = make_node("value_length", "col", "True", "==", "2")

    name, ok = qc._apply_value_length(node)
    assert name == "value_length"
    assert ok is True


def test_apply_percent_value_length_counts_only_matching_rows():
    df = pd.DataFrame({"col": ["ab", "cd", "xx", "long"]})
    qc = QualityChecker(df)
    node = make_node("percent_value_length", "col", "50", "==", "2")

    name, ok = qc._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert ok is True


def test_apply_percent_value_length_ignore_nulls_drops_blank_rows():
    df = pd.DataFrame({"col": ["ab", "", "  ", None]})
    qc = QualityChecker(df)
    node = make_node("percent_value_length", "col", "100", "True", "==", "2")

    name, ok = qc._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert ok is True


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "b": [2]})
    qc = QualityChecker(df)
    cols = make_node("cols", Tree("token", ['"A"']), Tree("token", ['"b"']))

    node_sensitive = make_node("has_columns", cols)
    name, ok = qc._apply_has_columns(node_sensitive)
    assert name == "has_columns"
    assert ok is True

    node_insensitive = make_node("has_columns", cols, "True")
    name, ok = qc._apply_has_columns(node_insensitive)
    assert name == "has_columns"
    assert ok is True


def test_apply_check_known_and_unknown_checks():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    qc = QualityChecker(df)

    row_count_check = make_check_node("row_count", "==", "3")
    assert qc._apply_check(row_count_check) == ("row_count", True)

    unique_check = make_check_node("is_unique", "a")
    assert qc._apply_check(unique_check) == ("is_unique", True)

    with pytest.raises(GenieException, match="not implemented"):
        qc._apply_check(make_check_node("totally_unknown", "a"))


def test_apply_check_keyerror_returns_false_with_message():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)

    result = qc._apply_check(make_check_node("is_unique", "missing"))
    assert result == ("Key error: is_unique for missing", False)


def test_apply_predicates_reverses_results_order():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    qc = QualityChecker(df)

    pred1 = Tree("code_block", [make_check_node("row_count", "==", "2")])
    pred2 = Tree("code_block", [make_check_node("is_unique", "b")])

    results = qc._apply_predicates([pred1, pred2])
    assert results == [("is_unique", True), ("row_count", True)]


def test_run_with_monkeypatched_parse_and_lowercasing_columns():
    df = pd.DataFrame({"Mixed": [1, 2], "Other": ["a", "b"]})
    qc = QualityChecker(df)

    ast = Tree("start", [Tree("predicates", [Tree("code_block", [make_check_node("row_count", "==", "2")])])])

    qc._parse_spec = lambda check_spec: ast
    result = qc.run("ignored-spec", ignore_column_case=True)

    assert result == [("row_count", True)]
    assert list(qc._obj.columns) == ["mixed", "other"]


def test_run_raises_for_invalid_ast_shape():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)

    with pytest.raises(GenieException, match="How is this possible"):
        qc._run(Tree("start", [Tree("predicates", []), Tree("predicates", [])]))
