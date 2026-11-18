import pytest
import pandas as pd
from lark import Tree, Token

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_check_node(data, children):
    return Tree(data, children)


def make_wrapped_check(node):
    return Tree("check", [node])


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        _ = df.dqc


def test_comparator_to_fn_supports_gt_eq_lt_and_default_false():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    other = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(3) is False
    assert eq(3) is True
    assert eq(4) is False
    assert lt(2) is True
    assert lt(3) is False
    assert other(100) is False


def test_treat_column_name_respects_ignore_column_case():
    df = pd.DataFrame({"Name": ["a"]})
    qc = df.dqc
    assert qc._treat_column_name("Name") == "Name"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Name") == "name"


def test_apply_check_row_count_passes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    qc = df.dqc
    node = make_check_node("row_count", [">", "2"])
    assert qc._apply_check([node]) == ("row_count", True)


def test_apply_check_row_count_fails():
    df = pd.DataFrame({"x": [1, 2, 3]})
    qc = df.dqc
    node = make_check_node("row_count", ["<", "3"])
    assert qc._apply_check([node]) == ("row_count", False)


def test_apply_check_has_columns_with_case_insensitive_match():
    df = pd.DataFrame({"Name": ["a"], "Age": [1]})
    qc = df.dqc
    node = make_check_node(
        "has_columns",
        [make_check_node("column_names", [Token("STRING", '"name"'), Token("STRING", '"AGE"')]), "True"],
    )
    assert qc._apply_check([node]) == ("has_columns", True)


def test_apply_check_is_unique_and_not_null():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    qc = df.dqc

    unique_node = make_check_node("is_unique", ["id"])
    not_null_node = make_check_node("not_null", ["name"])

    assert qc._apply_check([unique_node]) == ("is_unique", True)
    assert qc._apply_check([not_null_node]) == ("not_null", True)


def test_apply_check_is_positive_false_for_negative_values():
    df = pd.DataFrame({"score": [1, 0, -1]})
    qc = df.dqc
    node = make_check_node("is_positive", ["score"])
    assert qc._apply_check([node]) == ("is_positive", False)


def test_apply_check_has_one_of_case_sensitive_and_insensitive():
    df = pd.DataFrame({"color": ["Red", "Blue"]})
    qc = df.dqc

    allowed = make_check_node(
        "allowed_values",
        [Token("STRING", '"Red"'), Token("STRING", '"Blue"')],
    )
    node_sensitive = make_check_node("has_one_of", ["color", allowed])
    assert qc._apply_check([node_sensitive]) == ("has_one_of", True)

    df2 = pd.DataFrame({"color": ["Red", "Blue", "red"]})
    qc2 = df2.dqc
    node_insensitive = make_check_node("has_one_of", ["color", allowed, "True"])
    assert qc2._apply_check([node_insensitive]) == ("has_one_of", True)


def test_apply_check_quantile_uses_comparator():
    df = pd.DataFrame({"value": [10, 20, 30, 40]})
    qc = df.dqc
    node = make_check_node("quantile", ["value", "0.5", ">", "19"])
    assert qc._apply_check([node]) == ("quantile", True)


def test_apply_check_is_date_success_and_failure():
    df = pd.DataFrame({"when": ["2020-01-01", "2020-02-01"]})
    qc = df.dqc
    ok = make_check_node("is_date", ["when"])
    assert qc._apply_check([ok]) == ("is_date", True)

    df_bad = pd.DataFrame({"when": ["2020-01-01", "not-a-date"]})
    qc_bad = df_bad.dqc
    bad = make_check_node("is_date", ["when"])
    result = qc_bad._apply_check([bad])
    assert result[0].startswith("Date parse error:")
    assert result[1] is False


def test_apply_check_value_length_without_ignore_nulls():
    df = pd.DataFrame({"code": ["ab", "cd", None]})
    qc = df.dqc
    node = make_check_node("value_length", ["code", "==", "2"])
    assert qc._apply_check([node]) == ("value_length", False)


def test_apply_check_value_length_with_ignore_nulls():
    df = pd.DataFrame({"code": ["ab", "cd", None]})
    qc = df.dqc
    node = make_check_node("value_length", ["code", "True", "==", "2"])
    assert qc._apply_check([node]) == ("value_length", True)


def test_apply_check_percent_value_length_with_and_without_ignore_nulls():
    df = pd.DataFrame({"code": ["aa", "bb", "cc", "d", None]})
    qc = df.dqc

    node_no_ignore = make_check_node("percent_value_length", ["code", "60", "False", "2"])
    assert qc._apply_check([node_no_ignore]) == ("percent_value_length", True)

    node_ignore = make_check_node("percent_value_length", ["code", "60", "True", "2"])
    assert qc._apply_check([node_ignore]) == ("percent_value_length", False)


def test_apply_check_unknown_raises_genieexception():
    df = pd.DataFrame({"x": [1]})
    qc = df.dqc
    node = make_check_node("some_new_check", ["x"])
    with pytest.raises(GenieException, match="not implemented"):
        qc._apply_check([node])


def test_apply_check_missing_column_returns_key_error_message():
    df = pd.DataFrame({"x": [1]})
    qc = df.dqc
    node = make_check_node("is_unique", ["missing"])
    assert qc._apply_check([node]) == ("Key error: is_unique for missing", False)


def test_apply_predicates_returns_results_in_reverse_order():
    df = pd.DataFrame({"x": [1, 2, 3]})
    qc = df.dqc
    first = make_wrapped_check(make_check_node("row_count", [">", "1"]))
    second = make_wrapped_check(make_check_node("is_positive", ["x"]))

    results = qc._apply_predicates([first, second])
    assert results == [("is_positive", True), ("row_count", True)]


def test_run_applies_spec_and_lowercases_columns_when_requested():
    df = pd.DataFrame({"Name": ["Alice", "Bob"]})
    qc = df.dqc

    ast = Tree("root", [Tree("predicates", [make_wrapped_check(make_check_node("is_unique", ["Name"]))])])
    qc._parse_spec = staticmethod(lambda spec: ast)

    results = qc.run("ignored", ignore_column_case=True)
    assert results == [("is_unique", False)]
    assert list(qc._obj.columns) == ["name"]
