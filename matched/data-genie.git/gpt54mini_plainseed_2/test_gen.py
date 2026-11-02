import pytest
import pandas as pd

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


class DummyNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children


class DummyToken:
    def __init__(self, value):
        self.value = value


def make_qc(df):
    return QualityChecker(df.copy())


def test_validate_raises_on_empty_dataframe_with_all_nan_columns():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        QualityChecker(df)


def test_validate_allows_dataframe_with_at_least_one_non_null_column():
    df = pd.DataFrame({"a": [None, None], "b": [1, None]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)


def test_comparator_to_fn_gt_eq_lt_and_default():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    default = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True and gt(3) is False
    assert eq(3) is True and eq(4) is False
    assert lt(2) is True and lt(3) is False
    assert default(100) is False


def test_treat_column_name_respects_ignore_column_case_flag():
    qc = make_qc(pd.DataFrame({"Name": [1]}))
    assert qc._treat_column_name("Name") == "Name"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Name") == "name"


def test_apply_quantile_returns_true_for_matching_comparison():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    qc = make_qc(df)
    node = DummyNode("quantile", ["x", "0.5", ">", "2.5"])
    name, result = qc._apply_quantile(node)
    assert name == "quantile"
    assert result is True


def test_apply_quantile_returns_false_for_non_matching_comparison():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    qc = make_qc(df)
    node = DummyNode("quantile", ["x", "0.5", "<", "2.0"])
    _, result = qc._apply_quantile(node)
    assert result is False


def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"d": ["2020-01-01", "2021-12-31"]})
    qc = make_qc(df)
    ok_node = DummyNode("is_date", ["d"])
    name, result = qc._apply_date_validation(ok_node)
    assert name == "is_date"
    assert result is True

    bad_df = pd.DataFrame({"d": ["not-a-date", "2021-12-31"]})
    qc_bad = make_qc(bad_df)
    bad_node = DummyNode("is_date", ["d"])
    name, result = qc_bad._apply_date_validation(bad_node)
    assert name.startswith("Date parse error:")
    assert result is False


def test_apply_has_one_of_without_ignore_case():
    df = pd.DataFrame({"color": ["Red", "Blue"]})
    qc = make_qc(df)
    node = DummyNode(
        "has_one_of",
        ["color", DummyNode("values", [DummyToken('"Red"'), DummyToken('"Blue"')])],
    )
    _, result = qc._apply_has_one_of(node)
    assert result is True


def test_apply_has_one_of_with_ignore_case():
    df = pd.DataFrame({"color": ["Red", "blue"]})
    qc = make_qc(df)
    node = DummyNode(
        "has_one_of",
        [
            "color",
            DummyNode("values", [DummyToken('"red"'), DummyToken('"BLUE"')]),
            "True",
        ],
    )
    _, result = qc._apply_has_one_of(node)
    assert result is True


def test_apply_value_length_ignoring_nulls_false():
    df = pd.DataFrame({"code": ["abc", "def", "ghi"]})
    qc = make_qc(df)
    node = DummyNode("value_length", ["code", "==", "3"])
    _, result = qc._apply_value_length(node)
    assert result is True


def test_apply_value_length_ignoring_nulls_true():
    df = pd.DataFrame({"code": ["abc", None, "def"]})
    qc = make_qc(df)
    node = DummyNode("value_length", ["code", "True", "==", "3"])
    _, result = qc._apply_value_length(node)
    assert result is True


def test_apply_percent_value_length_without_ignoring_nulls():
    df = pd.DataFrame({"code": ["xx", "yy", "zzz", None]})
    qc = make_qc(df)
    node = DummyNode("percent_value_length", ["code", "50", "False", "2"])
    _, result = qc._apply_percent_value_length(node)
    assert result is True


def test_apply_has_columns_with_and_without_ignore_case():
    df = pd.DataFrame({"Name": [1], "Age": [2]})
    qc = make_qc(df)

    node = DummyNode(
        "has_columns",
        [DummyNode("columns", [DummyToken('"Name"'), DummyToken('"Age"')])],
    )
    _, result = qc._apply_has_columns(node)
    assert result is True

    node_ci = DummyNode(
        "has_columns",
        [DummyNode("columns", [DummyToken('"name"'), DummyToken('"age"')]), "True"],
    )
    _, result = qc._apply_has_columns(node_ci)
    assert result is True


def test_apply_check_dispatches_row_count_and_handles_key_error():
    df = pd.DataFrame({"x": [1, 2, 3]})
    qc = make_qc(df)

    row_count = DummyNode("row_count", [">", "2"])
    name, result = qc._apply_check([row_count])
    assert name == "row_count"
    assert result is True

    bad = DummyNode("not_null", ["missing_col"])
    name, result = qc._apply_check([bad])
    assert name == "Key error: not_null for missing_col"
    assert result is False


def test_apply_check_raises_genie_exception_for_unimplemented_check():
    df = pd.DataFrame({"x": [1]})
    qc = make_qc(df)
    node = DummyNode("does_not_exist", [])
    with pytest.raises(GenieException, match="does_not_exist seems to be not implemented"):
        qc._apply_check([node])


def test_apply_predicates_inserts_results_in_reverse_order():
    df = pd.DataFrame({"x": [1, 2, 3]})
    qc = make_qc(df)

    check1 = DummyNode("check", [DummyNode("row_count", [">", "2"])])
    check2 = DummyNode("check", [DummyNode("row_count", ["<", "10"])])

    predicate = DummyNode("code_block", [check1, check2])
    results = qc._apply_predicates([predicate])

    assert results == [("row_count", True), ("row_count", True)]
    # Because results are inserted at index 0, the second check is evaluated first in the list.
    assert len(results) == 2
