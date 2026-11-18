import pytest
import pandas as pd
import numpy as np
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


class DummyToken:
    def __init__(self, value):
        self.value = value


class DummyNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children


def make_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "x"],
            "C": ["2020-01-01", "2020-02-01", "not-a-date"],
            "D": ["aa", "bb", "cc"],
            "E": ["", "  ", "abc"],
        }
    )


def test_validate_raises_on_all_nan_dataframe():
    df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})
    with pytest.raises(AttributeError, match="DataFrame is empty!!!"):
        QualityChecker(df)


def test_validate_allows_dataframe_with_one_non_nan_column():
    df = pd.DataFrame({"A": [np.nan, np.nan], "B": [1, 2]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)


def test_comparator_to_fn_greater_equal_less_and_unknown():
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
    assert other(999) is False


def test_treat_column_name_respects_ignore_column_case_flag():
    qc = QualityChecker(make_df())
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_apply_quantile_returns_expected_boolean():
    df = pd.DataFrame({"A": [1, 2, 3, 4]})
    qc = QualityChecker(df)
    node = DummyNode("quantile", ["A", "0.5", ">", "2"])
    name, passed = qc._apply_quantile(node)
    assert name == "quantile"
    assert passed is True


def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({"C": ["2020-01-01", "2020-02-01", "not-a-date"]})
    qc = QualityChecker(df)

    ok_node = DummyNode("is_date", ["C"])
    name, passed = qc._apply_date_validation(ok_node)
    assert name == "Date parse error: is_date for C" or name == "is_date"
    assert passed is False

    df2 = pd.DataFrame({"C": ["2020-01-01", "2020-02-01"]})
    qc2 = QualityChecker(df2)
    name2, passed2 = qc2._apply_date_validation(ok_node)
    assert name2 == "is_date"
    assert passed2 is True


def test_apply_has_one_of_default_case_sensitive():
    df = pd.DataFrame({"B": ["x", "y", "x"]})
    qc = QualityChecker(df)
    node = DummyNode(
        "has_one_of",
        ["B", DummyNode("values", [DummyToken('"x"'), DummyToken('"y"')])],
    )
    name, passed = qc._apply_has_one_of(node)
    assert name == "has_one_of"
    assert passed is True


def test_apply_has_one_of_ignore_case_true():
    df = pd.DataFrame({"B": ["x", "Y", "x"]})
    qc = QualityChecker(df)
    node = DummyNode(
        "has_one_of",
        ["B", DummyNode("values", [DummyToken('"x"'), DummyToken('"y"')]), "True"],
    )
    name, passed = qc._apply_has_one_of(node)
    assert name == "has_one_of"
    assert passed is True


def test_apply_value_length_without_ignore_nulls():
    df = pd.DataFrame({"D": ["aa", "bb", "cc"]})
    qc = QualityChecker(df)
    node = DummyNode("value_length", ["D", "==", "2"])
    name, passed = qc._apply_value_length(node)
    assert name == "value_length"
    assert passed is True


def test_apply_value_length_with_ignore_nulls():
    df = pd.DataFrame({"D": ["aa", None, "  "]})
    qc = QualityChecker(df)
    node = DummyNode("value_length", ["D", "True", "==", "2"])
    name, passed = qc._apply_value_length(node)
    assert name == "value_length"
    assert passed is True


def test_apply_percent_value_length_counts_passing_rows():
    df = pd.DataFrame({"D": ["aa", "bb", "zz", "ccc"]})
    qc = QualityChecker(df)
    node = DummyNode("percent_value_length", ["D", "75", "==", "2"])
    name, passed = qc._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert passed is True
    assert "length" in qc._obj.columns


def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = QualityChecker(df)

    node = DummyNode("has_columns", [DummyNode("columns", [DummyToken('"A"'), DummyToken('"B"')])])
    name, passed = qc._apply_has_columns(node)
    assert name == "has_columns"
    assert passed is True

    node2 = DummyNode(
        "has_columns",
        [DummyNode("columns", [DummyToken('"a"'), DummyToken('"b"')]), "True"],
    )
    name2, passed2 = qc._apply_has_columns(node2)
    assert name2 == "has_columns"
    assert passed2 is True


def test_apply_check_row_count_and_is_unique_and_not_null_and_positive():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [10, 20, 30],
        }
    )
    qc = QualityChecker(df)

    row_node = DummyNode("row_count", [">", "2"])
    unique_node = DummyNode("is_unique", ["B"])
    not_null_node = DummyNode("not_null", ["A"])
    positive_node = DummyNode("is_positive", ["C"])

    assert qc._apply_check([row_node]) == ("row_count", True)
    assert qc._apply_check([unique_node]) == ("is_unique", True)
    assert qc._apply_check([not_null_node]) == ("not_null", True)
    assert qc._apply_check([positive_node]) == ("is_positive", True)


def test_apply_check_unknown_raises_genie_exception():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException, match="not implemented in the DSL"):
        qc._apply_check([DummyNode("unknown_check", [])])


def test_apply_check_keyerror_returns_message_and_false():
    qc = QualityChecker(make_df())
    bad_node = DummyNode("is_unique", ["missing_column"])
    msg, passed = qc._apply_check([bad_node])
    assert passed is False
    assert msg == "Key error: is_unique for missing_column"


def test_apply_predicates_inserts_results_in_reverse_order():
    qc = QualityChecker(pd.DataFrame({"A": [1, 2, 3]}))

    check1 = DummyNode("check", [DummyNode("row_count", [">", "2"])])
    check2 = DummyNode("check", [DummyNode("is_positive", ["A"])])
    code_block = DummyNode("code_block", [check1, check2])

    results = qc._apply_predicates([code_block])
    assert results == [("is_positive", True), ("row_count", True)]


def test_run_delegates_to_parsed_tree_and_applies_ignore_column_case(monkeypatch):
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = QualityChecker(df)

    ast = Tree("root", [Tree("predicates", [])])

    def fake_parse_spec(spec):
        assert spec == "dummy-spec"
        return ast

    def fake_run(parsed_ast):
        assert parsed_ast == ast
        return [("ok", True)]

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(fake_parse_spec))
    monkeypatch.setattr(QualityChecker, "_run", fake_run)

    result = qc.run("dummy-spec", ignore_column_case=True)
    assert result == [("ok", True)]
    assert qc.ignore_column_case is True
    assert list(qc._obj.columns) == ["a", "b"]
