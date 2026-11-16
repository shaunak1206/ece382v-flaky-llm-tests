import numpy as np
import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def check(node):
    """Wrap a check node the way _apply_check expects (a list whose first item is the node)."""
    return [node]


def strings(*values):
    """Build lark-like tokens whose .value includes the quotes, as the grammar produces."""
    return [Token("STRING", '"{}"'.format(v)) for v in values]


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", None, "z"],
    })


@pytest.fixture
def qc(df):
    return QualityChecker(df.copy())


def test_init_rejects_all_nan_frame():
    df = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_init_rejects_empty_frame():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_init_sets_default_ignore_column_case(df):
    qc = QualityChecker(df)
    assert qc.ignore_column_case is False


def test_comparator_greater_than():
    fn = QualityChecker._comparator_to_fn(">", 5)
    assert fn(6) is True
    assert fn(5) is False


def test_comparator_equal_and_less_than():
    eq = QualityChecker._comparator_to_fn("==", 5)
    lt = QualityChecker._comparator_to_fn("<", 5)
    assert eq(5) is True and eq(4) is False
    assert lt(4) is True and lt(5) is False


def test_comparator_unknown_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 5)
    assert fn(5) is False
    assert fn(10) is False


def test_treat_column_name_respects_ignore_flag(qc):
    assert qc._treat_column_name("ABC") == "ABC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("ABC") == "abc"


def test_is_unique_pass_and_fail():
    qc = QualityChecker(pd.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]}))
    assert qc._apply_check(check(Tree("is_unique", ["u"])))[1]
    assert not qc._apply_check(check(Tree("is_unique", ["d"])))[1]


def test_not_null_detects_missing_values():
    qc = QualityChecker(pd.DataFrame({"full": [1, 2], "holes": [1, None]}))
    assert qc._apply_check(check(Tree("not_null", ["full"])))[1]
    assert not qc._apply_check(check(Tree("not_null", ["holes"])))[1]


def test_is_positive_requires_all_values_above_zero():
    qc = QualityChecker(pd.DataFrame({"p": [1, 2, 3], "z": [1, 0, 3]}))
    assert qc._apply_check(check(Tree("is_positive", ["p"])))[1]
    assert not qc._apply_check(check(Tree("is_positive", ["z"])))[1]


def test_row_count_comparison(df):
    qc = QualityChecker(df)
    name, ok = qc._apply_check(check(Tree("row_count", ["==", "3"])))
    assert name == "row_count"
    assert ok
    _, ok = qc._apply_check(check(Tree("row_count", [">", "3"])))
    assert not ok


def test_has_columns_subset_and_missing():
    qc = QualityChecker(pd.DataFrame({"a": [1], "b": [2]}))
    present = Tree("has_columns", [Tree("columns", strings("a", "b"))])
    missing = Tree("has_columns", [Tree("columns", strings("a", "z"))])
    assert qc._apply_check(check(present))[1]
    assert not qc._apply_check(check(missing))[1]


def test_has_columns_ignore_case():
    qc = QualityChecker(pd.DataFrame({"a": [1], "b": [2]}))
    node = Tree("has_columns", [Tree("columns", strings("A")), "True"])
    assert qc._apply_check(check(node))[1]

    node_strict = Tree("has_columns", [Tree("columns", strings("A"))])
    assert not qc._apply_check(check(node_strict))[1]


def test_has_one_of_pass_and_fail():
    qc = QualityChecker(pd.DataFrame({"a": ["x", "y", "x"]}))
    ok_node = Tree("has_one_of", ["a", Tree("values", strings("x", "y"))])
    bad_node = Tree("has_one_of", ["a", Tree("values", strings("x"))])
    assert qc._apply_check(check(ok_node))[1]
    assert not qc._apply_check(check(bad_node))[1]


def test_has_one_of_ignore_case():
    qc = QualityChecker(pd.DataFrame({"a": ["x"]}))
    strict = Tree("has_one_of", ["a", Tree("values", strings("X"))])
    lenient = Tree("has_one_of", ["a", Tree("values", strings("X")), "True"])
    assert not qc._apply_check(check(strict))[1]
    assert qc._apply_check(check(lenient))[1]


def test_quantile_median_comparison():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))
    passing = Tree("quantile", ["a", "0.5", "==", "3"])
    failing = Tree("quantile", ["a", "0.5", ">", "3"])
    name, ok = qc._apply_check(check(passing))
    assert name == "quantile"
    assert ok
    assert not qc._apply_check(check(failing))[1]


def test_is_date_accepts_valid_dates():
    qc = QualityChecker(pd.DataFrame({"d": ["2020-01-01", "2021-02-03"]}))
    name, ok = qc._apply_check(check(Tree("is_date", ["d"])))
    assert name == "is_date"
    assert ok is True


def test_is_date_reports_parse_error():
    qc = QualityChecker(pd.DataFrame({"d": ["not a date at all"]}))
    msg, ok = qc._apply_check(check(Tree("is_date", ["d"])))
    assert ok is False
    assert msg.startswith("Date parse error")
    assert "d" in msg


def test_value_length_exact_match():
    qc = QualityChecker(pd.DataFrame({"a": ["aa", "bb"]}))
    _, ok = qc._apply_check(check(Tree("value_length", ["a", "==", "2"])))
    assert ok
    qc2 = QualityChecker(pd.DataFrame({"a": ["aa", "bbb"]}))
    _, ok = qc2._apply_check(check(Tree("value_length", ["a", "==", "2"])))
    assert not ok


def test_value_length_ignore_nulls_skips_blank_values():
    df = pd.DataFrame({"a": ["aa", "", "bb"]})
    ignoring = Tree("value_length", ["a", "True", "==", "2"])
    _, ok = QualityChecker(df.copy())._apply_check(check(ignoring))
    assert ok


def test_value_length_counts_blank_values_when_not_ignored():
    df = pd.DataFrame({"a": ["aa", "", "bb"]})
    counting = Tree("value_length", ["a", "==", "2"])
    _, ok = QualityChecker(df.copy())._apply_check(check(counting))
    assert not ok


def test_percent_value_length_threshold():
    df = pd.DataFrame({"a": ["aa", "bb", "c", "dd"]})
    node_ok = Tree("percent_value_length", ["a", "75", "==", "2"])
    node_bad = Tree("percent_value_length", ["a", "80", "==", "2"])
    name, ok = QualityChecker(df.copy())._apply_check(check(node_ok))
    assert name == "percent_value_length"
    assert ok
    _, ok = QualityChecker(df.copy())._apply_check(check(node_bad))
    assert not ok


def test_unknown_check_raises(qc):
    with pytest.raises(GenieException):
        qc._apply_check(check(Tree("not_a_real_check", ["a"])))


def test_missing_column_returns_key_error_message(qc):
    msg, ok = qc._apply_check(check(Tree("is_unique", ["does_not_exist"])))
    assert ok is False
    assert msg.startswith("Key error")
    assert "does_not_exist" in msg


def test_predicates_results_are_reversed_per_block():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, 1, 2]})
    qc = QualityChecker(df)
    predicates = [
        Tree("code_block", [
            Tree("check", [Tree("is_unique", ["a"])]),
            Tree("check", [Tree("not_null", ["b"])]),
        ])
    ]
    results = qc._apply_predicates(predicates)
    # Each result is inserted at the front, so the last check comes first.
    assert [name for name, _ in results] == ["not_null", "is_unique"]
    assert results[0][1] is np.False_ or results[0][1] is False or not results[0][1]
    assert results[1][1]


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 1]})
    tree = Tree("start", [
        Tree("predicates", [
            Tree("code_block", [Tree("check", [Tree("is_unique", ["a"])])])
        ])
    ])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))

    qc = QualityChecker(df)
    results = qc.run("ignored spec", ignore_column_case=True)

    assert list(df.columns) == ["a", "b"]
    assert len(results) == 1
    name, ok = results[0]
    assert name == "is_unique"
    assert ok
