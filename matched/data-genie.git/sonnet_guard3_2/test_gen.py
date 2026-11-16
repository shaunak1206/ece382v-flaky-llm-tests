import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(name, *children):
    """Build the argument list that QualityChecker._apply_check expects."""
    return [Tree(name, list(children))]


def _strings(*values):
    return Tree("values", [Token("STRING", '"%s"' % v) for v in values])


def _program(*checks):
    """Build a full predicates tree from (name, children) pairs."""
    blocks = [Tree("check", [Tree(name, list(children))]) for name, children in checks]
    return Tree("start", [Tree("predicates", [Tree("code_block", blocks)])])


# ---------------------------------------------------------------- construction

def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_valid_dataframe_defaults():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    assert qc.ignore_column_case is False


# ---------------------------------------------------------------- comparators

def test_comparator_to_fn_all_operators():
    fn_gt = QualityChecker._comparator_to_fn(">", 5)
    fn_eq = QualityChecker._comparator_to_fn("==", 5)
    fn_lt = QualityChecker._comparator_to_fn("<", 5)
    assert fn_gt(6) and not fn_gt(5)
    assert fn_eq(5) and not fn_eq(4)
    assert fn_lt(4) and not fn_lt(5)


def test_comparator_to_fn_unknown_is_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 5)
    assert fn(1) is False
    assert fn(5) is False


# ---------------------------------------------------------------- individual checks

def test_row_count():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    assert qc._apply_check(_check("row_count", ">", "2")) == ("row_count", True)
    assert qc._apply_check(_check("row_count", "==", "3")) == ("row_count", True)
    assert qc._apply_check(_check("row_count", "<", "3")) == ("row_count", False)


def test_has_columns_case_sensitive_and_insensitive():
    qc = QualityChecker(pd.DataFrame({"Name": [1], "Age": [2]}))
    ok = qc._apply_check(_check("has_columns", _strings("Name", "Age")))
    assert ok == ("has_columns", True)
    miss = qc._apply_check(_check("has_columns", _strings("name")))
    assert miss == ("has_columns", False)
    ci = qc._apply_check(_check("has_columns", _strings("name", "AGE"), "True"))
    assert ci == ("has_columns", True)
    missing = qc._apply_check(_check("has_columns", _strings("Name", "Other")))
    assert missing == ("has_columns", False)


def test_is_unique():
    qc = QualityChecker(pd.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]}))
    assert bool(qc._apply_check(_check("is_unique", "u"))[1]) is True
    assert bool(qc._apply_check(_check("is_unique", "d"))[1]) is False


def test_not_null():
    qc = QualityChecker(pd.DataFrame({"full": [1, 2], "holes": [1.0, None]}))
    assert bool(qc._apply_check(_check("not_null", "full"))[1]) is True
    assert bool(qc._apply_check(_check("not_null", "holes"))[1]) is False


def test_is_positive():
    qc = QualityChecker(pd.DataFrame({"pos": [1, 2, 3], "mixed": [1, 0, 3]}))
    assert bool(qc._apply_check(_check("is_positive", "pos"))[1]) is True
    assert bool(qc._apply_check(_check("is_positive", "mixed"))[1]) is False


def test_has_one_of_with_and_without_ignore_case():
    qc = QualityChecker(pd.DataFrame({"c": ["A", "b", "A"]}))
    strict = qc._apply_check(_check("has_one_of", "c", _strings("A", "b", "z")))
    assert strict == ("has_one_of", True)
    strict_fail = qc._apply_check(_check("has_one_of", "c", _strings("a", "b")))
    assert strict_fail == ("has_one_of", False)
    ci = qc._apply_check(_check("has_one_of", "c", _strings("a", "B"), "True"))
    assert ci == ("has_one_of", True)
    not_ci = qc._apply_check(_check("has_one_of", "c", _strings("a", "B"), "False"))
    assert not_ci == ("has_one_of", False)


def test_quantile():
    qc = QualityChecker(pd.DataFrame({"n": [1, 2, 3, 4, 5]}))
    assert qc._apply_check(_check("quantile", "n", "0.5", "==", "3")) == ("quantile", True)
    assert qc._apply_check(_check("quantile", "n", "0.5", ">", "3")) == ("quantile", False)
    assert qc._apply_check(_check("quantile", "n", "1.0", ">", "4")) == ("quantile", True)


def test_is_date_valid_and_invalid():
    qc = QualityChecker(pd.DataFrame({
        "good": ["2020-01-01", "2021-12-31"],
        "bad": ["definitely not", "a date"],
    }))
    assert qc._apply_check(_check("is_date", "good")) == ("is_date", True)
    name, ok = qc._apply_check(_check("is_date", "bad"))
    assert ok is False
    assert name.startswith("Date parse error")
    assert "bad" in name


def test_value_length_without_ignore_nulls_arg():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "def"], "m": ["a", "abc"]}))
    assert qc._apply_check(_check("value_length", "s", "==", "3")) == ("value_length", True)
    assert qc._apply_check(_check("value_length", "s", ">", "3")) == ("value_length", False)
    # Different lengths -> never a single unique length
    assert qc._apply_check(_check("value_length", "m", ">", "0")) == ("value_length", False)


def test_value_length_ignore_nulls_skips_blank_values():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "", "def", "   "]}))
    ignoring = qc._apply_check(_check("value_length", "s", "True", "==", "3"))
    assert ignoring == ("value_length", True)
    not_ignoring = qc._apply_check(_check("value_length", "s", "False", "==", "3"))
    assert not_ignoring == ("value_length", False)


def test_percent_value_length_thresholds():
    df = pd.DataFrame({"s": ["abc", "abcd", "abc", "abc"]})
    qc = QualityChecker(df)
    assert qc._apply_check(_check("percent_value_length", "s", "75", "==", "3")) == (
        "percent_value_length", True)
    qc2 = QualityChecker(pd.DataFrame({"s": ["abc", "abcd", "abc", "abc"]}))
    assert qc2._apply_check(_check("percent_value_length", "s", "80", "==", "3")) == (
        "percent_value_length", False)


def test_percent_value_length_with_ignore_nulls_flag():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "", "abc", "abc"]}))
    result = qc._apply_check(
        _check("percent_value_length", "s", "75", "True", "==", "3"))
    assert result == ("percent_value_length", True)


# ---------------------------------------------------------------- error handling

def test_missing_column_returns_key_error_message():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    name, ok = qc._apply_check(_check("is_unique", "missing"))
    assert ok is False
    assert name == "Key error: is_unique for missing"


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException):
        qc._apply_check(_check("no_such_check", "a"))


# ---------------------------------------------------------------- _run / run

def test_run_tree_results_are_in_reverse_order():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    tree = _program(("row_count", [">", "1"]), ("not_null", ["a"]), ("is_unique", ["a"]))
    results = qc._run(tree)
    assert [name for name, _ in results] == ["is_unique", "not_null", "row_count"]
    assert all(bool(ok) for _, ok in results)


def test_run_rejects_bad_trees():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("something_else", [])]))


def test_run_uses_parsed_spec_and_ignore_column_case(monkeypatch):
    tree = _program(("not_null", ["Name"]), ("row_count", ["==", "2"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(pd.DataFrame({"Name": ["x", "y"]}))
    results = qc.run("ignored spec", ignore_column_case=True)
    assert qc.ignore_column_case is True
    assert list(qc._obj.columns) == ["name"]
    assert [name for name, _ in results] == ["row_count", "not_null"]
    assert all(bool(ok) for _, ok in results)


def test_run_case_sensitive_reports_key_error_for_wrong_case(monkeypatch):
    tree = _program(("not_null", ["name"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(pd.DataFrame({"Name": ["x", "y"]}))
    results = qc.run("ignored spec")
    assert len(results) == 1
    assert results[0][1] is False
    assert results[0][0].startswith("Key error")


def test_validate_spec_rejects_garbage():
    message, ok = QualityChecker.validate_spec("@@@ ### !!! this is not a spec ###")
    assert ok is False
    assert isinstance(message, str)
