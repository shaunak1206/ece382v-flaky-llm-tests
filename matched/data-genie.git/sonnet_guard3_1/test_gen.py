import pandas as pd
import pytest
from lark import Tree, Token

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(name, *children):
    return Tree("check", [Tree(name, list(children))])


def _program(*checks):
    return Tree("start", [Tree("predicates", [Tree("code_block", list(checks))])])


def _run_one(df, name, *children):
    qc = QualityChecker(df)
    results = qc._run(_program(_check(name, *children)))
    assert len(results) == 1
    return results[0]


def _strings(*values):
    return Tree("values", [Token("STRING", '"%s"' % v) for v in values])


def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_comparator_to_fn_operators():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    assert gt(4) and not gt(3)
    assert eq(3) and not eq(4)
    assert lt(2) and not lt(3)


def test_comparator_to_fn_unknown_is_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 3)
    assert fn(1) is False
    assert fn(3) is False


def test_row_count_checks():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert _run_one(df, "row_count", "==", "3") == ("row_count", True)
    assert _run_one(df, "row_count", ">", "3") == ("row_count", False)
    assert _run_one(df, "row_count", "<", "10") == ("row_count", True)


def test_is_unique():
    df = pd.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]})
    assert _run_one(df, "is_unique", "u") == ("is_unique", True)
    assert _run_one(df, "is_unique", "d") == ("is_unique", False)


def test_not_null():
    df = pd.DataFrame({"full": [1, 2, 3], "holes": [1.0, None, 3.0]})
    assert _run_one(df, "not_null", "full") == ("not_null", True)
    assert _run_one(df, "not_null", "holes") == ("not_null", False)


def test_is_positive():
    df = pd.DataFrame({"pos": [1, 2, 3], "mixed": [1, 0, 3]})
    assert _run_one(df, "is_positive", "pos") == ("is_positive", True)
    assert _run_one(df, "is_positive", "mixed") == ("is_positive", False)


def test_missing_column_returns_key_error_message():
    df = pd.DataFrame({"a": [1, 2]})
    msg, ok = _run_one(df, "is_unique", "missing")
    assert ok is False
    assert msg == "Key error: is_unique for missing"


def test_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"A": [1], "B": [2]})
    cols = Tree("cols", [Token("STRING", '"a"'), Token("STRING", '"b"')])
    assert _run_one(df, "has_columns", cols) == ("has_columns", False)
    df2 = pd.DataFrame({"A": [1], "B": [2]})
    assert _run_one(df2, "has_columns", cols, Token("BOOL", "True")) == ("has_columns", True)


def test_has_columns_exact_match_and_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    present = Tree("cols", [Token("STRING", '"a"'), Token("STRING", '"b"')])
    absent = Tree("cols", [Token("STRING", '"a"'), Token("STRING", '"z"')])
    assert _run_one(df, "has_columns", present) == ("has_columns", True)
    assert _run_one(df, "has_columns", absent) == ("has_columns", False)


def test_has_one_of():
    df = pd.DataFrame({"c": ["a", "b", "a"]})
    assert _run_one(df, "has_one_of", "c", _strings("a", "b", "z")) == ("has_one_of", True)
    assert _run_one(df, "has_one_of", "c", _strings("a")) == ("has_one_of", False)


def test_has_one_of_ignore_case():
    df = pd.DataFrame({"c": ["A", "b"]})
    allowed = _strings("a", "b")
    assert _run_one(df, "has_one_of", "c", allowed) == ("has_one_of", False)
    df2 = pd.DataFrame({"c": ["A", "b"]})
    assert _run_one(df2, "has_one_of", "c", allowed, Token("BOOL", "True")) == ("has_one_of", True)


def test_quantile():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    assert _run_one(df, "quantile", "v", "0.5", "==", "3") == ("quantile", True)
    assert _run_one(df, "quantile", "v", "0.5", ">", "10") == ("quantile", False)
    assert _run_one(df, "quantile", "v", "0.5", "<", "10") == ("quantile", True)


def test_is_date_valid_and_invalid():
    good = pd.DataFrame({"d": ["2020-01-01", "2021-12-31"]})
    assert _run_one(good, "is_date", "d") == ("is_date", True)
    bad = pd.DataFrame({"d": ["notadate", "alsobad"]})
    msg, ok = _run_one(bad, "is_date", "d")
    assert ok is False
    assert msg == "Date parse error: is_date for d"


def test_value_length_uniform_and_mixed():
    df = pd.DataFrame({"a": ["xx", "yy", "zz"]})
    assert _run_one(df, "value_length", "a", "==", "2") == ("value_length", True)
    assert _run_one(df, "value_length", "a", ">", "2") == ("value_length", False)
    mixed = pd.DataFrame({"a": ["x", "yy"]})
    assert _run_one(mixed, "value_length", "a", "==", "2") == ("value_length", False)


def test_value_length_ignore_nulls():
    df = pd.DataFrame({"a": ["xx", "yy", "", " "]})
    assert _run_one(df, "value_length", "a", "False", "==", "2") == ("value_length", False)
    df2 = pd.DataFrame({"a": ["xx", "yy", "", " "]})
    assert _run_one(df2, "value_length", "a", "True", "==", "2") == ("value_length", True)


def test_percent_value_length():
    df = pd.DataFrame({"a": ["xx", "xx", "xxx", "xx"]})
    assert _run_one(df, "percent_value_length", "a", "75", "==", "2") == ("percent_value_length", True)
    df2 = pd.DataFrame({"a": ["xx", "xx", "xxx", "xx"]})
    assert _run_one(df2, "percent_value_length", "a", "80", "==", "2") == ("percent_value_length", False)


def test_results_are_in_reverse_order_of_checks():
    df = pd.DataFrame({"a": [1, 2, 3]})
    qc = QualityChecker(df)
    results = qc._run(_program(
        _check("row_count", "==", "3"),
        _check("is_unique", "a"),
        _check("is_positive", "a"),
    ))
    assert [name for name, _ in results] == ["is_positive", "is_unique", "row_count"]
    assert all(bool(ok) for _, ok in results)


def test_unimplemented_check_raises_genie_exception():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._run(_program(_check("bogus", "a")))


def test_run_rejects_bad_tree_shapes():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("other", [])]))


def test_run_with_patched_parser_and_ignore_column_case(monkeypatch):
    df = pd.DataFrame({"NAME": ["a", "b"]})
    tree = _program(_check("not_null", "NAME"))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(df)
    results = qc.run("ignored", ignore_column_case=True)
    assert results == [("not_null", True)]
    assert list(df.columns) == ["name"]


def test_validate_spec_reports_success_and_failure(monkeypatch):
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: Tree("start", [])))
    assert QualityChecker.validate_spec("anything") == (None, True)

    def boom(spec):
        raise ValueError("bad spec")

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(boom))
    msg, ok = QualityChecker.validate_spec("anything")
    assert ok is False
    assert msg == "bad spec"
