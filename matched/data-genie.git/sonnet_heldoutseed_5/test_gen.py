import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_tree(*checks):
    """Build a parse tree equivalent to start -> predicates -> code_block -> checks."""
    return Tree(
        "start",
        [Tree("predicates", [Tree("code_block", [Tree("check", [c]) for c in checks])])],
    )


def tok(s):
    return Token("STRING", '"%s"' % s)


def str_list(*values):
    return Tree("string_list", [tok(v) for v in values])


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "x", "y"]})


def test_empty_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_accessor_is_registered(df):
    assert isinstance(df.dqc, QualityChecker)


def test_comparator_to_fn():
    assert QualityChecker._comparator_to_fn(">", 3)(5) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("==", 3)(3) is True
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("<", 3)(4) is False
    assert QualityChecker._comparator_to_fn("!!", 3)(3) is False


def test_treat_column_name(df):
    qc = QualityChecker(df)
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_row_count(df):
    qc = QualityChecker(df)
    assert qc._apply_check([Tree("row_count", [">", "4"])]) == ("row_count", True)
    assert qc._apply_check([Tree("row_count", ["==", "4"])]) == ("row_count", False)
    assert qc._apply_check([Tree("row_count", ["<", "6"])]) == ("row_count", True)


def test_is_unique(df):
    qc = QualityChecker(df)
    assert qc._apply_check([Tree("is_unique", ["a"])]) == ("is_unique", True)
    assert qc._apply_check([Tree("is_unique", ["b"])]) == ("is_unique", False)


def test_not_null():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2], "b": [1, None]}))
    assert qc._apply_check([Tree("not_null", ["a"])]) == ("not_null", True)
    assert qc._apply_check([Tree("not_null", ["b"])]) == ("not_null", False)


def test_is_positive():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2], "b": [0, 2]}))
    assert bool(qc._apply_check([Tree("is_positive", ["a"])])[1]) is True
    assert bool(qc._apply_check([Tree("is_positive", ["b"])])[1]) is False


def test_missing_column_returns_key_error_result(df):
    qc = QualityChecker(df)
    msg, ok = qc._apply_check([Tree("is_unique", ["nope"])])
    assert ok is False
    assert msg.startswith("Key error")
    assert "is_unique" in msg and "nope" in msg


def test_unknown_check_raises_genie_exception(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._apply_check([Tree("something_weird", ["a"])])


def test_has_columns(df):
    qc = QualityChecker(df)
    ok = Tree("has_columns", [Tree("cols", [tok("a"), tok("b")])])
    missing = Tree("has_columns", [Tree("cols", [tok("a"), tok("zzz")])])
    assert qc._apply_check([ok]) == ("has_columns", True)
    assert qc._apply_check([missing]) == ("has_columns", False)


def test_has_columns_ignore_case():
    qc = QualityChecker(pd.DataFrame({"Alpha": [1], "Beta": [2]}))
    cols = Tree("cols", [tok("alpha"), tok("BETA")])
    assert qc._apply_check([Tree("has_columns", [cols])])[1] is False
    assert qc._apply_check([Tree("has_columns", [cols, "True"])])[1] is True


def test_has_one_of():
    qc = QualityChecker(pd.DataFrame({"c": ["A", "b", "A"]}))
    exact = Tree("has_one_of", ["c", str_list("A", "b", "z")])
    wrong_case = Tree("has_one_of", ["c", str_list("a", "B")])
    insensitive = Tree("has_one_of", ["c", str_list("a", "B"), "True"])
    not_allowed = Tree("has_one_of", ["c", str_list("A")])
    assert qc._apply_check([exact]) == ("has_one_of", True)
    assert qc._apply_check([wrong_case])[1] is False
    assert qc._apply_check([insensitive])[1] is True
    assert qc._apply_check([not_allowed])[1] is False


def test_quantile(df):
    qc = QualityChecker(df)
    assert qc._apply_check([Tree("quantile", ["a", "0.5", "==", "3"])]) == ("quantile", True)
    assert qc._apply_check([Tree("quantile", ["a", "0.5", ">", "3"])])[1] is False
    assert qc._apply_check([Tree("quantile", ["a", "1", "<", "10"])])[1] is True


def test_is_date():
    qc = QualityChecker(pd.DataFrame({"good": ["2020-01-01", "2021-05-06"],
                                      "bad": ["not a date", "also not"]}))
    assert qc._apply_check([Tree("is_date", ["good"])]) == ("is_date", True)
    msg, ok = qc._apply_check([Tree("is_date", ["bad"])])
    assert ok is False
    assert msg.startswith("Date parse error")


def test_value_length():
    qc = QualityChecker(pd.DataFrame({"a": ["abc", "def", "ghi"], "b": ["a", "bb", "ccc"]}))
    assert qc._apply_check([Tree("value_length", ["a", "==", "3"])]) == ("value_length", True)
    assert qc._apply_check([Tree("value_length", ["a", "<", "3"])])[1] is False
    # differing lengths are never accepted
    assert qc._apply_check([Tree("value_length", ["b", ">", "0"])])[1] is False


def test_value_length_ignore_nulls():
    qc = QualityChecker(pd.DataFrame({"a": ["abc", None, "def"]}))
    # Without ignoring nulls, None becomes "None" (length 4) -> mixed lengths
    assert qc._apply_check([Tree("value_length", ["a", "==", "3"])])[1] is False
    assert qc._apply_check([Tree("value_length", ["a", "True", "==", "3"])])[1] is True


def test_percent_value_length():
    df = pd.DataFrame({"a": ["abc", "abd", "ab", "abcd"]})
    qc = QualityChecker(df)
    assert qc._apply_check([Tree("percent_value_length", ["a", "50", "%", "3"])]) == (
        "percent_value_length", True)
    assert qc._apply_check([Tree("percent_value_length", ["a", "75", "%", "3"])])[1] is False


def test_run_results_are_in_reverse_order(df):
    qc = QualityChecker(df)
    tree = make_tree(
        Tree("row_count", ["==", "5"]),
        Tree("is_unique", ["b"]),
    )
    results = qc._run(tree)
    assert results == [("is_unique", False), ("row_count", True)]


def test_run_rejects_bad_trees(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("predicates", []), Tree("predicates", [])]))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("other", [])]))


def test_run_with_ignore_column_case(monkeypatch):
    df = pd.DataFrame({"Name": ["a", "b"], "AGE": [1, 2]})
    tree = make_tree(Tree("is_unique", ["name"]), Tree("not_null", ["age"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(df)
    results = qc.run("ignored", ignore_column_case=True)
    assert list(qc._obj.columns) == ["name", "age"]
    assert sorted(results) == [("is_unique", True), ("not_null", True)]


def test_run_without_ignore_case_reports_key_error(monkeypatch):
    df = pd.DataFrame({"Name": ["a", "b"]})
    tree = make_tree(Tree("is_unique", ["name"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    results = QualityChecker(df).run("ignored")
    assert len(results) == 1
    assert results[0][1] is False
    assert results[0][0].startswith("Key error")


def test_validate_spec_rejects_garbage():
    message, ok = QualityChecker.validate_spec("@@@ definitely !!! not a valid spec ###")
    assert ok is False
    assert isinstance(message, str) and message
