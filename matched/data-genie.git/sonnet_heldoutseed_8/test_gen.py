import pandas as pd
import pytest
from lark import Tree

import genie_pkg.dqc  # noqa: F401  (registers the accessor)
from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def check(name, *children):
    """Build a check wrapped the way _apply_check expects it (a list holding one Tree)."""
    return [Tree(name, list(children))]


def strings(*values):
    """Build a Tree of quoted string tokens, like the grammar would produce."""
    from lark import Token
    return Tree("items", [Token("STRING", '"%s"' % v) for v in values])


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["abc", "def", "ghi", "jkl"],
        "score": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def qc(df):
    return QualityChecker(df)


def test_validate_rejects_empty_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_validate_rejects_all_nan_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None], "b": [None, None]}))


def test_accessor_is_registered(df):
    assert isinstance(df.dqc, QualityChecker)


def test_comparator_to_fn():
    f = QualityChecker._comparator_to_fn
    assert f(">", 3)(5) is True
    assert f(">", 3)(2) is False
    assert f("==", 3)(3) is True
    assert f("==", 3)(4) is False
    assert f("<", 3)(2) is True
    assert f("<", 3)(3) is False
    assert f("!!", 3)(1) is False


def test_treat_column_name(qc):
    assert qc._treat_column_name("ABC") == "ABC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("ABC") == "abc"


def test_row_count(qc):
    assert qc._apply_check(check("row_count", ">", "3")) == ("row_count", True)
    assert qc._apply_check(check("row_count", "==", "4")) == ("row_count", True)
    assert qc._apply_check(check("row_count", "<", "4")) == ("row_count", False)


def test_is_unique(df):
    df["dup"] = [1, 1, 2, 3]
    qc = QualityChecker(df)
    assert qc._apply_check(check("is_unique", "id")) == ("is_unique", True)
    assert qc._apply_check(check("is_unique", "dup")) == ("is_unique", False)


def test_not_null(df):
    df["nulls"] = [1, None, 3, 4]
    qc = QualityChecker(df)
    assert qc._apply_check(check("not_null", "id")) == ("not_null", True)
    assert qc._apply_check(check("not_null", "nulls")) == ("not_null", False)


def test_is_positive(df):
    df["mixed"] = [1, -1, 2, 3]
    qc = QualityChecker(df)
    assert bool(qc._apply_check(check("is_positive", "id"))[1]) is True
    assert bool(qc._apply_check(check("is_positive", "mixed"))[1]) is False


def test_missing_column_returns_key_error_message(qc):
    msg, ok = qc._apply_check(check("is_unique", "nope"))
    assert ok is False
    assert msg.startswith("Key error")
    assert "nope" in msg


def test_unknown_check_raises_genie_exception(qc):
    with pytest.raises(GenieException):
        qc._apply_check(check("bogus", "id"))


def test_has_columns(qc):
    ok_cols = Tree("columns", [])
    ok_cols.children = strings("id", "name").children
    name, result = qc._apply_check(check("has_columns", ok_cols))
    assert (name, result) == ("has_columns", True)

    missing = strings("id", "missing")
    assert qc._apply_check(check("has_columns", missing))[1] is False


def test_has_columns_ignore_case(qc):
    cols = strings("ID", "NAME")
    assert qc._apply_check(check("has_columns", cols))[1] is False
    assert qc._apply_check(check("has_columns", cols, "True"))[1] is True
    assert qc._apply_check(check("has_columns", cols, "False"))[1] is False


def test_has_one_of(df):
    df["color"] = ["Red", "Blue", "Red", "Blue"]
    qc = QualityChecker(df)
    assert qc._apply_check(check("has_one_of", "color", strings("Red", "Blue", "Green"))) == ("has_one_of", True)
    assert qc._apply_check(check("has_one_of", "color", strings("Red")))[1] is False


def test_has_one_of_ignore_case(df):
    df["color"] = ["Red", "Blue", "Red", "Blue"]
    qc = QualityChecker(df)
    allowed = strings("red", "blue")
    assert qc._apply_check(check("has_one_of", "color", allowed))[1] is False
    assert qc._apply_check(check("has_one_of", "color", allowed, "True"))[1] is True


def test_quantile(qc):
    # median of 1..4 is 2.5
    assert qc._apply_check(check("quantile", "score", "0.5", ">", "2"))[1] is True
    assert qc._apply_check(check("quantile", "score", "0.5", "<", "2"))[1] is False
    assert qc._apply_check(check("quantile", "score", "0.5", "==", "2.5"))[1] is True


def test_is_date(df):
    df["when"] = ["2020-01-01", "2020-02-01", "2021-03-04", "2022-12-31"]
    df["bad"] = ["2020-01-01", "not a date", "x", "y"]
    qc = QualityChecker(df)
    assert qc._apply_check(check("is_date", "when")) == ("is_date", True)
    msg, ok = qc._apply_check(check("is_date", "bad"))
    assert ok is False
    assert msg.startswith("Date parse error")


def test_value_length(df):
    df["mixed"] = ["a", "bb", "ccc", "dddd"]
    qc = QualityChecker(df)
    assert qc._apply_check(check("value_length", "name", "==", "3")) == ("value_length", True)
    assert qc._apply_check(check("value_length", "name", "<", "3"))[1] is False
    # mixed lengths can never satisfy the check
    assert qc._apply_check(check("value_length", "mixed", ">", "0"))[1] is False


def test_value_length_ignore_nulls(df):
    df["opt"] = ["abc", "", "def", "  "]
    qc = QualityChecker(df)
    assert qc._apply_check(check("value_length", "opt", "False", "==", "3"))[1] is False
    assert qc._apply_check(check("value_length", "opt", "True", "==", "3"))[1] is True


def test_percent_value_length(df):
    df["code"] = ["abc", "abc", "abc", "abcd"]
    qc = QualityChecker(df)
    # 75% of the rows have length 3
    assert qc._apply_check(check("percent_value_length", "code", "75", "==", "3")) == ("percent_value_length", True)
    assert qc._apply_check(check("percent_value_length", "code", "80", "==", "3"))[1] is False


def test_percent_value_length_with_ignore_nulls(df):
    df["code"] = ["abc", "abc", "", "abc"]
    qc = QualityChecker(df)
    assert qc._apply_check(check("percent_value_length", "code", "75", "False", "==", "3"))[1] is True
    assert qc._apply_check(check("percent_value_length", "code", "75", "True", "==", "3"))[1] is True
    assert qc._apply_check(check("percent_value_length", "code", "100", "False", "==", "3"))[1] is False


def test_run_tree_returns_results_in_reverse_order(qc):
    code_block = Tree("code_block", [
        Tree("check", [Tree("row_count", ["==", "4"])]),
        Tree("check", [Tree("is_unique", ["id"])]),
        Tree("check", [Tree("not_null", ["nope"])]),
    ])
    tree = Tree("start", [Tree("predicates", [code_block])])
    results = qc._run(tree)
    assert [r[0] for r in results][1:] == ["is_unique", "row_count"]
    assert results[0][0].startswith("Key error")
    assert [r[1] for r in results] == [False, True, True]


def test_run_rejects_bad_trees(qc):
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("other", [])]))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("predicates", []), Tree("predicates", [])]))


def test_validate_spec_rejects_garbage():
    msg, ok = QualityChecker.validate_spec("@@@ this is !!! not a valid spec ###")
    assert ok is False
    assert isinstance(msg, str) and msg
