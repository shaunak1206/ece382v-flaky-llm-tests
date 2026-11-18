import pandas as pd
import pytest
from lark import Tree, Token

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_qc(df):
    return df.dqc


def token(value):
    return Token("STRING", value)


def wrap_check(node):
    return Tree("check", [node])


def make_predicates(*check_nodes):
    return Tree(
        "predicates",
        [Tree("code_block", [wrap_check(node) for node in check_nodes])],
    )


def make_root(*check_nodes):
    return Tree("root", [make_predicates(*check_nodes)])


def test_validate_raises_on_all_empty_dataframe():
    df = pd.DataFrame({"a": [None], "b": [None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        _ = df.dqc


def test_comparator_to_fn_greater_equal_less_and_default():
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
    qc = make_qc(pd.DataFrame({"A": [1]}))
    assert qc._treat_column_name("A") == "A"
    qc.ignore_column_case = True
    assert qc._treat_column_name("A") == "a"


def test_apply_check_row_count():
    qc = make_qc(pd.DataFrame({"a": [1, 2, 3]}))
    node = Tree("row_count", [">", "2"])
    assert qc._apply_check([node]) == ("row_count", True)


def test_apply_check_is_unique_true_and_false():
    qc1 = make_qc(pd.DataFrame({"a": [1, 2, 3]}))
    assert qc1._apply_check([Tree("is_unique", ["a"])]) == ("is_unique", True)

    qc2 = make_qc(pd.DataFrame({"a": [1, 1, 2]}))
    assert qc2._apply_check([Tree("is_unique", ["a"])]) == ("is_unique", False)


def test_apply_check_not_null_and_is_positive():
    qc = make_qc(pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]}))
    assert qc._apply_check([Tree("not_null", ["a"])]) == ("not_null", True)
    assert qc._apply_check([Tree("not_null", ["b"])]) == ("not_null", False)
    assert qc._apply_check([Tree("is_positive", ["a"])]) == ("is_positive", True)


def test_apply_has_one_of_default_case():
    qc = make_qc(pd.DataFrame({"a": ["x", "y", "x"]}))
    node = Tree("has_one_of", ["a", Tree("allowed", [token('"x"'), token('"y"')])])
    assert qc._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_has_one_of_ignore_case():
    qc = make_qc(pd.DataFrame({"a": ["x", "Y"]}))
    node = Tree(
        "has_one_of",
        ["a", Tree("allowed", [token('"x"'), token('"y"')]), "True"],
    )
    assert qc._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_quantile_true_and_false():
    qc = make_qc(pd.DataFrame({"a": [1, 2, 3, 4]}))
    node_true = Tree("quantile", ["a", "0.5", ">", "2"])
    node_false = Tree("quantile", ["a", "0.5", "<", "2"])
    assert qc._apply_quantile(node_true) == ("quantile", True)
    assert qc._apply_quantile(node_false) == ("quantile", False)


def test_apply_date_validation_success_and_failure():
    qc = make_qc(pd.DataFrame({"a": ["2020-01-01", "2021-02-02"]}))
    ok = Tree("is_date", ["a"])
    bad_qc = make_qc(pd.DataFrame({"a": ["2020-01-01", "not-a-date"]}))
    bad = Tree("is_date", ["a"])

    assert qc._apply_date_validation(ok) == ("is_date", True)
    result = bad_qc._apply_date_validation(bad)
    assert result[0].startswith("Date parse error: is_date for a")
    assert result[1] is False


def test_apply_value_length_without_ignore_nulls():
    qc = make_qc(pd.DataFrame({"a": ["aa", "bb"]}))
    node = Tree("value_length", ["a", "==", "2"])
    assert qc._apply_value_length(node) == ("value_length", True)


def test_apply_value_length_with_ignore_nulls():
    qc = make_qc(pd.DataFrame({"a": ["aa", "", None]}))
    node = Tree("value_length", ["a", "True", "==", "2"])
    assert qc._apply_value_length(node) == ("value_length", True)


def test_apply_percent_value_length_without_ignore_nulls():
    qc = make_qc(pd.DataFrame({"a": ["aa", "bb", "c"]}))
    node = Tree("percent_value_length", ["a", "66", "==", "2"])
    assert qc._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_has_columns_case_sensitive_and_insensitive():
    qc = make_qc(pd.DataFrame({"A": [1], "b": [2]}))
    node_sensitive = Tree(
        "has_columns",
        [Tree("cols", [token('"A"'), token('"b"')])],
    )
    node_insensitive = Tree(
        "has_columns",
        [Tree("cols", [token('"a"'), token('"B"')]), "True"],
    )
    assert qc._apply_has_columns(node_sensitive) == ("has_columns", True)
    assert qc._apply_has_columns(node_insensitive) == ("has_columns", True)


def test_apply_check_unknown_raises_genie_exception():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException, match="not implemented"):
        qc._apply_check([Tree("made_up_check", ["a"])])


def test_apply_check_keyerror_returns_failure_tuple():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    result = qc._apply_check([Tree("is_unique", ["missing"])])
    assert result == ("Key error: is_unique for missing", False)


def test_apply_predicates_reverses_result_order():
    qc = make_qc(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    predicates = [
        Tree(
            "code_block",
            [
                wrap_check(Tree("row_count", [">", "1"])),
                wrap_check(Tree("is_unique", ["a"])),
            ],
        )
    ]
    result = qc._apply_predicates(predicates)
    assert result == [("is_unique", True), ("row_count", True)]


def test_run_uses_parse_spec_and_optionally_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    qc = make_qc(df)

    ast = make_root(Tree("row_count", [">", "1"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda _: ast))

    result = qc.run("ignored", ignore_column_case=True)
    assert result == [("row_count", True)]
    assert list(qc._obj.columns) == ["a", "b"]


def test_validate_spec_success_and_failure(monkeypatch):
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda _: Tree("ok", [])))
    assert QualityChecker.validate_spec("ignored") == (None, True)

    def boom(_):
        raise ValueError("bad spec")

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(boom))
    msg, ok = QualityChecker.validate_spec("ignored")
    assert ok is False
    assert "bad spec" in msg
