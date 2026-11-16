import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["abc", "abd", "xyz", "pqr", "lmn"],
        "color": ["Red", "green", "RED", "Green", "red"],
    })


def check(node):
    """Wrap a check node the way the parser structures it."""
    return [node]


def str_list(*values):
    return Tree("values", [Token("STRING", '"%s"' % v) for v in values])


def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_accessor_registered_on_dataframe():
    df = make_df()
    assert isinstance(df.dqc, QualityChecker)


def test_comparator_to_fn_all_operators():
    assert QualityChecker._comparator_to_fn(">", 3)(5) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("==", 3)(3) is True
    assert QualityChecker._comparator_to_fn("==", 3)(4) is False
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("<", 3)(3) is False


def test_comparator_to_fn_unknown_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 3)
    assert fn(1) is False
    assert fn(3) is False


def test_treat_column_name_respects_ignore_case_flag():
    qc = QualityChecker(make_df())
    assert qc._treat_column_name("MyCol") == "MyCol"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MyCol") == "mycol"


def test_row_count_check():
    qc = QualityChecker(make_df())
    assert qc._apply_check(check(Tree("row_count", [">", "4"]))) == ("row_count", True)
    assert qc._apply_check(check(Tree("row_count", ["==", "5"]))) == ("row_count", True)
    assert qc._apply_check(check(Tree("row_count", ["<", "5"]))) == ("row_count", False)


def test_is_unique_check():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 2]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("is_unique", ["a"]))) == ("is_unique", True)
    assert qc._apply_check(check(Tree("is_unique", ["b"]))) == ("is_unique", False)


def test_not_null_check():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 2]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("not_null", ["a"]))) == ("not_null", True)
    assert qc._apply_check(check(Tree("not_null", ["b"]))) == ("not_null", False)


def test_is_positive_check():
    df = pd.DataFrame({"pos": [1, 2, 3], "neg": [1, 0, -2]})
    qc = QualityChecker(df)
    name, ok = qc._apply_check(check(Tree("is_positive", ["pos"])))
    assert name == "is_positive" and bool(ok) is True
    name, ok = qc._apply_check(check(Tree("is_positive", ["neg"])))
    assert name == "is_positive" and bool(ok) is False


def test_missing_column_returns_key_error_message():
    qc = QualityChecker(make_df())
    msg, ok = qc._apply_check(check(Tree("is_unique", ["nope"])))
    assert ok is False
    assert msg.startswith("Key error")
    assert "nope" in msg


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._apply_check(check(Tree("bogus_check", [])))


def test_has_one_of_exact_match():
    qc = QualityChecker(make_df())
    node_ok = Tree("has_one_of", ["color", str_list("Red", "green", "RED", "Green", "red")])
    node_bad = Tree("has_one_of", ["color", str_list("Red", "green")])
    assert qc._apply_check(check(node_ok)) == ("has_one_of", True)
    assert qc._apply_check(check(node_bad)) == ("has_one_of", False)


def test_has_one_of_ignore_case():
    qc = QualityChecker(make_df())
    node = Tree("has_one_of", ["color", str_list("red", "green"), "True"])
    assert qc._apply_check(check(node)) == ("has_one_of", True)
    node_false_flag = Tree("has_one_of", ["color", str_list("red", "green"), "False"])
    assert qc._apply_check(check(node_false_flag)) == ("has_one_of", False)


def test_quantile_check():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("quantile", ["v", "0.5", "==", "3"]))) == ("quantile", True)
    assert qc._apply_check(check(Tree("quantile", ["v", "0.5", ">", "3"]))) == ("quantile", False)
    assert qc._apply_check(check(Tree("quantile", ["v", "1.0", "<", "10"]))) == ("quantile", True)


def test_is_date_check():
    df = pd.DataFrame({"good": ["2020-01-01", "2021-05-06"], "bad": ["not a date", "also not"]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("is_date", ["good"]))) == ("is_date", True)
    msg, ok = qc._apply_check(check(Tree("is_date", ["bad"])))
    assert ok is False
    assert msg.startswith("Date parse error")


def test_value_length_without_ignore_nulls():
    df = pd.DataFrame({"code": ["abc", "def", "ghi"], "mixed": ["abc", "de", "ghi"]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("value_length", ["code", "==", "3"]))) == ("value_length", True)
    assert qc._apply_check(check(Tree("value_length", ["code", ">", "3"]))) == ("value_length", False)
    # varying lengths -> never passes
    assert qc._apply_check(check(Tree("value_length", ["mixed", ">", "0"]))) == ("value_length", False)


def test_value_length_ignore_nulls():
    df = pd.DataFrame({"code": ["abc", "", "def", "   "]})
    qc = QualityChecker(df)
    node_ignore = Tree("value_length", ["code", "True", "==", "3"])
    node_keep = Tree("value_length", ["code", "False", "==", "3"])
    assert qc._apply_check(check(node_ignore)) == ("value_length", True)
    assert qc._apply_check(check(node_keep)) == ("value_length", False)


def test_percent_value_length():
    df = pd.DataFrame({"c": ["abc", "abd", "x", "yy"]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("percent_value_length", ["c", "50", "==", "3"]))) == (
        "percent_value_length", True)
    df2 = pd.DataFrame({"c": ["abc", "abd", "x", "yy"]})
    qc2 = QualityChecker(df2)
    assert qc2._apply_check(check(Tree("percent_value_length", ["c", "100", "==", "3"]))) == (
        "percent_value_length", False)


def test_has_columns_case_sensitivity():
    df = pd.DataFrame({"Alpha": [1], "Beta": [2]})
    qc = QualityChecker(df)
    exact = Tree("has_columns", [str_list("Alpha", "Beta")])
    wrong_case = Tree("has_columns", [str_list("alpha", "beta")])
    wrong_case_ignored = Tree("has_columns", [str_list("alpha", "beta"), "True"])
    missing = Tree("has_columns", [str_list("Alpha", "Gamma")])
    assert qc._apply_check(check(exact)) == ("has_columns", True)
    assert qc._apply_check(check(wrong_case)) == ("has_columns", False)
    assert qc._apply_check(check(wrong_case_ignored)) == ("has_columns", True)
    assert qc._apply_check(check(missing)) == ("has_columns", False)


def _program(*check_nodes):
    block = Tree("code_block", [Tree("check", [n]) for n in check_nodes])
    return Tree("start", [Tree("predicates", [block])])


def test_run_returns_results_in_reverse_order(monkeypatch):
    tree = _program(
        Tree("row_count", ["==", "5"]),
        Tree("is_unique", ["id"]),
        Tree("not_null", ["missing_col"]),
    )
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(make_df())
    results = qc.run("ignored spec")
    assert len(results) == 3
    assert results[0][1] is False and results[0][0].startswith("Key error")
    assert results[1] == ("is_unique", True)
    assert results[2] == ("row_count", True)


def test_run_rejects_tree_with_multiple_children():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("predicates", []), Tree("predicates", [])]))


def test_run_rejects_unknown_instruction():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("something_else", [])]))


def test_validate_spec_rejects_garbage():
    msg, ok = QualityChecker.validate_spec("@@@ ### this is definitely not a valid spec $$$")
    assert ok is False
    assert isinstance(msg, str) and msg
