import pandas as pd
import pytest
from lark import Token, Tree

import genie_pkg.dqc  # noqa: F401  (registers the accessor)
from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def apply(df, node):
    return QualityChecker(df)._apply_check([node])


def str_list(values):
    return Tree("values", [Token("STRING", '"%s"' % v) for v in values])


def build_program(*nodes):
    checks = [Tree("check", [n]) for n in nodes]
    return Tree("start", [Tree("predicates", [Tree("code_block", checks)])])


def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_accessor_is_registered_on_dataframe():
    df = pd.DataFrame({"a": [1, 2]})
    assert isinstance(df.dqc, QualityChecker)


def test_comparator_to_fn_operators():
    assert QualityChecker._comparator_to_fn(">", 1)(2) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("==", 2)(2) is True
    assert QualityChecker._comparator_to_fn("==", 3)(2) is False
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("<", 1)(2) is False


def test_comparator_to_fn_unknown_comparator_is_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 1)
    assert fn(5) is False
    assert fn(1) is False


def test_treat_column_name_respects_ignore_case_flag():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_row_count_check():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert apply(df, Tree("row_count", [">", "2"])) == ("row_count", True)
    assert apply(df, Tree("row_count", ["==", "3"])) == ("row_count", True)
    assert apply(df, Tree("row_count", ["<", "3"])) == ("row_count", False)


def test_is_unique_check():
    df = pd.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]})
    assert apply(df, Tree("is_unique", ["u"])) == ("is_unique", True)
    assert apply(df, Tree("is_unique", ["d"])) == ("is_unique", False)


def test_not_null_check():
    df = pd.DataFrame({"full": [1, 2, 3], "holes": [1, None, 3]})
    name, ok = apply(df, Tree("not_null", ["full"]))
    assert name == "not_null" and bool(ok) is True
    name, ok = apply(df, Tree("not_null", ["holes"]))
    assert name == "not_null" and bool(ok) is False


def test_is_positive_check():
    df = pd.DataFrame({"pos": [1, 2, 3], "mixed": [1, 0, 3]})
    name, ok = apply(df, Tree("is_positive", ["pos"]))
    assert name == "is_positive" and bool(ok) is True
    name, ok = apply(df, Tree("is_positive", ["mixed"]))
    assert bool(ok) is False


def test_missing_column_returns_key_error_message():
    df = pd.DataFrame({"a": [1, 2]})
    msg, ok = apply(df, Tree("is_unique", ["missing"]))
    assert ok is False
    assert msg.startswith("Key error")
    assert "is_unique" in msg and "missing" in msg


def test_unimplemented_check_raises_genie_exception():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(GenieException):
        apply(df, Tree("bogus_check", []))


def test_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({"Name": [1], "Age": [2]})
    present = Tree("has_columns", [str_list(["Name", "Age"])])
    assert apply(df, present) == ("has_columns", True)

    wrong_case = Tree("has_columns", [str_list(["name"])])
    assert apply(df, wrong_case) == ("has_columns", False)

    ignoring_case = Tree("has_columns", [str_list(["name", "AGE"]), "True"])
    assert apply(df, ignoring_case) == ("has_columns", True)

    missing = Tree("has_columns", [str_list(["Name", "Other"])])
    assert apply(df, missing) == ("has_columns", False)


def test_has_one_of_check():
    df = pd.DataFrame({"color": ["red", "green", "red"]})
    ok_node = Tree("has_one_of", ["color", str_list(["red", "green", "blue"])])
    assert apply(df, ok_node) == ("has_one_of", True)

    bad_node = Tree("has_one_of", ["color", str_list(["red"])])
    assert apply(df, bad_node) == ("has_one_of", False)


def test_has_one_of_ignore_case():
    df = pd.DataFrame({"color": ["Red", "GREEN"]})
    strict = Tree("has_one_of", ["color", str_list(["red", "green"]), "False"])
    assert apply(df, strict) == ("has_one_of", False)
    relaxed = Tree("has_one_of", ["color", str_list(["red", "green"]), "True"])
    assert apply(df, relaxed) == ("has_one_of", True)


def test_quantile_check():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert apply(df, Tree("quantile", ["a", "0.5", ">", "2"])) == ("quantile", True)
    assert apply(df, Tree("quantile", ["a", "0.5", "<", "2"])) == ("quantile", False)
    assert apply(df, Tree("quantile", ["a", "0.5", "==", "3"])) == ("quantile", True)


def test_is_date_check():
    good = pd.DataFrame({"d": ["2020-01-01", "2021-05-06"]})
    assert apply(good, Tree("is_date", ["d"])) == ("is_date", True)

    bad = pd.DataFrame({"d": ["definitely not a date", "nope"]})
    msg, ok = apply(bad, Tree("is_date", ["d"]))
    assert ok is False
    assert msg.startswith("Date parse error")


def test_value_length_check():
    df = pd.DataFrame({"code": ["abc", "xyz", "123"]})
    assert apply(df, Tree("value_length", ["code", "==", "3"])) == ("value_length", True)
    assert apply(df, Tree("value_length", ["code", ">", "3"])) == ("value_length", False)

    mixed = pd.DataFrame({"code": ["abc", "abcd"]})
    assert apply(mixed, Tree("value_length", ["code", "==", "3"])) == ("value_length", False)


def test_value_length_ignore_nulls():
    df = pd.DataFrame({"code": ["abc", "xyz", ""]})
    # Without ignoring nulls the empty string has a different length
    assert apply(df, Tree("value_length", ["code", "False", "==", "3"])) == ("value_length", False)
    # Ignoring blanks leaves only length-3 values
    assert apply(df, Tree("value_length", ["code", "True", "==", "3"])) == ("value_length", True)


def test_percent_value_length_check():
    df = pd.DataFrame({"w": ["abc", "abcd", "xyz", "ab"]})
    assert apply(df, Tree("percent_value_length", ["w", "50", "%", "3"])) == ("percent_value_length", True)
    df2 = pd.DataFrame({"w": ["abc", "abcd", "xyz", "ab"]})
    assert apply(df2, Tree("percent_value_length", ["w", "75", "%", "3"])) == ("percent_value_length", False)


def test_run_returns_results_in_reverse_order_of_checks(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3]})
    tree = build_program(Tree("row_count", ["==", "3"]), Tree("is_unique", ["a"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))

    results = QualityChecker(df).run("ignored spec")
    assert [name for name, _ in results] == ["is_unique", "row_count"]
    assert all(bool(ok) for _, ok in results)


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"ABC": [1, 2, 3]})
    tree = build_program(Tree("is_unique", ["ABC"]))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))

    results = QualityChecker(df).run("ignored spec", ignore_column_case=True)
    assert results == [("is_unique", True)]
    assert list(df.columns) == ["abc"]


def test_run_rejects_malformed_trees():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("something_else", [])]))


def test_validate_spec_rejects_garbage():
    result = QualityChecker.validate_spec("@@@ this is definitely !!! not a valid spec ###")
    assert result[1] is False
    assert isinstance(result[0], str)
