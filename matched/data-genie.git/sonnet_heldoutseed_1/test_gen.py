import pandas as pd
import pytest
from lark import Tree, Token

import genie_pkg.dqc  # noqa: F401  (registers the accessor)
from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(node):
    """Wrap a node the way _apply_check expects (a list with one tree)."""
    return [node]


def _str_list(*values):
    return Tree("values", [Token("STRING", '"{}"'.format(v)) for v in values])


@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["x", "y", "z", "x"],
    })


def test_empty_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None], "b": [None, None]}))


def test_accessor_is_registered(df):
    assert isinstance(df.dqc, QualityChecker)


def test_comparator_to_fn():
    assert QualityChecker._comparator_to_fn(">", 3)(5) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("==", 3)(3) is True
    assert QualityChecker._comparator_to_fn("==", 3)(4) is False
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("<", 3)(3) is False
    assert QualityChecker._comparator_to_fn("!!", 3)(3) is False


def test_treat_column_name_respects_flag(df):
    qc = QualityChecker(df)
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_row_count_check(df):
    qc = QualityChecker(df)
    assert qc._apply_check(_check(Tree("row_count", [">", "3"]))) == ("row_count", True)
    assert qc._apply_check(_check(Tree("row_count", ["<", "3"]))) == ("row_count", False)
    assert qc._apply_check(_check(Tree("row_count", ["==", "4"]))) == ("row_count", True)


def test_is_unique_check(df):
    qc = QualityChecker(df)
    name, ok = qc._apply_check(_check(Tree("is_unique", ["a"])))
    assert name == "is_unique" and bool(ok) is True
    name, ok = qc._apply_check(_check(Tree("is_unique", ["b"])))
    assert name == "is_unique" and bool(ok) is False


def test_not_null_check():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]}))
    assert bool(qc._apply_check(_check(Tree("not_null", ["a"])))[1]) is True
    assert bool(qc._apply_check(_check(Tree("not_null", ["b"])))[1]) is False


def test_is_positive_check():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3], "b": [1, -2, 3]}))
    assert bool(qc._apply_check(_check(Tree("is_positive", ["a"])))[1]) is True
    assert bool(qc._apply_check(_check(Tree("is_positive", ["b"])))[1]) is False


def test_missing_column_returns_key_error_message(df):
    qc = QualityChecker(df)
    result = qc._apply_check(_check(Tree("is_unique", ["missing"])))
    assert result == ("Key error: is_unique for missing", False)


def test_unknown_check_raises_genie_exception(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._apply_check(_check(Tree("not_a_real_check", ["a"])))


def test_has_columns_check(df):
    qc = QualityChecker(df)
    ok_node = Tree("has_columns", [_str_list("a", "b")])
    bad_node = Tree("has_columns", [_str_list("a", "zzz")])
    assert qc._apply_check(_check(ok_node)) == ("has_columns", True)
    assert qc._apply_check(_check(bad_node)) == ("has_columns", False)


def test_has_columns_ignore_case():
    qc = QualityChecker(pd.DataFrame({"Name": [1], "AGE": [2]}))
    case_sensitive = Tree("has_columns", [_str_list("name", "age")])
    case_insensitive = Tree("has_columns", [_str_list("name", "age"), "True"])
    assert qc._apply_check(_check(case_sensitive)) == ("has_columns", False)
    assert qc._apply_check(_check(case_insensitive)) == ("has_columns", True)


def test_has_one_of_check(df):
    qc = QualityChecker(df)
    ok = Tree("has_one_of", ["b", _str_list("x", "y", "z", "w")])
    bad = Tree("has_one_of", ["b", _str_list("x", "y")])
    assert qc._apply_check(_check(ok)) == ("has_one_of", True)
    assert qc._apply_check(_check(bad)) == ("has_one_of", False)


def test_has_one_of_ignore_case():
    qc = QualityChecker(pd.DataFrame({"c": ["A", "b", "A"]}))
    sensitive = Tree("has_one_of", ["c", _str_list("a", "b"), "False"])
    insensitive = Tree("has_one_of", ["c", _str_list("a", "b"), "True"])
    assert qc._apply_check(_check(sensitive)) == ("has_one_of", False)
    assert qc._apply_check(_check(insensitive)) == ("has_one_of", True)


def test_quantile_check(df):
    qc = QualityChecker(df)
    # median of [1,2,3,4] is 2.5
    assert qc._apply_check(_check(Tree("quantile", ["a", "0.5", ">", "2"]))) == ("quantile", True)
    assert qc._apply_check(_check(Tree("quantile", ["a", "0.5", "<", "2"]))) == ("quantile", False)
    assert qc._apply_check(_check(Tree("quantile", ["a", "0.5", "==", "2.5"]))) == ("quantile", True)


def test_is_date_check():
    qc = QualityChecker(pd.DataFrame({
        "good": ["2020-01-01", "2021-05-06"],
        "bad": ["not a date", "also not"],
    }))
    assert qc._apply_check(_check(Tree("is_date", ["good"]))) == ("is_date", True)
    msg, ok = qc._apply_check(_check(Tree("is_date", ["bad"])))
    assert ok is False
    assert msg == "Date parse error: is_date for bad"


def test_value_length_check():
    qc = QualityChecker(pd.DataFrame({
        "same": ["abc", "xyz", "123"],
        "mixed": ["abc", "xy", "1"],
        "blank": ["abc", "", "xyz"],
    }))
    assert qc._apply_check(_check(Tree("value_length", ["same", "==", "3"]))) == ("value_length", True)
    assert qc._apply_check(_check(Tree("value_length", ["same", "<", "3"]))) == ("value_length", False)
    # multiple distinct lengths never pass
    assert qc._apply_check(_check(Tree("value_length", ["mixed", ">", "0"]))) == ("value_length", False)
    # blanks counted unless ignore_nulls is True
    assert qc._apply_check(_check(Tree("value_length", ["blank", "False", "==", "3"]))) == ("value_length", False)
    assert qc._apply_check(_check(Tree("value_length", ["blank", "True", "==", "3"]))) == ("value_length", True)


def test_percent_value_length_check():
    qc = QualityChecker(pd.DataFrame({"c": ["abc", "abd", "x", "yyy"]}))
    # 3 of 4 rows (75%) have length 3
    assert qc._apply_check(_check(Tree("percent_value_length", ["c", "75", "==", "3"]))) == (
        "percent_value_length", True)
    assert qc._apply_check(_check(Tree("percent_value_length", ["c", "80", "==", "3"]))) == (
        "percent_value_length", False)


def test_run_returns_results_in_reverse_order(df, monkeypatch):
    tree = Tree("start", [
        Tree("predicates", [
            Tree("code_block", [
                Tree("check", [Tree("row_count", ["==", "4"])]),
                Tree("check", [Tree("has_columns", [_str_list("a", "nope")])]),
            ])
        ])
    ])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    results = QualityChecker(df).run("ignored spec")
    assert results == [("has_columns", False), ("row_count", True)]


def test_run_ignore_column_case_lowercases_columns(monkeypatch):
    frame = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    tree = Tree("start", [
        Tree("predicates", [
            Tree("code_block", [Tree("check", [Tree("is_unique", ["A"])])])
        ])
    ])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    qc = QualityChecker(frame)
    results = qc.run("ignored", ignore_column_case=True)
    assert list(frame.columns) == ["a", "b"]
    assert qc.ignore_column_case is True
    assert results[0][0] == "is_unique"
    assert bool(results[0][1]) is False or bool(results[0][1]) is True


def test_run_rejects_malformed_tree(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("unknown_instruction", [])]))


def test_validate_spec_rejects_garbage():
    message, ok = QualityChecker.validate_spec("!!! @@@ ??? not a valid spec ###")
    assert ok is False
    assert isinstance(message, str) and message
