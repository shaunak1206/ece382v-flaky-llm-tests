import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(name, *children):
    """Build a 'check' tree wrapping a single DSL check."""
    return Tree("check", [Tree(name, list(children))])


def _program(*checks):
    """Build a full parse tree as _run() expects it."""
    return Tree("start", [Tree("predicates", [Tree("code_block", list(checks))])])


def _strings(*values):
    return Tree("strings", [Token("STRING", '"%s"' % v) for v in values])


@pytest.fixture
def df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["ab", "cd", "ef", "gh"],
        "color": ["red", "green", "red", "blue"],
    })


def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_comparator_to_fn_all_branches():
    assert QualityChecker._comparator_to_fn(">", 3)(5) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("==", 3)(3) is True
    assert QualityChecker._comparator_to_fn("==", 3)(4) is False
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("<", 3)(3) is False


def test_comparator_unknown_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 3)
    assert fn(1) is False
    assert fn(3) is False


def test_treat_column_name_respects_ignore_case_flag(df):
    qc = QualityChecker(df)
    assert qc._treat_column_name("MyCol") == "MyCol"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MyCol") == "mycol"


def test_row_count_checks(df):
    qc = QualityChecker(df)
    tree = _program(
        _check("row_count", ">", "3"),
        _check("row_count", "==", "5"),
    )
    results = qc._run(tree)
    # results are inserted at the front, so order is reversed
    assert results == [("row_count", False), ("row_count", True)]


def test_results_are_in_reverse_order_of_checks(df):
    qc = QualityChecker(df)
    tree = _program(
        _check("row_count", "==", "4"),
        _check("is_unique", "id"),
        _check("not_null", "name"),
    )
    results = qc._run(tree)
    assert [r[0] for r in results] == ["not_null", "is_unique", "row_count"]
    assert all(bool(r[1]) for r in results)


def test_is_unique_true_and_false(df):
    qc = QualityChecker(df)
    name, ok = qc._apply_check([Tree("is_unique", ["id"])])
    assert name == "is_unique" and bool(ok)
    name, ok = qc._apply_check([Tree("is_unique", ["color"])])
    assert name == "is_unique" and not bool(ok)


def test_not_null_detects_nulls():
    frame = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    qc = QualityChecker(frame)
    assert not bool(qc._apply_check([Tree("not_null", ["a"])])[1])
    assert bool(qc._apply_check([Tree("not_null", ["b"])])[1])


def test_is_positive():
    frame = pd.DataFrame({"pos": [1, 2, 3], "mixed": [1, 0, 3]})
    qc = QualityChecker(frame)
    assert bool(qc._apply_check([Tree("is_positive", ["pos"])])[1])
    assert not bool(qc._apply_check([Tree("is_positive", ["mixed"])])[1])


def test_missing_column_returns_key_error_message(df):
    qc = QualityChecker(df)
    msg, ok = qc._apply_check([Tree("is_unique", ["nope"])])
    assert msg == "Key error: is_unique for nope"
    assert ok is False


def test_unknown_check_raises_genie_exception(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._apply_check([Tree("totally_unknown", ["id"])])


def test_has_columns_case_sensitive_and_insensitive(df):
    qc = QualityChecker(df)
    ok_node = Tree("has_columns", [_strings("id", "name")])
    assert qc._apply_has_columns(ok_node) == ("has_columns", True)

    missing_node = Tree("has_columns", [_strings("id", "absent")])
    assert qc._apply_has_columns(missing_node) == ("has_columns", False)

    upper_node = Tree("has_columns", [_strings("ID", "NAME")])
    assert qc._apply_has_columns(upper_node)[1] is False

    upper_ci_node = Tree("has_columns", [_strings("ID", "NAME"), Token("BOOL", "True")])
    assert qc._apply_has_columns(upper_ci_node)[1] is True


def test_has_one_of_values(df):
    qc = QualityChecker(df)
    allowed = Tree("has_one_of", ["color", _strings("red", "green", "blue", "pink")])
    assert qc._apply_has_one_of(allowed) == ("has_one_of", True)

    too_few = Tree("has_one_of", ["color", _strings("red", "green")])
    assert qc._apply_has_one_of(too_few) == ("has_one_of", False)


def test_has_one_of_ignore_case():
    frame = pd.DataFrame({"c": ["Red", "GREEN", "red"]})
    qc = QualityChecker(frame)
    sensitive = Tree("has_one_of", ["c", _strings("red", "green")])
    assert qc._apply_has_one_of(sensitive)[1] is False

    insensitive = Tree("has_one_of", ["c", _strings("red", "green"), Token("BOOL", "True")])
    assert qc._apply_has_one_of(insensitive)[1] is True


def test_quantile_comparisons():
    frame = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    qc = QualityChecker(frame)
    # median is 3.0
    assert qc._apply_quantile(Tree("quantile", ["v", "0.5", "==", "3"])) == ("quantile", True)
    assert qc._apply_quantile(Tree("quantile", ["v", "0.5", ">", "2"]))[1] is True
    assert qc._apply_quantile(Tree("quantile", ["v", "0.5", "<", "2"]))[1] is False


def test_is_date_valid_and_invalid():
    frame = pd.DataFrame({"good": ["2020-01-01", "2021-05-06"], "bad": ["not a date", "nope"]})
    qc = QualityChecker(frame)
    assert qc._apply_date_validation(Tree("is_date", ["good"])) == ("is_date", True)
    msg, ok = qc._apply_date_validation(Tree("is_date", ["bad"]))
    assert ok is False
    assert msg == "Date parse error: is_date for bad"


def test_value_length_without_ignore_nulls_argument(df):
    qc = QualityChecker(df)
    assert qc._apply_value_length(Tree("value_length", ["name", "==", "2"])) == ("value_length", True)
    assert qc._apply_value_length(Tree("value_length", ["name", ">", "2"]))[1] is False


def test_value_length_mixed_lengths_fails():
    frame = pd.DataFrame({"s": ["ab", "abc"]})
    qc = QualityChecker(frame)
    assert qc._apply_value_length(Tree("value_length", ["s", "==", "2"]))[1] is False


def test_value_length_ignore_nulls_skips_blank_values():
    frame = pd.DataFrame({"s": ["ab", "", "cd"]})
    qc = QualityChecker(frame)
    without = Tree("value_length", ["s", Token("BOOL", "False"), "==", "2"])
    assert qc._apply_value_length(without)[1] is False
    with_ignore = Tree("value_length", ["s", Token("BOOL", "True"), "==", "2"])
    assert qc._apply_value_length(with_ignore)[1] is True


def test_percent_value_length_thresholds():
    frame = pd.DataFrame({"s": ["ab", "cd", "e", "fg"]})
    qc = QualityChecker(frame)
    # children: column, percent, comparator, rhs  -> 3 of 4 (75%) have length 2
    assert qc._apply_percent_value_length(
        Tree("percent_value_length", ["s", "75", "==", "2"]))[1] is True
    frame2 = pd.DataFrame({"s": ["ab", "cd", "e", "fg"]})
    qc2 = QualityChecker(frame2)
    assert qc2._apply_percent_value_length(
        Tree("percent_value_length", ["s", "80", "==", "2"]))[1] is False


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    frame = pd.DataFrame({"ID": [1, 2, 3], "Name": ["a", "b", "c"]})
    qc = QualityChecker(frame)
    tree = _program(_check("is_unique", "id"), _check("not_null", "name"))
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))

    results = qc.run("ignored spec", ignore_column_case=True)

    assert list(frame.columns) == ["id", "name"]
    assert qc.ignore_column_case is True
    assert len(results) == 2
    assert all(bool(r[1]) for r in results)


def test_run_dispatches_unknown_instruction_and_bad_tree(df):
    qc = QualityChecker(df)
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("something_else", [])]))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("predicates", []), Tree("predicates", [])]))


def test_validate_spec_rejects_garbage():
    message, ok = QualityChecker.validate_spec("@@@ !!! ???")
    assert ok is False
    assert isinstance(message, str)
