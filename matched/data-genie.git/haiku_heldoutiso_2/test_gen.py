import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_qc(data):
    return QualityChecker(pd.DataFrame(data))


def node(name, *children):
    return Tree(name, list(children))


def apply(qc, tree_node):
    # _apply_check receives the list of child trees of a check wrapper
    return qc._apply_check([tree_node])


def test_init_rejects_empty_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_init_rejects_all_nan_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_init_stores_frame_and_defaults_ignore_case_to_false():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)
    assert qc._obj is df
    assert qc.ignore_column_case is False


def test_dataframe_accessor_is_registered():
    df = pd.DataFrame({"a": [1]})
    assert isinstance(df.dqc, QualityChecker)


@pytest.mark.parametrize(
    "comparator, lhs, expected",
    [
        (">", 3, True),
        (">", 1, False),
        ("==", 2, True),
        ("==", 5, False),
        ("<", 1, True),
        ("<", 3, False),
    ],
)
def test_comparator_to_fn_supported_operators(comparator, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, 2)
    assert bool(fn(lhs)) is expected


def test_comparator_to_fn_unknown_operator_is_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 2)
    assert fn(2) is False
    assert fn(100) is False


def test_treat_column_name_respects_ignore_flag():
    qc = make_qc({"a": [1]})
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


@pytest.mark.parametrize("operator, rhs, expected", [(">", "2", True), (">", "3", False)])
def test_row_count(operator, rhs, expected):
    qc = make_qc({"a": [1, 2, 3]})
    name, result = apply(qc, node("row_count", operator, rhs))
    assert name == "row_count"
    assert bool(result) is expected


def test_is_unique_detects_duplicates():
    assert bool(apply(make_qc({"a": [1, 2, 3]}), node("is_unique", "a"))[1]) is True
    assert bool(apply(make_qc({"a": [1, 1, 3]}), node("is_unique", "a"))[1]) is False


def test_not_null_detects_missing_values():
    ok = apply(make_qc({"a": [1, 2]}), node("not_null", "a"))
    bad = apply(make_qc({"a": [1, None]}), node("not_null", "a"))
    assert ok == ("not_null", True) or (ok[0] == "not_null" and bool(ok[1]) is True)
    assert bool(bad[1]) is False


def test_is_positive_requires_all_values_positive():
    good = apply(make_qc({"a": [1, 2]}), node("is_positive", "a"))
    bad = apply(make_qc({"a": [1, -2]}), node("is_positive", "a"))
    assert bool(good[1]) is True
    assert bool(bad[1]) is False


def test_has_one_of_case_sensitive_rejects_unlisted_value():
    values = Tree("values", [Token("STRING", '"a"')])
    qc = make_qc({"c": ["a", "b"]})
    name, result = apply(qc, Tree("has_one_of", ["c", values]))
    assert name == "has_one_of"
    assert bool(result) is False


def test_has_one_of_ignore_case_accepts_matching_values_in_any_case():
    values = Tree("values", [Token("STRING", '"red"'), Token("STRING", '"BLUE"')])
    qc = make_qc({"Color": ["RED", "blue"]})
    _, result = apply(qc, Tree("has_one_of", ["Color", values, "True"]))
    assert bool(result) is True


def test_has_columns_passes_when_all_present():
    cols = Tree("columns", [Token("STRING", '"a"'), Token("STRING", '"b"')])
    qc = make_qc({"a": [1], "b": [2], "c": [3]})
    name, result = apply(qc, Tree("has_columns", [cols]))
    assert name == "has_columns"
    assert bool(result) is True


def test_has_columns_fails_when_column_missing():
    cols = Tree("columns", [Token("STRING", '"a"'), Token("STRING", '"z"')])
    qc = make_qc({"a": [1], "b": [2]})
    _, result = apply(qc, Tree("has_columns", [cols]))
    assert bool(result) is False


def test_has_columns_ignore_case_matches_mixed_case_names():
    cols = Tree("columns", [Token("STRING", '"A"')])
    qc = make_qc({"a": [1]})
    _, result = apply(qc, Tree("has_columns", [cols, "True"]))
    assert bool(result) is True


def test_quantile_compares_median():
    qc = make_qc({"v": [1, 2, 3, 4, 5]})
    name, above = apply(qc, node("quantile", "v", "0.5", ">", "2"))
    _, below = apply(qc, node("quantile", "v", "0.5", "<", "2"))
    assert name == "quantile"
    assert bool(above) is True
    assert bool(below) is False


def test_is_date_accepts_valid_dates():
    qc = make_qc({"d": ["2020-01-01", "2021-06-30"]})
    assert apply(qc, node("is_date", "d")) == ("is_date", True)


def test_is_date_reports_parse_error_for_invalid_dates():
    qc = make_qc({"d": ["not a date", "zzz"]})
    message, result = apply(qc, node("is_date", "d"))
    assert message.startswith("Date parse error: is_date for d")
    assert result is False


def test_value_length_exact_match_three_children():
    assert bool(apply(make_qc({"v": ["abc", "def"]}), node("value_length", "v", "==", "3"))[1]) is True
    assert bool(apply(make_qc({"v": ["abc", "de"]}), node("value_length", "v", "==", "3"))[1]) is False


def test_value_length_with_explicit_ignore_nulls_false():
    qc = make_qc({"v": ["abc", "xyz"]})
    _, result = apply(qc, node("value_length", "v", "False", ">", "2"))
    assert bool(result) is True


def test_percent_value_length_threshold():
    qc_low = make_qc({"v": ["abc", "abcd", "ab"]})
    _, passes = apply(qc_low, node("percent_value_length", "v", "30", "==", "3"))
    qc_high = make_qc({"v": ["abc", "abcd", "ab"]})
    _, fails = apply(qc_high, node("percent_value_length", "v", "50", "==", "3"))
    assert bool(passes) is True
    assert bool(fails) is False


def test_unknown_check_raises_genie_exception():
    qc = make_qc({"a": [1]})
    with pytest.raises(GenieException):
        apply(qc, node("not_a_real_check"))


def test_missing_column_returns_key_error_message():
    qc = make_qc({"a": [1]})
    message, result = apply(qc, node("is_unique", "missing"))
    assert message == "Key error: is_unique for missing"
    assert result is False


def test_predicates_results_are_inserted_in_reverse_order():
    qc = make_qc({"a": [1, 2, 3], "b": [1, None, 3]})
    code_block = node(
        "code_block",
        node("line", node("is_unique", "a")),
        node("line", node("not_null", "b")),
    )
    results = qc._apply_predicates([code_block])
    assert [name for name, _ in results] == ["not_null", "is_unique"]


def test_run_rejects_multiple_top_level_children():
    qc = make_qc({"a": [1]})
    tree = Tree("start", [Tree("a", []), Tree("b", [])])
    with pytest.raises(GenieException):
        qc._run(tree)


def test_run_rejects_unknown_instruction():
    qc = make_qc({"a": [1]})
    tree = Tree("start", [Tree("nope", [])])
    with pytest.raises(GenieException):
        qc._run(tree)


def test_validate_spec_reports_failure_for_garbage_input():
    message, valid = QualityChecker.validate_spec("@@@ this is not a valid spec !!!")
    assert valid is False
    assert isinstance(message, str)
