"""Unit tests for QualityChecker (genie_pkg/dqc.py).

Each test builds a small pandas DataFrame, runs DSL checks through
QualityChecker and asserts on the (name, passed) pairs it returns.
"""

import pandas as pd
import pytest
from lark import Tree
from lark.exceptions import LarkError

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _spec(*checks):
    return "apply checks { " + " ".join(checks) + " }"


def _check(df, *checks, ignore_case=False):
    return QualityChecker(df).run(_spec(*checks), ignore_column_case=ignore_case)


def _outcomes(results):
    return [(name, bool(passed)) for name, passed in results]


def _frame():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


def test_empty_or_all_nan_dataframe_rejected():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_partially_null_column_is_accepted():
    df = pd.DataFrame({"a": [None, 1], "b": [None, None]})
    qc = QualityChecker(df)
    assert qc.ignore_column_case is False


@pytest.mark.parametrize("check, expected", [
    ("row_count > 2", True),
    ("row_count > 3", False),
    ("row_count == 3", True),
    ("row_count < 3", False),
])
def test_row_count_comparators(check, expected):
    assert _outcomes(_check(_frame(), check)) == [("row_count", expected)]


def test_has_columns_present_and_missing():
    assert _outcomes(_check(_frame(), 'has_columns(["id", "name"])')) == [("has_columns", True)]
    assert _outcomes(_check(_frame(), 'has_columns(["id", "email"])')) == [("has_columns", False)]


def test_has_columns_ignore_case_flag():
    df = pd.DataFrame({"Name": [1], "AGE": [2]})
    assert _outcomes(_check(df, 'has_columns(["name", "age"], ignore_case=True)')) == [("has_columns", True)]
    assert _outcomes(_check(df, 'has_columns(["name", "age"])')) == [("has_columns", False)]


@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3], True),
    ([1, 1, 2], False),
])
def test_is_unique_detects_duplicates(values, expected):
    assert _outcomes(_check(pd.DataFrame({"id": values}), "is_unique(id)")) == [("is_unique", expected)]


def test_not_null_and_is_positive_flag_bad_values():
    assert _outcomes(_check(pd.DataFrame({"n": [1.0, 2.0]}), "is_not_null(n)")) == [("not_null", True)]
    assert _outcomes(_check(pd.DataFrame({"n": [1.0, None]}), "is_not_null(n)")) == [("not_null", False)]
    assert _outcomes(_check(pd.DataFrame({"n": [1, 2, 3]}), "is_positive(n)")) == [("is_positive", True)]
    assert _outcomes(_check(pd.DataFrame({"n": [1, 0, 3]}), "is_positive(n)")) == [("is_positive", False)]


def test_has_one_of_exact_and_ignore_case():
    exact = pd.DataFrame({"color": ["red", "blue", "red"]})
    assert _outcomes(_check(exact, 'has_one_of(color, ["red", "blue"])')) == [("has_one_of", True)]

    mixed = pd.DataFrame({"color": ["RED", "Blue"]})
    assert _outcomes(_check(mixed, 'has_one_of(color, ["red", "blue"], ignore_case=True)')) == [("has_one_of", True)]
    assert _outcomes(_check(mixed, 'has_one_of(color, ["red", "blue"])')) == [("has_one_of", False)]


@pytest.mark.parametrize("check, expected", [
    ("quantile(score,0.5) == 3", True),
    ("quantile(score,0.5) > 3", False),
    ("quantile(score,0.5) < 4", True),
])
def test_quantile_median_comparisons(check, expected):
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
    assert _outcomes(_check(df, check)) == [("quantile", expected)]


@pytest.mark.parametrize("check, name", [
    ("is_unique(missing)", "is_unique"),
    ("is_not_null(missing)", "not_null"),
    ("is_positive(missing)", "is_positive"),
    ('has_one_of(missing, ["a"])', "has_one_of"),
    ("quantile(missing,0.5) > 1", "quantile"),
])
def test_missing_column_reports_key_error(check, name):
    results = _check(_frame(), check)
    assert results == [("Key error: {} for missing".format(name), False)]


def test_is_date_iso_valid_invalid_and_missing():
    valid = pd.DataFrame({"when": ["2024-01-01", "2024-02-29"]})
    assert _outcomes(_check(valid, "is_date(when)")) == [("is_date", True)]

    invalid = pd.DataFrame({"when": ["2024-01-01", "not a date"]})
    assert _outcomes(_check(invalid, "is_date(when)")) == [
        ("Date parse error: is_date for when", False)
    ]

    # The generic exception handler in _apply_date_validation also catches the
    # KeyError for a missing column, so it fails closed with a parse-error name.
    name, passed = _check(_frame(), "is_date(missing)")[0]
    assert passed is False
    assert name.startswith("Date parse error")


def test_value_length_exact_match():
    assert _outcomes(_check(pd.DataFrame({"code": ["abc", "def"]}), "value_length(code) == 3")) == [
        ("value_length", True)
    ]
    assert _outcomes(_check(pd.DataFrame({"code": ["abc", "def"]}), "value_length(code) == 2")) == [
        ("value_length", False)
    ]
    assert _outcomes(_check(pd.DataFrame({"code": ["abc", "de"]}), "value_length(code) == 3")) == [
        ("value_length", False)
    ]


def test_value_length_null_and_blank_handling():
    with_null = pd.DataFrame({"code": ["abc", None]})
    assert _outcomes(_check(with_null, "value_length(code) == 3")) == [("value_length", False)]
    assert _outcomes(_check(with_null, "value_length(code,ignore_nulls=True) == 3")) == [
        ("value_length", True)
    ]

    with_blank = pd.DataFrame({"code": ["abc", ""]})
    assert _outcomes(_check(with_blank, "value_length(code) == 3")) == [("value_length", False)]
    assert _outcomes(_check(with_blank, "value_length(code,ignore_nulls=True) == 3")) == [
        ("value_length", True)
    ]


def test_percent_value_length_threshold_boundary():
    def codes():
        return pd.DataFrame({"code": ["abc", "abc", "x", "yyyy"]})

    # Two of four rows have length 3, which is exactly 50 percent.
    assert _outcomes(_check(
        codes(), "percent_of_values_have_length(code,pass_percent_threshold=50) == 3"
    )) == [("percent_value_length", True)]
    assert _outcomes(_check(
        codes(), "percent_of_values_have_length(code,pass_percent_threshold=51) == 3"
    )) == [("percent_value_length", False)]


def test_percent_value_length_ignore_nulls_drops_missing_rows():
    def codes():
        return pd.DataFrame({"code": ["abc", None, None]})

    # Without ignore_nulls the two missing rows count as length 0 (66.7 percent).
    assert _outcomes(_check(
        codes(), "percent_of_values_have_length(code,pass_percent_threshold=50) == 0"
    )) == [("percent_value_length", True)]
    # With ignore_nulls the missing rows are dropped, leaving 0 percent.
    assert _outcomes(_check(
        codes(), "percent_of_values_have_length(code,pass_percent_threshold=50,ignore_nulls=True) == 0"
    )) == [("percent_value_length", False)]


def test_comparator_helper_unknown_operator_never_passes():
    assert QualityChecker._comparator_to_fn(">", 2)(3) is True
    assert QualityChecker._comparator_to_fn("==", 2)(2) is True
    assert QualityChecker._comparator_to_fn("<", 2)(1) is True
    assert QualityChecker._comparator_to_fn("!=", 2)(2) is False


def test_invalid_trees_and_unknown_checks_raise_genie_exception():
    qc = QualityChecker(_frame())
    with pytest.raises(GenieException):
        qc._apply_check([Tree("mystery_check", [])])
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("first", []), Tree("second", [])]))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("bogus", [])]))


def test_multiple_instructions_rejected_by_run():
    qc = QualityChecker(_frame())
    with pytest.raises(GenieException):
        qc.run("apply checks { row_count > 1 } apply checks { row_count > 1 }")


def test_validate_spec_accepts_good_and_rejects_bad_specs():
    assert QualityChecker.validate_spec(_spec("row_count > 1")) == (None, True)
    message, passed = QualityChecker.validate_spec("apply checks { not_a_check(id) }")
    assert passed is False
    assert message


def test_run_raises_lark_error_for_unparseable_spec():
    with pytest.raises(LarkError):
        QualityChecker(_frame()).run("apply checks { not_a_check(id) }")


def test_ignore_column_case_matches_across_case():
    # Flag on: both the spec name and the frame's column are lowercased.
    df = pd.DataFrame({"Name": ["a", "b"]})
    assert _outcomes(_check(df, "is_unique(Name)", ignore_case=True)) == [("is_unique", True)]

    # Flag off: the lowercase spec name does not match the frame's "Name" column.
    df = pd.DataFrame({"Name": ["a", "b"]})
    assert _check(df, "is_unique(name)") == [("Key error: is_unique for name", False)]


def test_results_come_back_in_reverse_check_order():
    results = QualityChecker(_frame()).run(
        "apply checks { row_count > 1 is_unique(id) is_not_null(id) }"
    )
    assert [name for name, _ in results] == ["not_null", "is_unique", "row_count"]
