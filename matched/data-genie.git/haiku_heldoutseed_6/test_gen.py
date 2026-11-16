import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_tree(data, *children):
    return Tree(data, list(children))


def make_values_tree(*values):
    # has_one_of / has_columns expect children[k].children to be tokens with .value
    return Tree("values", [Token("ESCAPED_STRING", '"{}"'.format(v)) for v in values])


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_init_rejects_frame_with_all_null_columns():
    df = pd.DataFrame({"a": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_init_accepts_nonempty_frame_with_defaults():
    df = pd.DataFrame({"a": [1, 2]})
    checker = QualityChecker(df)
    assert checker._obj is df
    assert checker.ignore_column_case is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "comparator, rhs, lhs, expected",
    [
        (">", 2, 3, True),
        (">", 2, 2, False),
        ("==", 2, 2, True),
        ("==", 2, 3, False),
        ("<", 2, 1, True),
        ("<", 2, 2, False),
        ("!=", 2, 2, False),  # unsupported comparator always returns False
    ],
)
def test_comparator_to_fn(comparator, rhs, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert fn(lhs) is expected


def test_treat_column_name_respects_ignore_case():
    checker = QualityChecker(pd.DataFrame({"Abc": [1]}))
    assert checker._treat_column_name("Abc") == "Abc"
    checker.ignore_column_case = True
    assert checker._treat_column_name("Abc") == "abc"


# ---------------------------------------------------------------------------
# Individual DSL checks
# ---------------------------------------------------------------------------

def test_quantile_median_passes():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    node = make_tree("quantile", "x", "0.5", ">", "2")
    name, passed = QualityChecker(df)._apply_quantile(node)
    assert name == "quantile"
    assert passed is True


def test_quantile_median_fails_when_threshold_too_high():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    node = make_tree("quantile", "x", "0.5", "<", "2")
    _, passed = QualityChecker(df)._apply_quantile(node)
    assert passed is False


def test_date_validation_accepts_valid_dates():
    df = pd.DataFrame({"d": ["2020-01-01", "2021-06-15"]})
    name, passed = QualityChecker(df)._apply_date_validation(make_tree("is_date", "d"))
    assert name == "is_date"
    assert passed is True


def test_date_validation_reports_parse_error():
    df = pd.DataFrame({"d": ["not a date at all"]})
    message, passed = QualityChecker(df)._apply_date_validation(make_tree("is_date", "d"))
    assert passed is False
    assert message.startswith("Date parse error: is_date for d")


def test_has_one_of_passes_when_all_values_allowed():
    df = pd.DataFrame({"c": ["a", "b", "a"]})
    node = make_tree("has_one_of", "c", make_values_tree("a", "b"))
    name, passed = QualityChecker(df)._apply_has_one_of(node)
    assert name == "has_one_of"
    assert passed is True


def test_has_one_of_fails_for_disallowed_value():
    df = pd.DataFrame({"c": ["a", "z"]})
    node = make_tree("has_one_of", "c", make_values_tree("a", "b"))
    _, passed = QualityChecker(df)._apply_has_one_of(node)
    assert passed is False


def test_has_one_of_ignore_case_flag():
    df = pd.DataFrame({"c": ["A", "B"]})
    checker = QualityChecker(df)

    strict = make_tree("has_one_of", "c", make_values_tree("a", "b"), "False")
    _, strict_passed = checker._apply_has_one_of(strict)
    assert strict_passed is False

    relaxed = make_tree("has_one_of", "c", make_values_tree("a", "b"), "True")
    _, relaxed_passed = checker._apply_has_one_of(relaxed)
    assert relaxed_passed is True


def test_value_length_exact_match_passes():
    df = pd.DataFrame({"c": ["ab", "cd"]})
    node = make_tree("value_length", "c", "==", "2")
    name, passed = QualityChecker(df)._apply_value_length(node)
    assert name == "value_length"
    assert passed is True


def test_value_length_fails_with_mixed_lengths():
    df = pd.DataFrame({"c": ["ab", "abc"]})
    node = make_tree("value_length", "c", "==", "2")
    _, passed = QualityChecker(df)._apply_value_length(node)
    assert passed is False


def test_value_length_ignore_nulls_skips_blank_and_null_values():
    df = pd.DataFrame({"c": ["abc", "", None, "xyz"]})
    node = make_tree("value_length", "c", "True", "==", "3")
    _, passed = QualityChecker(df)._apply_value_length(node)
    assert passed is True


def test_percent_value_length_meets_threshold():
    df = pd.DataFrame({"c": ["abc", "abcd", "ab", "xyz"]})
    # children: column, percent, (ignored comparator slot), rhs
    node = make_tree("percent_value_length", "c", "50", "==", "3")
    name, passed = QualityChecker(df)._apply_percent_value_length(node)
    assert name == "percent_value_length"
    assert passed is True


def test_percent_value_length_ignore_nulls_excludes_nulls_from_numerator():
    df = pd.DataFrame({"c": ["abc", None, "abc", "ab"]})
    # ignore_nulls=True: 2 passing rows out of 4 total rows = 50%
    node = make_tree("percent_value_length", "c", "50", "True", "==", "3")
    _, passed = QualityChecker(df)._apply_percent_value_length(node)
    assert passed is True

    node_strict = make_tree("percent_value_length", "c", "60", "True", "==", "3")
    _, passed_strict = QualityChecker(df)._apply_percent_value_length(node_strict)
    assert passed_strict is False


def test_has_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    node = make_tree("has_columns", make_values_tree("a", "b"))
    name, passed = QualityChecker(df)._apply_has_columns(node)
    assert name == "has_columns"
    assert passed is True


def test_has_columns_fails_when_one_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    node = make_tree("has_columns", make_values_tree("a", "c"))
    _, passed = QualityChecker(df)._apply_has_columns(node)
    assert passed is False


def test_has_columns_ignore_case():
    df = pd.DataFrame({"a": [1]})
    checker = QualityChecker(df)

    node_strict = make_tree("has_columns", make_values_tree("A"))
    _, strict_passed = checker._apply_has_columns(node_strict)
    assert strict_passed is False

    node_relaxed = make_tree("has_columns", make_values_tree("A"), "True")
    _, relaxed_passed = checker._apply_has_columns(node_relaxed)
    assert relaxed_passed is True


# ---------------------------------------------------------------------------
# _apply_check dispatch
# ---------------------------------------------------------------------------

def test_apply_check_row_count():
    df = pd.DataFrame({"a": [1, 2, 3]})
    checker = QualityChecker(df)
    name, passed = checker._apply_check([make_tree("row_count", "==", "3")])
    assert name == "row_count"
    assert passed is True


@pytest.mark.parametrize(
    "check_name, data, expected",
    [
        ("is_unique", {"x": [1, 2, 3]}, True),
        ("is_unique", {"x": [1, 1, 3]}, False),
        ("not_null", {"x": [1, 2, 3]}, True),
        ("not_null", {"x": [1, None, 3]}, False),
        ("is_positive", {"x": [1, 2, 3]}, True),
        ("is_positive", {"x": [1, 0, 3]}, False),
    ],
)
def test_apply_check_column_predicates(check_name, data, expected):
    df = pd.DataFrame(data)
    checker = QualityChecker(df)
    name, passed = checker._apply_check([make_tree(check_name, "x")])
    assert name == check_name
    assert passed is expected


def test_apply_check_missing_column_returns_key_error_message():
    df = pd.DataFrame({"x": [1, 2]})
    checker = QualityChecker(df)
    message, passed = checker._apply_check([make_tree("is_unique", "missing")])
    assert passed is False
    assert message == "Key error: is_unique for missing"


def test_apply_check_unknown_check_raises():
    df = pd.DataFrame({"x": [1, 2]})
    checker = QualityChecker(df)
    with pytest.raises(GenieException):
        checker._apply_check([make_tree("not_a_real_check", "x")])


# ---------------------------------------------------------------------------
# Predicate and top-level run dispatch
# ---------------------------------------------------------------------------

def test_apply_predicates_inserts_results_in_reverse_order():
    df = pd.DataFrame({"x": [1, 2, 3]})
    checker = QualityChecker(df)
    predicates = [
        make_tree(
            "code_block",
            make_tree("check", make_tree("is_unique", "x")),
            make_tree("check", make_tree("row_count", ">", "1")),
        )
    ]
    results = checker._apply_predicates(predicates)
    assert [name for name, _ in results] == ["row_count", "is_unique"]
    assert all(passed for _, passed in results)


def test_run_returns_results_for_predicates_instruction():
    df = pd.DataFrame({"x": [1, 2, 3]})
    checker = QualityChecker(df)
    ast = make_tree(
        "start",
        make_tree(
            "predicates",
            make_tree("code_block", make_tree("check", make_tree("row_count", ">", "0"))),
        ),
    )
    results = checker._run(ast)
    assert results == [("row_count", True)]


def test_run_rejects_unexpected_child_count():
    df = pd.DataFrame({"x": [1]})
    checker = QualityChecker(df)
    with pytest.raises(GenieException):
        checker._run(make_tree("start"))


def test_validate_spec_reports_invalid_spec():
    message, ok = QualityChecker.validate_spec("@@@ definitely not a valid spec !!!")
    assert ok is False
    assert isinstance(message, str)
