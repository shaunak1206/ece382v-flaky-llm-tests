import numpy as np
import pandas as pd
import pytest

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _spec(*checks):
    return "apply checks {\n" + "\n".join(checks) + "\n}"


def _run(df, *checks, ignore_column_case=False):
    return QualityChecker(df).run(_spec(*checks), ignore_column_case=ignore_column_case)


def _only(results):
    assert len(results) == 1
    name, ok = results[0]
    return name, bool(ok)


def test_constructor_rejects_all_nan_frame():
    df = pd.DataFrame({"a": [None, None], "b": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_constructor_rejects_empty_frame():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_constructor_keeps_frame_with_some_data():
    df = pd.DataFrame({"a": [None, None], "b": [1, 2]})
    qc = QualityChecker(df)
    assert qc.ignore_column_case is False
    assert _only(qc.run(_spec("row_count == 2"))) == ("row_count", True)


def test_validate_spec_reports_good_and_bad_specs():
    msg, ok = QualityChecker.validate_spec(_spec("is_unique(id)"))
    assert msg is None
    assert ok is True

    msg, ok = QualityChecker.validate_spec("apply checks { not_a_check(id) }")
    assert ok is False
    assert isinstance(msg, str) and msg


def test_comparator_to_fn_operators():
    assert QualityChecker._comparator_to_fn(">", 3)(4)
    assert not QualityChecker._comparator_to_fn(">", 3)(3)
    assert QualityChecker._comparator_to_fn("==", 3)(3)
    assert not QualityChecker._comparator_to_fn("==", 3)(4)
    assert QualityChecker._comparator_to_fn("<", 3)(2)
    assert not QualityChecker._comparator_to_fn("<", 3)(3)
    # Unsupported comparators fall through to a constant False.
    assert not QualityChecker._comparator_to_fn(">=", 3)(100)


def test_row_count_comparisons():
    df = pd.DataFrame({"id": [1, 2, 3]})
    assert _only(_run(df, "row_count > 2")) == ("row_count", True)
    assert _only(_run(df, "row_count > 3")) == ("row_count", False)
    assert _only(_run(df, "row_count == 3")) == ("row_count", True)
    assert _only(_run(df, "row_count < 3")) == ("row_count", False)


def test_is_unique_detects_duplicates():
    assert _only(_run(pd.DataFrame({"id": [1, 2, 3]}), "is_unique(id)")) == ("is_unique", True)
    assert _only(_run(pd.DataFrame({"id": [1, 2, 2]}), "is_unique(id)")) == ("is_unique", False)


def test_not_null_detects_missing_values():
    assert _only(_run(pd.DataFrame({"x": [1, 2, 3]}), "is_not_null(x)")) == ("not_null", True)
    assert _only(_run(pd.DataFrame({"x": [1, None, 3]}), "is_not_null(x)")) == ("not_null", False)


def test_is_positive_requires_strictly_positive_values():
    assert _only(_run(pd.DataFrame({"x": [1, 2, 3]}), "is_positive(x)")) == ("is_positive", True)
    assert _only(_run(pd.DataFrame({"x": [1, 0, 3]}), "is_positive(x)")) == ("is_positive", False)


def test_has_one_of_membership_and_ignore_case():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    assert _only(_run(df, 'has_one_of(color, ["red", "blue"])')) == ("has_one_of", True)
    assert _only(_run(df, 'has_one_of(color, ["red"])')) == ("has_one_of", False)

    mixed = pd.DataFrame({"color": ["Red", "BLUE"]})
    assert _only(_run(mixed, 'has_one_of(color, ["red", "blue"])')) == ("has_one_of", False)
    assert _only(_run(mixed, 'has_one_of(color, ["red", "blue"], ignore_case=True)')) == ("has_one_of", True)


def test_has_columns_requires_all_named_columns():
    df = pd.DataFrame({"Name": ["a"], "age": [1]})
    assert _only(_run(df, 'has_columns(["Name", "age"])')) == ("has_columns", True)
    assert _only(_run(df, 'has_columns(["Name", "zip"])')) == ("has_columns", False)
    # Case-sensitive by default, case-insensitive with the flag.
    assert _only(_run(df, 'has_columns(["name"])')) == ("has_columns", False)
    assert _only(_run(df, 'has_columns(["name"], ignore_case=True)')) == ("has_columns", True)


def test_quantile_comparisons():
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
    assert _only(_run(df, "quantile(score, 0.5) == 3")) == ("quantile", True)
    assert _only(_run(df, "quantile(score, 0.5) > 3")) == ("quantile", False)
    # Linear interpolation: the 0.25 quantile of 1..5 is 2.0.
    assert _only(_run(df, "quantile(score, 0.25) == 2")) == ("quantile", True)


def test_is_date_accepts_parseable_dates():
    df = pd.DataFrame({"when": ["2024-01-15", "2024-02-01"]})
    assert _only(_run(df, "is_date(when)")) == ("is_date", True)


def test_is_date_reports_parse_error():
    df = pd.DataFrame({"when": ["2024-01-15", "not-a-date"]})
    name, ok = _only(_run(df, "is_date(when)"))
    assert ok is False
    assert name == "Date parse error: is_date for when"


def test_value_length_requires_exact_length():
    df = pd.DataFrame({"code": ["abc", "def", "ghi"]})
    assert _only(_run(df, "value_length(code) == 3")) == ("value_length", True)
    assert _only(_run(df, "value_length(code) == 4")) == ("value_length", False)

    mixed = pd.DataFrame({"code": ["abc", "de"]})
    assert _only(_run(mixed, "value_length(code) == 3")) == ("value_length", False)


def test_value_length_ignore_nulls_skips_blank_strings():
    def blank_df():
        # "  " has length 2, so it only passes when blanks are ignored.
        return pd.DataFrame({"code": ["abc", "  ", "def"]})

    assert _only(_run(blank_df(), "value_length(code, ignore_nulls=True) == 3")) == ("value_length", True)
    assert _only(_run(blank_df(), "value_length(code, ignore_nulls=False) == 3")) == ("value_length", False)


def test_percent_value_length_threshold():
    def names_df():
        return pd.DataFrame({"name": ["ab", "cd", "e", "fgh"]})

    # Two of four values have length 2, which is exactly 50 percent.
    assert _only(_run(names_df(), "percent_of_values_have_length(name, pass_percent_threshold=50) == 2")) == (
        "percent_value_length",
        True,
    )
    assert _only(_run(names_df(), "percent_of_values_have_length(name, pass_percent_threshold=51) == 2")) == (
        "percent_value_length",
        False,
    )


def test_percent_value_length_ignore_nulls_drops_blank_rows():
    def blank_df():
        return pd.DataFrame({"name": ["ab", "  ", "cd"]})

    spec = "percent_of_values_have_length(name, pass_percent_threshold=80, ignore_nulls={}) == 2"
    # Without ignoring nulls all three rows have length 2 (100 percent).
    assert _only(_run(blank_df(), spec.format("False"))) == ("percent_value_length", True)
    # Ignoring nulls drops the blank row from the passing count (2 of 3, about 66.7 percent).
    assert _only(_run(blank_df(), spec.format("True"))) == ("percent_value_length", False)


def test_results_come_back_in_reverse_spec_order():
    df = pd.DataFrame({"id": [1, 2, 3]})
    results = _run(df, "row_count > 0", "is_unique(id)", "not_null(id)")
    assert [name for name, _ in results] == ["not_null", "is_unique", "row_count"]


def test_multiple_apply_blocks_raise_genie_exception():
    spec = _spec("row_count > 0") + "\n" + _spec("row_count > 0")
    with pytest.raises(GenieException):
        QualityChecker(pd.DataFrame({"id": [1, 2, 3]})).run(spec)


def test_unknown_column_reports_key_error():
    df = pd.DataFrame({"id": [1, 2, 3]})
    name, ok = _only(_run(df, "is_unique(missing)"))
    assert ok is False
    assert name == "Key error: is_unique for missing"


def test_ignore_column_case_matches_mixed_case_columns():
    name, ok = _only(_run(pd.DataFrame({"Score": [1, 2, 3]}), "is_positive(score)"))
    assert ok is False
    assert name.startswith("Key error")

    df = pd.DataFrame({"Score": [1, 2, 3]})
    assert _only(_run(df, "is_positive(Score)", ignore_column_case=True)) == ("is_positive", True)


def test_dataframe_accessor_runs_spec():
    df = pd.DataFrame({"id": [1, 2, 3]})
    results = df.dqc.run(_spec("is_unique(id)"))
    assert [(name, bool(ok)) for name, ok in results] == [("is_unique", True)]
