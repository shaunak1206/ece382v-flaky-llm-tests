import numpy as np
import pandas as pd
import pytest

from genie_pkg.dqc import QualityChecker


def run(df, body, **kwargs):
    return QualityChecker(df).run("apply checks { %s }" % body, **kwargs)


def test_empty_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises():
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_comparator_to_fn_operators():
    assert QualityChecker._comparator_to_fn(">", 3)(5) is True
    assert QualityChecker._comparator_to_fn(">", 3)(2) is False
    assert QualityChecker._comparator_to_fn("<", 3)(2) is True
    assert QualityChecker._comparator_to_fn("==", 3)(3) is True
    assert QualityChecker._comparator_to_fn("==", 3)(4) is False


def test_comparator_to_fn_unknown_comparator_is_false():
    fn = QualityChecker._comparator_to_fn("!=", 3)
    assert fn(100) is False
    assert fn(3) is False


def test_treat_column_name_respects_flag():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    assert qc._treat_column_name("MyCol") == "MyCol"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MyCol") == "mycol"


def test_row_count_greater_than_passes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    results = run(df, "row_count > 2")
    assert len(results) == 1
    assert results[0][0] == "row_count"
    assert bool(results[0][1]) is True


def test_row_count_equals_and_less_than():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert bool(run(df, "row_count == 3")[0][1]) is True
    assert bool(run(df, "row_count == 4")[0][1]) is False
    assert bool(run(df, "row_count < 3")[0][1]) is False
    assert bool(run(df, "row_count < 10")[0][1]) is True


def test_is_unique_true_and_false():
    unique = pd.DataFrame({"a": [1, 2, 3]})
    dup = pd.DataFrame({"a": [1, 1, 3]})
    assert bool(run(unique, "is_unique(a)")[0][1]) is True
    res = run(dup, "is_unique(a)")
    assert res[0][0] == "is_unique"
    assert bool(res[0][1]) is False


def test_not_null_checks():
    clean = pd.DataFrame({"a": [1, 2]})
    dirty = pd.DataFrame({"a": [1, np.nan]})
    res_clean = run(clean, "is_not_null(a)")
    assert res_clean[0][0] == "not_null"
    assert bool(res_clean[0][1]) is True
    assert bool(run(dirty, "is_not_null(a)")[0][1]) is False


def test_is_positive_checks():
    pos = pd.DataFrame({"a": [1, 2, 3]})
    neg = pd.DataFrame({"a": [1, 0, 3]})
    assert bool(run(pos, "is_positive(a)")[0][1]) is True
    assert bool(run(neg, "is_positive(a)")[0][1]) is False


def test_has_columns_present_and_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert bool(run(df, 'has_columns(["a","b"])')[0][1]) is True
    assert bool(run(df, 'has_columns(["a","z"])')[0][1]) is False


def test_has_columns_ignore_case():
    df = pd.DataFrame({"Alpha": [1], "Beta": [2]})
    assert bool(run(df, 'has_columns(["alpha","beta"])')[0][1]) is False
    df2 = pd.DataFrame({"Alpha": [1], "Beta": [2]})
    assert bool(run(df2, 'has_columns(["alpha","beta"], ignore_case=True)')[0][1]) is True


def test_has_one_of_pass_and_fail():
    df = pd.DataFrame({"c": ["x", "y", "x"]})
    assert bool(run(df, 'has_one_of(c, ["x","y","z"])')[0][1]) is True
    assert bool(run(df, 'has_one_of(c, ["x"])')[0][1]) is False


def test_has_one_of_ignore_case():
    df = pd.DataFrame({"c": ["x", "Y"]})
    assert bool(run(df, 'has_one_of(c, ["x","y"])')[0][1]) is False
    assert bool(run(df, 'has_one_of(c, ["x","y"], ignore_case=True)')[0][1]) is True


def test_quantile_median_comparisons():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    res = run(df, "quantile(a, 0.5) > 2")
    assert res[0][0] == "quantile"
    assert bool(res[0][1]) is True
    assert bool(run(df, "quantile(a, 0.5) < 2")[0][1]) is False
    assert bool(run(df, "quantile(a, 0.5) == 3")[0][1]) is True


def test_is_date_valid_and_invalid():
    good = pd.DataFrame({"d": ["2020-01-01", "2021-05-06"]})
    bad = pd.DataFrame({"d": ["not a date", "also nope"]})
    res_good = run(good, "is_date(d)")
    assert res_good[0] == ("is_date", True)
    res_bad = run(bad, "is_date(d)")
    assert res_bad[0][1] is False
    assert res_bad[0][0].startswith("Date parse error")


def test_value_length_uniform_and_mixed():
    uniform = pd.DataFrame({"s": ["abc", "def", "ghi"]})
    mixed = pd.DataFrame({"s": ["abc", "de", "ghi"]})
    res = run(uniform, "value_length(s) == 3")
    assert res[0][0] == "value_length"
    assert bool(res[0][1]) is True
    assert bool(run(uniform, "value_length(s) == 4")[0][1]) is False
    assert bool(run(mixed, "value_length(s) == 3")[0][1]) is False


def test_value_length_ignore_nulls_skips_blank_values():
    df = pd.DataFrame({"s": ["abc", "def", "", "   "]})
    assert bool(run(df, "value_length(s) == 3")[0][1]) is False
    assert bool(run(df, "value_length(s, ignore_nulls=True) == 3")[0][1]) is True


def test_percent_value_length_threshold():
    df = pd.DataFrame({"s": ["abc", "def", "abcd", "x"]})
    res = run(df, "percent_of_values_have_length(s, pass_percent_threshold=50) == 3")
    assert res[0][0] == "percent_value_length"
    assert bool(res[0][1]) is True
    df2 = pd.DataFrame({"s": ["abc", "def", "abcd", "x"]})
    res2 = run(df2, "percent_of_values_have_length(s, pass_percent_threshold=75) == 3")
    assert bool(res2[0][1]) is False


def test_missing_column_reports_key_error():
    df = pd.DataFrame({"a": [1, 2]})
    res = run(df, "is_unique(missing)")
    assert len(res) == 1
    assert res[0][0].startswith("Key error")
    assert res[0][1] is False


def test_ignore_column_case_lowercases_columns():
    df = pd.DataFrame({"Name": [1, 2, 3]})
    res = QualityChecker(df).run("apply checks { is_unique(NAME) }", ignore_column_case=True)
    assert bool(res[0][1]) is True


def test_multiple_checks_results_are_in_reverse_order():
    df = pd.DataFrame({"a": [1, 1, 2]})
    res = run(df, "row_count > 1\n is_unique(a)")
    assert len(res) == 2
    assert res[0][0] == "is_unique"
    assert bool(res[0][1]) is False
    assert res[1][0] == "row_count"
    assert bool(res[1][1]) is True


def test_validate_spec_valid():
    err, ok = QualityChecker.validate_spec("apply checks { row_count > 1 }")
    assert err is None
    assert ok is True


def test_validate_spec_invalid():
    err, ok = QualityChecker.validate_spec("this is not a valid spec")
    assert ok is False
    assert isinstance(err, str)
    assert err != ""
