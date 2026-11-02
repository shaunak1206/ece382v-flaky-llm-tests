import pandas as pd
import pytest

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "price": [1.0, 2.0, 3.0],
        "code": ["ab", "cd", "ef"],
        "color": ["red", "green", "red"],
        "when": ["2020-01-01", "2020-02-01", "2020-03-01"],
    })


def run(df, body, **kwargs):
    return QualityChecker(df).run("apply checks { %s }" % body, **kwargs)


def test_empty_dataframe_rejected():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_null_dataframe_rejected():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_comparator_to_fn_operators():
    assert QualityChecker._comparator_to_fn(">", 2)(3) is True
    assert QualityChecker._comparator_to_fn(">", 2)(2) is False
    assert QualityChecker._comparator_to_fn("<", 2)(1) is True
    assert QualityChecker._comparator_to_fn("==", 2)(2) is True
    assert QualityChecker._comparator_to_fn("==", 2)(3) is False


def test_comparator_to_fn_unknown_returns_false():
    assert QualityChecker._comparator_to_fn("!=", 2)(3) is False


def test_validate_spec_valid():
    msg, ok = QualityChecker.validate_spec("apply checks { row_count > 1 }")
    assert msg is None
    assert ok is True


def test_validate_spec_invalid():
    msg, ok = QualityChecker.validate_spec("apply checks { this is nonsense }")
    assert ok is False
    assert isinstance(msg, str)
    assert msg


def test_row_count_comparators():
    df = make_df()
    assert run(df, "row_count > 2") == [("row_count", True)]
    assert run(df, "row_count < 2") == [("row_count", False)]
    assert run(df, "row_count == 3") == [("row_count", True)]


def test_has_columns_present_and_missing():
    df = make_df()
    assert run(df, 'has_columns(["id", "price"])') == [("has_columns", True)]
    assert run(df, 'has_columns(["id", "nope"])') == [("has_columns", False)]


def test_has_columns_ignore_case():
    df = make_df()
    assert run(df, 'has_columns(["ID", "PRICE"])') == [("has_columns", False)]
    assert run(df, 'has_columns(["ID", "PRICE"], ignore_case=True)') == [("has_columns", True)]


def test_is_unique():
    df = make_df()
    assert run(df, "is_unique(id)") == [("is_unique", True)]
    assert run(df, "is_unique(color)") == [("is_unique", False)]


def test_not_null():
    df = make_df()
    df.loc[1, "price"] = None
    assert run(df, "is_not_null(id)") == [("not_null", True)]
    assert run(df, "is_not_null(price)") == [("not_null", False)]


def test_is_positive():
    df = make_df()
    assert run(df, "is_positive(price)") == [("is_positive", True)]
    df.loc[0, "price"] = -1.0
    assert run(df, "is_positive(price)") == [("is_positive", False)]


def test_has_one_of():
    df = make_df()
    assert run(df, 'has_one_of(color, ["red", "green", "blue"])') == [("has_one_of", True)]
    assert run(df, 'has_one_of(color, ["red"])') == [("has_one_of", False)]


def test_has_one_of_ignore_case():
    df = make_df()
    spec = 'has_one_of(color, ["RED", "GREEN"])'
    assert run(df, spec) == [("has_one_of", False)]
    assert run(df, 'has_one_of(color, ["RED", "GREEN"], ignore_case=True)') == [("has_one_of", True)]


def test_quantile():
    df = make_df()
    assert run(df, "quantile(price, 0.5) > 1") == [("quantile", True)]
    assert run(df, "quantile(price, 0.5) < 1") == [("quantile", False)]


def test_is_date():
    df = make_df()
    assert run(df, "is_date(when)") == [("is_date", True)]
    df["when"] = ["not a date", "also not", "nope"]
    results = run(df, "is_date(when)")
    assert len(results) == 1
    assert results[0][1] is False
    assert "Date parse error" in results[0][0]


def test_value_length():
    df = make_df()
    assert run(df, "value_length(code) == 2") == [("value_length", True)]
    assert run(df, "value_length(code) == 5") == [("value_length", False)]
    df.loc[0, "code"] = "abcdef"
    assert run(df, "value_length(code) == 2") == [("value_length", False)]


def test_percent_value_length():
    df = make_df()
    spec = "percent_of_values_have_length(code, pass_percent_threshold=50) == 2"
    assert run(df, spec) == [("percent_value_length", True)]
    spec = "percent_of_values_have_length(code, pass_percent_threshold=50) == 7"
    assert run(df, spec) == [("percent_value_length", False)]


def test_missing_column_reports_key_error():
    df = make_df()
    results = run(df, "is_unique(missing_col)")
    assert len(results) == 1
    assert results[0][1] is False
    assert results[0][0].startswith("Key error")
    assert "missing_col" in results[0][0]


def test_multiple_checks_reported_in_reverse_order():
    df = make_df()
    results = run(df, "row_count == 3 is_unique(id) is_positive(price)")
    assert [name for name, _ in results] == ["is_positive", "is_unique", "row_count"]
    assert all(bool(ok) for _, ok in results)


def test_dataframe_accessor_registered():
    df = make_df()
    results = df.dqc.run("apply checks { row_count == 3 }")
    assert results == [("row_count", True)]
