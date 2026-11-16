import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _str_token(value):
    return Token("STRING", value)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

def test_empty_or_all_nan_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


def test_non_empty_dataframe_accepted_with_default_case_setting():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)
    assert qc.ignore_column_case is False
    assert list(qc._obj.columns) == ["a"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "comparator, rhs, lhs, expected",
    [
        (">", 2, 3, True),
        (">", 3, 3, False),
        ("==", 3, 3, True),
        ("==", 2, 3, False),
        ("<", 4, 3, True),
        ("<", 3, 3, False),
        ("!=", 1, 3, False),  # unsupported comparator always returns False
    ],
)
def test_comparator_to_fn(comparator, rhs, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert bool(fn(lhs)) is expected


def test_treat_column_name_respects_ignore_case_flag():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    assert qc._treat_column_name("MixedCase") == "MixedCase"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MixedCase") == "mixedcase"


# ---------------------------------------------------------------------------
# Individual checks dispatched through _apply_check
# ---------------------------------------------------------------------------

def test_row_count_check():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    name, ok = qc._apply_check([Tree("row_count", [">", "2"])])
    assert name == "row_count"
    assert bool(ok) is True

    name, ok = qc._apply_check([Tree("row_count", ["==", "2"])])
    assert bool(ok) is False


def test_is_unique_check():
    unique = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    name, ok = unique._apply_check([Tree("is_unique", ["a"])])
    assert name == "is_unique"
    assert bool(ok) is True

    duplicated = QualityChecker(pd.DataFrame({"a": [1, 1, 3]}))
    _, ok = duplicated._apply_check([Tree("is_unique", ["a"])])
    assert bool(ok) is False


def test_not_null_check():
    clean = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    _, ok = clean._apply_check([Tree("not_null", ["a"])])
    assert bool(ok) is True

    with_null = QualityChecker(pd.DataFrame({"a": [1.0, None]}))
    _, ok = with_null._apply_check([Tree("not_null", ["a"])])
    assert bool(ok) is False


def test_is_positive_check():
    positive = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    _, ok = positive._apply_check([Tree("is_positive", ["a"])])
    assert bool(ok) is True

    with_zero = QualityChecker(pd.DataFrame({"a": [1, 0, 3]}))
    _, ok = with_zero._apply_check([Tree("is_positive", ["a"])])
    assert bool(ok) is False


def test_has_one_of_check():
    df = pd.DataFrame({"color": ["red", "blue", "red"]})
    qc = QualityChecker(df)
    allowed = Tree("values", [_str_token('"red"'), _str_token('"blue"')])
    name, ok = qc._apply_check([Tree("has_one_of", ["color", allowed])])
    assert name == "has_one_of"
    assert bool(ok) is True

    narrow = Tree("values", [_str_token('"red"')])
    _, ok = qc._apply_check([Tree("has_one_of", ["color", narrow])])
    assert bool(ok) is False


def test_has_one_of_ignore_case_check():
    df = pd.DataFrame({"color": ["RED", "Blue"]})
    qc = QualityChecker(df)
    allowed = Tree("values", [_str_token('"red"'), _str_token('"blue"')])

    _, ok = qc._apply_check([Tree("has_one_of", ["color", allowed, "True"])])
    assert bool(ok) is True

    _, ok = qc._apply_check([Tree("has_one_of", ["color", allowed, "False"])])
    assert bool(ok) is False


def test_quantile_check():
    qc = QualityChecker(pd.DataFrame({"n": [1, 2, 3, 4, 5]}))
    # median of 1..5 is 3.0
    name, ok = qc._apply_check([Tree("quantile", ["n", "0.5", "==", "3"])])
    assert name == "quantile"
    assert bool(ok) is True

    _, ok = qc._apply_check([Tree("quantile", ["n", "0.5", "<", "3"])])
    assert bool(ok) is False


def test_is_date_valid_and_invalid():
    valid = QualityChecker(pd.DataFrame({"d": ["2020-01-01", "2021-06-15"]}))
    name, ok = valid._apply_check([Tree("is_date", ["d"])])
    assert name == "is_date"
    assert bool(ok) is True

    invalid = QualityChecker(pd.DataFrame({"d": ["not-a-date-xyz"]}))
    message, ok = invalid._apply_check([Tree("is_date", ["d"])])
    assert bool(ok) is False
    assert message.startswith("Date parse error")
    assert "d" in message


def test_value_length_without_ignoring_nulls():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "def"]}))
    name, ok = qc._apply_check([Tree("value_length", ["s", "==", "3"])])
    assert name == "value_length"
    assert bool(ok) is True

    _, ok = qc._apply_check([Tree("value_length", ["s", "==", "4"])])
    assert bool(ok) is False


def test_value_length_ignoring_nulls_and_blanks():
    df = pd.DataFrame({"s": ["abc", None, "   "]})
    qc = QualityChecker(df)

    # Nulls and whitespace-only strings are dropped when ignoring nulls.
    _, ok = qc._apply_check([Tree("value_length", ["s", "True", "==", "3"])])
    assert bool(ok) is True

    # Without ignoring nulls, "None" (4 chars) and "   " (3 chars) both count.
    _, ok = qc._apply_check([Tree("value_length", ["s", "False", "==", "3"])])
    assert bool(ok) is False


def test_percent_value_length_meets_threshold():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "de", "fghi", "jkl"]}))
    # Two of four values have length 3 -> 50%, which meets a 50% threshold.
    name, ok = qc._apply_check(
        [Tree("percent_value_length", ["s", "50", "==", "3"])]
    )
    assert name == "percent_value_length"
    assert bool(ok) is True


def test_percent_value_length_below_threshold():
    qc = QualityChecker(pd.DataFrame({"s": ["abc", "de", "fghi", "jkl"]}))
    _, ok = qc._apply_check(
        [Tree("percent_value_length", ["s", "80", "==", "3"])]
    )
    assert bool(ok) is False


def test_has_columns_check():
    qc = QualityChecker(pd.DataFrame({"a": [1], "b": [2]}))
    cols = Tree("columns", [_str_token('"a"'), _str_token('"b"')])
    name, ok = qc._apply_check([Tree("has_columns", [cols])])
    assert name == "has_columns"
    assert bool(ok) is True

    missing = Tree("columns", [_str_token('"a"'), _str_token('"c"')])
    _, ok = qc._apply_check([Tree("has_columns", [missing])])
    assert bool(ok) is False


def test_has_columns_ignore_case_check():
    qc = QualityChecker(pd.DataFrame({"A": [1], "B": [2]}))
    cols = Tree("columns", [_str_token('"a"'), _str_token('"b"')])

    _, ok = qc._apply_check([Tree("has_columns", [cols, "True"])])
    assert bool(ok) is True

    _, ok = qc._apply_check([Tree("has_columns", [cols, "False"])])
    assert bool(ok) is False


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException):
        qc._apply_check([Tree("no_such_check", ["a"])])


def test_missing_column_reports_key_error_instead_of_raising():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    message, ok = qc._apply_check([Tree("is_unique", ["missing"])])
    assert bool(ok) is False
    assert message == "Key error: is_unique for missing"


# ---------------------------------------------------------------------------
# Predicate and run plumbing
# ---------------------------------------------------------------------------

def test_apply_predicates_inserts_results_at_front():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    first = Tree("check", [Tree("row_count", [">", "0"])])
    second = Tree("check", [Tree("row_count", [">", "10"])])
    predicates = [Tree("code_block", [first, second])]

    results = qc._apply_predicates(predicates)
    # Each result is inserted at index 0, so the second check comes first.
    assert [name for name, _ in results] == ["row_count", "row_count"]
    assert [bool(ok) for _, ok in results] == [False, True]


def test_run_evaluates_parsed_tree(monkeypatch):
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    tree = Tree(
        "start",
        [
            Tree(
                "predicates",
                [
                    Tree(
                        "code_block",
                        [Tree("check", [Tree("row_count", [">", "1"])])],
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(
        QualityChecker, "_parse_spec", staticmethod(lambda spec: tree)
    )

    results = qc.run("ignored spec", ignore_column_case=False)
    assert [(name, bool(ok)) for name, ok in results] == [("row_count", True)]
    assert qc.ignore_column_case is False


def test_run_rejects_tree_with_wrong_number_of_children(monkeypatch):
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    monkeypatch.setattr(
        QualityChecker, "_parse_spec", staticmethod(lambda spec: Tree("start", []))
    )
    with pytest.raises(GenieException):
        qc.run("ignored spec")


def test_validate_spec_reports_success_and_failure(monkeypatch):
    ok_tree = Tree("start", [])
    monkeypatch.setattr(
        QualityChecker, "_parse_spec", staticmethod(lambda spec: ok_tree)
    )
    assert QualityChecker.validate_spec("spec") == (None, True)

    def _fail(spec):
        raise ValueError("bad spec")

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(_fail))
    assert QualityChecker.validate_spec("spec") == ("bad spec", False)
