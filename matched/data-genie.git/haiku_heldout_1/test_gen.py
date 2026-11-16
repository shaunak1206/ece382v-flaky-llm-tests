import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(name, *children):
    """Build a single check node (e.g. row_count, is_unique) as a Tree."""
    return Tree(name, list(children))


def _token_list(*values):
    """Build a Tree whose children carry a .value attribute, as the grammar does."""
    return Tree("values", [Token("ESCAPED_STRING", v) for v in values])


def test_init_rejects_empty_or_all_nan_dataframes():
    empty = pd.DataFrame()
    all_nan = pd.DataFrame({"a": [None, None], "b": [None, None]})

    with pytest.raises(AttributeError):
        QualityChecker(empty)
    with pytest.raises(AttributeError):
        QualityChecker(all_nan)


def test_init_accepts_valid_dataframe_and_defaults_ignore_case_false():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)

    assert qc.ignore_column_case is False
    assert qc._obj is df


def test_dqc_accessor_is_registered_on_dataframe():
    df = pd.DataFrame({"a": [1, 2]})

    assert isinstance(df.dqc, QualityChecker)


@pytest.mark.parametrize(
    "comparator, rhs, lhs, expected",
    [
        (">", 2, 3, True),
        (">", 2, 2, False),
        ("==", 2, 2, True),
        ("==", 2, 3, False),
        ("<", 2, 1, True),
        ("<", 2, 2, False),
        ("!=", 2, 2, False),  # unknown comparator always yields False
        ("??", 2, 5, False),
    ],
)
def test_comparator_to_fn_handles_known_and_unknown_operators(comparator, rhs, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)

    assert fn(lhs) == expected


def test_treat_column_name_respects_ignore_column_case():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))

    assert qc._treat_column_name("MyCol") == "MyCol"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MyCol") == "mycol"


def test_row_count_check_compares_number_of_rows():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))

    assert qc._apply_check([_check("row_count", "==", "3")]) == ("row_count", True)
    assert qc._apply_check([_check("row_count", ">", "3")]) == ("row_count", False)


def test_is_unique_check():
    assert QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))._apply_check(
        [_check("is_unique", "a")]
    ) == ("is_unique", True)
    assert QualityChecker(pd.DataFrame({"a": [1, 1, 3]}))._apply_check(
        [_check("is_unique", "a")]
    ) == ("is_unique", False)


def test_not_null_check():
    assert QualityChecker(pd.DataFrame({"a": [1, 2]}))._apply_check(
        [_check("not_null", "a")]
    ) == ("not_null", True)
    assert QualityChecker(pd.DataFrame({"a": [1, None]}))._apply_check(
        [_check("not_null", "a")]
    ) == ("not_null", False)


def test_is_positive_check():
    assert QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))._apply_check(
        [_check("is_positive", "a")]
    ) == ("is_positive", True)
    assert QualityChecker(pd.DataFrame({"a": [1, 0, 3]}))._apply_check(
        [_check("is_positive", "a")]
    ) == ("is_positive", False)


def test_has_one_of_case_sensitive_and_insensitive():
    df = pd.DataFrame({"a": ["x", "Y"]})
    values = _token_list('"x"', '"y"')

    qc = QualityChecker(df.copy())
    assert qc._apply_check([_check("has_one_of", "a", values)]) == ("has_one_of", False)

    qc = QualityChecker(df.copy())
    assert qc._apply_check([_check("has_one_of", "a", values, "True")]) == ("has_one_of", True)


def test_has_columns_exact_match():
    df = pd.DataFrame({"a": [1], "b": [2]})

    qc = QualityChecker(df.copy())
    ok = qc._apply_check([_check("has_columns", _token_list('"a"', '"b"'))])
    assert ok == ("has_columns", True)

    qc = QualityChecker(df.copy())
    missing = qc._apply_check([_check("has_columns", _token_list('"a"', '"c"'))])
    assert missing == ("has_columns", False)


def test_has_columns_ignore_case_match():
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = QualityChecker(df)

    result = qc._apply_check([_check("has_columns", _token_list('"a"', '"b"'), "True")])

    assert result == ("has_columns", True)


def test_is_date_valid_and_invalid_values():
    qc = QualityChecker(pd.DataFrame({"d": ["2020-01-01", "2021-06-15"]}))
    assert qc._apply_check([_check("is_date", "d")]) == ("is_date", True)

    qc = QualityChecker(pd.DataFrame({"d": ["not a date"]}))
    name, passed = qc._apply_check([_check("is_date", "d")])
    assert passed is False
    assert name == "Date parse error: is_date for d"


def test_quantile_check_uses_column_quantile():
    qc = QualityChecker(pd.DataFrame({"col": [1, 2, 3, 4, 5]}))

    # median of 1..5 is 3.0
    assert qc._apply_check([_check("quantile", "col", "0.5", ">", "2")]) == ("quantile", True)
    assert qc._apply_check([_check("quantile", "col", "0.5", "<", "2")]) == ("quantile", False)


def test_value_length_check_requires_single_unique_length():
    qc = QualityChecker(pd.DataFrame({"col": ["abc", "def"]}))
    assert qc._apply_check([_check("value_length", "col", "==", "3")]) == ("value_length", True)

    qc = QualityChecker(pd.DataFrame({"col": ["abc", "de"]}))
    assert qc._apply_check([_check("value_length", "col", "==", "3")]) == ("value_length", False)


def test_percent_value_length_check_respects_threshold():
    df = pd.DataFrame({"col": ["abc", "de", "fgh", "ij"]})  # 2 of 4 rows have length 3 (50%)

    qc = QualityChecker(df.copy())
    assert qc._apply_check(
        [_check("percent_value_length", "col", "50", "==", "3")]
    ) == ("percent_value_length", True)

    qc = QualityChecker(df.copy())
    assert qc._apply_check(
        [_check("percent_value_length", "col", "60", "==", "3")]
    ) == ("percent_value_length", False)


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))

    with pytest.raises(GenieException):
        qc._apply_check([_check("not_a_real_check", "a")])


def test_missing_column_yields_key_error_result_instead_of_raising():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))

    name, passed = qc._apply_check([_check("is_unique", "missing")])

    assert passed is False
    assert name == "Key error: is_unique for missing"


def test_apply_predicates_collects_results_in_reverse_order():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    predicates = [
        Tree(
            "code_block",
            [
                Tree("check", [_check("row_count", ">", "0")]),
                Tree("check", [_check("row_count", ">", "5")]),
            ],
        )
    ]

    results = qc._apply_predicates(predicates)

    # Each result is inserted at position 0, so the last check comes first.
    assert results == [("row_count", False), ("row_count", True)]


def test_run_rejects_unexpected_tree_structure():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    bad_tree = Tree("start", [Tree("predicates", []), Tree("extra", [])])

    with pytest.raises(GenieException):
        qc._run(bad_tree)


def test_run_rejects_unknown_top_level_instruction():
    qc = QualityChecker(pd.DataFrame({"a": [1]}))
    tree = Tree("start", [Tree("unknown_instruction", [])])

    with pytest.raises(GenieException):
        qc._run(tree)


def test_validate_spec_reports_success_and_failure(monkeypatch):
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: None))
    assert QualityChecker.validate_spec("anything") == (None, True)

    def _boom(spec):
        raise ValueError("bad spec")

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(_boom))
    message, ok = QualityChecker.validate_spec("anything")
    assert ok is False
    assert "bad spec" in message


def test_run_applies_parsed_checks_and_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"A": [1, 2]})
    qc = QualityChecker(df)
    tree = Tree(
        "start",
        [
            Tree(
                "predicates",
                [
                    Tree(
                        "code_block",
                        [Tree("check", [_check("is_unique", "A")])],
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))

    results = qc.run("ignored spec", ignore_column_case=True)

    assert results == [("is_unique", True)]
    assert list(qc._obj.columns) == ["a"]
