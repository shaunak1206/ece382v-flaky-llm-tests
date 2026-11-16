import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_checker(data):
    return QualityChecker(pd.DataFrame(data))


def string_list_tree(name, values):
    """Build a Tree whose children are quoted STRING tokens, as the DSL produces."""
    return Tree(name, [Token("STRING", '"{}"'.format(v)) for v in values])


def test_accessor_is_registered():
    df = pd.DataFrame({"a": [1, 2]})
    assert isinstance(df.dqc, QualityChecker)


def test_init_stores_dataframe_and_default_case_flag():
    df = pd.DataFrame({"a": [1, 2]})
    checker = QualityChecker(df)
    assert checker._obj is df
    assert checker.ignore_column_case is False


def test_init_empty_dataframe_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_init_all_nan_columns_raises():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({"a": [None, None]}))


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


def test_treat_column_name_respects_case_flag():
    checker = make_checker({"Abc": [1]})
    assert checker._treat_column_name("Abc") == "Abc"
    checker.ignore_column_case = True
    assert checker._treat_column_name("Abc") == "abc"


def test_apply_quantile_passes():
    checker = make_checker({"a": [1, 2, 3, 4, 5]})
    node = Tree("quantile", ["a", "0.5", ">", "2"])
    assert checker._apply_quantile(node) == ("quantile", True)


def test_apply_quantile_fails():
    checker = make_checker({"a": [1, 2, 3, 4, 5]})
    node = Tree("quantile", ["a", "0.5", "<", "2"])
    assert checker._apply_quantile(node) == ("quantile", False)


def test_apply_date_validation_valid_dates():
    checker = make_checker({"d": ["2020-01-01", "2021-05-05"]})
    assert checker._apply_date_validation(Tree("is_date", ["d"])) == ("is_date", True)


def test_apply_date_validation_invalid_dates_returns_message():
    checker = make_checker({"d": ["not a date"]})
    name, ok = checker._apply_date_validation(Tree("is_date", ["d"]))
    assert ok is False
    assert name.startswith("Date parse error")
    assert "for d" in name


def test_apply_has_one_of_all_allowed():
    checker = make_checker({"c": ["a", "b", "a"]})
    node = Tree("has_one_of", ["c", string_list_tree("values", ["a", "b"])])
    assert checker._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_has_one_of_disallowed_value():
    checker = make_checker({"c": ["a", "z"]})
    node = Tree("has_one_of", ["c", string_list_tree("values", ["a", "b"])])
    assert checker._apply_has_one_of(node) == ("has_one_of", False)


def test_apply_has_one_of_ignore_case():
    checker = make_checker({"c": ["A", "b"]})
    node = Tree("has_one_of", ["c", string_list_tree("values", ["a", "B"]), "True"])
    assert checker._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_value_length_uniform_passes():
    checker = make_checker({"s": ["ab", "cd"]})
    node = Tree("value_length", ["s", "==", "2"])
    assert checker._apply_value_length(node) == ("value_length", True)


def test_apply_value_length_mixed_lengths_fails():
    checker = make_checker({"s": ["a", "bb"]})
    node = Tree("value_length", ["s", "==", "1"])
    assert checker._apply_value_length(node) == ("value_length", False)


def test_apply_value_length_ignore_nulls_skips_blank_values():
    checker = make_checker({"s": ["ab", None, "   ", "cd"]})
    node = Tree("value_length", ["s", "True", "==", "2"])
    assert checker._apply_value_length(node) == ("value_length", True)


def test_apply_percent_value_length_passes_above_threshold():
    checker = make_checker({"s": ["ab", "cd", "e"]})
    node = Tree("percent_value_length", ["s", "50", "False", "==", "2"])
    assert checker._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_percent_value_length_fails_below_threshold():
    checker = make_checker({"s": ["ab", "cd", "e"]})
    node = Tree("percent_value_length", ["s", "90", "False", "==", "2"])
    assert checker._apply_percent_value_length(node) == ("percent_value_length", False)


def test_apply_has_columns_all_present():
    checker = make_checker({"a": [1], "b": [2]})
    node = Tree("has_columns", [string_list_tree("columns", ["a", "b"])])
    assert checker._apply_has_columns(node) == ("has_columns", True)


def test_apply_has_columns_missing_column():
    checker = make_checker({"a": [1]})
    node = Tree("has_columns", [string_list_tree("columns", ["a", "b"])])
    assert checker._apply_has_columns(node) == ("has_columns", False)


def test_apply_has_columns_ignore_case():
    checker = make_checker({"A": [1], "B": [2]})
    node = Tree("has_columns", [string_list_tree("columns", ["a", "b"]), "True"])
    assert checker._apply_has_columns(node) == ("has_columns", True)


def test_apply_check_row_count():
    checker = make_checker({"a": [1, 2, 3]})
    assert checker._apply_check([Tree("row_count", [">", "2"])]) == ("row_count", True)
    assert checker._apply_check([Tree("row_count", ["==", "2"])]) == ("row_count", False)


def test_apply_check_is_unique():
    assert make_checker({"a": [1, 2]})._apply_check([Tree("is_unique", ["a"])]) == ("is_unique", True)
    assert make_checker({"a": [1, 1]})._apply_check([Tree("is_unique", ["a"])]) == ("is_unique", False)


def test_apply_check_not_null():
    assert make_checker({"a": [1, 2]})._apply_check([Tree("not_null", ["a"])]) == ("not_null", True)
    assert make_checker({"a": [1, None]})._apply_check([Tree("not_null", ["a"])]) == ("not_null", False)


def test_apply_check_is_positive():
    assert make_checker({"a": [1, 2]})._apply_check([Tree("is_positive", ["a"])]) == ("is_positive", True)
    assert make_checker({"a": [0, 2]})._apply_check([Tree("is_positive", ["a"])]) == ("is_positive", False)


def test_apply_check_unknown_check_raises_genie_exception():
    checker = make_checker({"a": [1]})
    with pytest.raises(GenieException):
        checker._apply_check([Tree("not_a_real_check", ["a"])])


def test_apply_check_missing_column_returns_key_error_result():
    checker = make_checker({"a": [1]})
    result = checker._apply_check([Tree("is_unique", ["missing"])])
    assert result == ("Key error: is_unique for missing", False)


def test_apply_predicates_code_block_results_are_reversed():
    checker = make_checker({"a": [1, 2]})
    predicates = [
        Tree(
            "code_block",
            [
                Tree("check", [Tree("row_count", [">", "0"])]),
                Tree("check", [Tree("is_unique", ["a"])]),
            ],
        )
    ]
    results = checker._apply_predicates(predicates)
    # Results are inserted at index 0, so the last check comes first.
    assert [name for name, _ in results] == ["is_unique", "row_count"]
    assert all(ok for _, ok in results)


def test_run_with_no_children_raises():
    checker = make_checker({"a": [1]})
    with pytest.raises(GenieException):
        checker._run(Tree("start", []))


def test_run_unknown_instruction_raises():
    checker = make_checker({"a": [1]})
    with pytest.raises(GenieException):
        checker._run(Tree("start", [Tree("bogus", [])]))


def test_validate_spec_rejects_garbage():
    message, ok = QualityChecker.validate_spec("@@@ not a valid spec !!!")
    assert ok is False
    assert isinstance(message, str)


def test_run_ignore_column_case_lowercases_columns(monkeypatch):
    checker = make_checker({"A": [1, 2], "B": [3, 4]})
    empty_ast = Tree("start", [Tree("predicates", [])])
    monkeypatch.setattr(
        QualityChecker, "_parse_spec", staticmethod(lambda spec: empty_ast)
    )
    result = checker.run("ignored", ignore_column_case=True)
    assert result == []
    assert checker.ignore_column_case is True
    assert list(checker._obj.columns) == ["a", "b"]
