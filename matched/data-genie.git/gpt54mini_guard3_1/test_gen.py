import pandas as pd
import pytest
from lark import Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_checker(df):
    return QualityChecker(df)


def test_validate_raises_on_empty_dataframe():
    df = pd.DataFrame({"a": [None], "b": [None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        _ = df.dqc


def test_comparator_to_fn_gt_eq_lt_and_default_false():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    default = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(3) is False
    assert eq(3) is True
    assert eq(4) is False
    assert lt(2) is True
    assert lt(3) is False
    assert default(999) is False


def test_treat_column_name_respects_ignore_column_case():
    qc = make_checker(pd.DataFrame({"MixedCase": [1]}))
    assert qc._treat_column_name("MixedCase") == "MixedCase"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MixedCase") == "mixedcase"


def test_apply_date_validation_success_and_failure():
    qc = make_checker(pd.DataFrame({"dt": ["2024-01-01", "2024-02-03"]}))
    ok_node = Tree("is_date", ["dt"])
    assert qc._apply_date_validation(ok_node) == ("is_date", True)

    bad_qc = make_checker(pd.DataFrame({"dt": ["not-a-date", "2024-01-01"]}))
    bad_node = Tree("is_date", ["dt"])
    msg, passed = bad_qc._apply_date_validation(bad_node)
    assert passed is False
    assert msg == "Date parse error: is_date for dt"


def test_apply_has_one_of_case_sensitive_true_and_false():
    df = pd.DataFrame({"color": ["Red", "BLUE", "green"]})
    qc = make_checker(df)

    values_tree = Tree("values", [Tree("value", ['"Red"']), Tree("value", ['"BLUE"']), Tree("value", ['"green"'])])
    node = Tree("has_one_of", ["color", values_tree])
    assert qc._apply_has_one_of(node) == ("has_one_of", True)

    # Case-sensitive mismatch
    values_tree2 = Tree("values", [Tree("value", ['"red"'])])
    node2 = Tree("has_one_of", ["color", values_tree2])
    assert qc._apply_has_one_of(node2) == ("has_one_of", False)

    # Ignore case via explicit third child
    node3 = Tree("has_one_of", ["color", values_tree2, "True"])
    assert qc._apply_has_one_of(node3) == ("has_one_of", False) is False or qc._apply_has_one_of(node3)[1] is False


def test_apply_has_one_of_ignore_case_passes_when_values_match():
    df = pd.DataFrame({"color": ["Red", "BLUE"]})
    qc = make_checker(df)
    values_tree = Tree("values", [Tree("value", ['"red"']), Tree("value", ['"blue"'])])
    node = Tree("has_one_of", ["color", values_tree, "True"])
    assert qc._apply_has_one_of(node) == ("has_one_of", True)


def test_apply_value_length_without_nulls_passes():
    df = pd.DataFrame({"code": ["ab", "cd", "ef"]})
    qc = make_checker(df)
    node = Tree("value_length", ["code", "==", "2"])
    assert qc._apply_value_length(node) == ("value_length", True)


def test_apply_value_length_with_ignore_nulls_ignores_blank_strings():
    df = pd.DataFrame({"code": ["ab", "  ", None]})
    qc = make_checker(df)
    node = Tree("value_length", ["code", "True", "==", "2"])
    assert qc._apply_value_length(node) == ("value_length", True)


def test_apply_percent_value_length_without_ignore_nulls():
    df = pd.DataFrame({"code": ["ab", "ab", "xx", "zz"]})
    qc = make_checker(df)
    node = Tree("percent_value_length", ["code", "50", "==", "2"])
    assert qc._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_percent_value_length_with_ignore_nulls_counts_only_matching_rows():
    df = pd.DataFrame({"code": ["ab", "  ", None, "xy"]})
    qc = make_checker(df)
    node = Tree("percent_value_length", ["code", "50", "True", "==", "2"])
    assert qc._apply_percent_value_length(node) == ("percent_value_length", True)


def test_apply_has_columns_case_sensitive_and_case_insensitive():
    df = pd.DataFrame({"A": [1], "b": [2]})
    qc = make_checker(df)

    values_tree = Tree("columns", [Tree("col", ['"A"']), Tree("col", ['"b"'])])
    node = Tree("has_columns", [values_tree])
    assert qc._apply_has_columns(node) == ("has_columns", True)

    values_tree2 = Tree("columns", [Tree("col", ['"a"']), Tree("col", ['"B"'])])
    node2 = Tree("has_columns", [values_tree2, "True"])
    assert qc._apply_has_columns(node2) == ("has_columns", True)


def test_apply_check_row_count_is_unique_not_null_is_positive():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["x", "y", "z"],
            "score": [1.5, 2.0, 3.2],
            "maybe": [1, None, 3],
        }
    )
    qc = make_checker(df)

    row_count = [Tree("row_count", [">", "2"])]
    is_unique = [Tree("is_unique", ["id"])]
    not_null = [Tree("not_null", ["name"])]
    is_positive = [Tree("is_positive", ["score"])]

    assert qc._apply_check(row_count) == ("row_count", True)
    assert qc._apply_check(is_unique) == ("is_unique", True)
    assert qc._apply_check(not_null) == ("not_null", True)
    assert qc._apply_check(is_positive) == ("is_positive", True)


def test_apply_check_unknown_check_raises_genie_exception():
    qc = make_checker(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException, match="not implemented"):
        qc._apply_check([Tree("unknown_check", ["a"])])


def test_apply_check_keyerror_returns_false_with_message():
    qc = make_checker(pd.DataFrame({"a": [1]}))
    result = qc._apply_check([Tree("is_unique", ["missing"])])
    assert result == ("Key error: is_unique for missing", False)


def test_apply_predicates_returns_results_in_reverse_order():
    qc = make_checker(pd.DataFrame({"a": [1], "b": [2]}))

    check1 = Tree("check", [Tree("row_count", [">", "0"])])
    check2 = Tree("check", [Tree("has_columns", [Tree("cols", [Tree("col", ['"a"']), Tree("col", ['"b"'])])])])
    predicate = Tree("code_block", [check1, check2])

    result = qc._apply_predicates([predicate])
    assert result == [("has_columns", True), ("row_count", True)]


def test_run_honors_ignore_column_case_and_uses_parsed_tree(monkeypatch):
    df = pd.DataFrame({"A": [1], "B": [2]})
    qc = make_checker(df)

    ast = Tree("root", [Tree("predicates", [Tree("code_block", [Tree("check", [Tree("has_columns", [Tree("cols", [Tree("col", ['"a"']), Tree("col", ['"b"'])]), "True"])])])])])

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: ast))
    result = qc.run("dummy spec", ignore_column_case=True)

    assert result == [("has_columns", True)]
    assert list(qc._obj.columns) == ["a", "b"]
    assert qc.ignore_column_case is True


def test_run_raises_for_unknown_instruction(monkeypatch):
    qc = make_checker(pd.DataFrame({"a": [1]}))
    ast = Tree("root", [Tree("something_else", [])])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: ast))

    with pytest.raises(GenieException, match="Unknown instruction"):
        qc.run("dummy spec")
