import pandas as pd
import pytest
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_qc(df):
    return QualityChecker(df.copy())


def wrap_check(node):
    return Tree("check", [node])


def code_block(*checks):
    return Tree("code_block", list(checks))


def predicates(*blocks):
    return Tree("predicates", list(blocks))


def program(*blocks):
    return Tree("program", [predicates(*blocks)])


def test_validate_raises_on_all_empty_dataframe():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError, match="DataFrame is empty!!!"):
        QualityChecker(df)


def test_comparator_to_fn_gt_eq_lt_and_default():
    gt = QualityChecker._comparator_to_fn(">", 3)
    eq = QualityChecker._comparator_to_fn("==", 3)
    lt = QualityChecker._comparator_to_fn("<", 3)
    other = QualityChecker._comparator_to_fn("!=", 3)

    assert gt(4) is True
    assert gt(3) is False
    assert eq(3) is True
    assert eq(2) is False
    assert lt(2) is True
    assert lt(3) is False
    assert other(999) is False


def test_treat_column_name_respects_ignore_column_case_flag():
    qc = make_qc(pd.DataFrame({"A": [1]}))
    assert qc._treat_column_name("A") == "A"
    qc.ignore_column_case = True
    assert qc._treat_column_name("A") == "a"


def test_apply_check_row_count():
    qc = make_qc(pd.DataFrame({"a": [1, 2, 3]}))
    node = Tree("row_count", [">", "2"])
    assert qc._apply_check([wrap_check(node)]) == ("row_count", True)


def test_apply_check_has_columns_case_sensitive_and_case_insensitive():
    df = pd.DataFrame({"Foo": [1], "Bar": [2]})
    qc = make_qc(df)

    node_sensitive = Tree(
        "has_columns",
        [Tree("columns", [Tree("col", ['"Foo"']), Tree("col", ['"Bar"'])])],
    )
    assert qc._apply_check([wrap_check(node_sensitive)]) == ("has_columns", True)

    node_insensitive = Tree(
        "has_columns",
        [Tree("columns", [Tree("col", ['"foo"']), Tree("col", ['"bar"'])]), "True"],
    )
    assert qc._apply_check([wrap_check(node_insensitive)]) == ("has_columns", True)


def test_apply_check_is_unique_not_null_and_positive():
    df = pd.DataFrame({"u": [1, 2, 3], "n": [1, None, 3], "p": [1, 2, 3]})
    qc = make_qc(df)

    assert qc._apply_check([wrap_check(Tree("is_unique", ["u"]))]) == ("is_unique", True)
    assert qc._apply_check([wrap_check(Tree("not_null", ["n"]))]) == ("not_null", False)
    assert qc._apply_check([wrap_check(Tree("is_positive", ["p"]))]) == ("is_positive", True)


def test_apply_check_has_one_of_ignore_case_true():
    df = pd.DataFrame({"color": ["Red", "blue", "RED"]})
    qc = make_qc(df)

    node = Tree(
        "has_one_of",
        [
            "color",
            Tree("allowed", [Tree("val", ['"red"']), Tree("val", ['"blue"'])]),
            "True",
        ],
    )
    assert qc._apply_check([wrap_check(node)]) == ("has_one_of", True)


def test_apply_check_quantile():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    qc = make_qc(df)

    node = Tree("quantile", ["x", "0.5", ">", "2"])
    assert qc._apply_check([wrap_check(node)]) == ("quantile", True)


def test_apply_check_is_date_success_and_failure_message():
    df = pd.DataFrame({"good": ["2023-01-01", "2024-02-02"], "bad": ["2023-01-01", "not-a-date"]})
    qc = make_qc(df)

    ok = qc._apply_check([wrap_check(Tree("is_date", ["good"]))])
    bad = qc._apply_check([wrap_check(Tree("is_date", ["bad"]))])

    assert ok == ("is_date", True)
    assert bad[0] == "Date parse error: is_date for bad"
    assert bad[1] is False


def test_apply_check_value_length_with_ignore_nulls_false_and_true():
    df = pd.DataFrame({"txt": ["ab", "cd", None]})
    qc = make_qc(df)

    node_false = Tree("value_length", ["txt", "==", "2"])
    assert qc._apply_check([wrap_check(node_false)]) == ("value_length", False)

    node_true = Tree("value_length", ["txt", "True", "==", "2"])
    assert qc._apply_check([wrap_check(node_true)]) == ("value_length", True)


def test_apply_check_percent_value_length_with_and_without_ignore_nulls():
    df = pd.DataFrame({"txt": ["aa", "bb", "ccc", None]})
    qc = make_qc(df)

    node_no_ignore = Tree("percent_value_length", ["txt", "50", "==", "2"])
    assert qc._apply_check([wrap_check(node_no_ignore)]) == ("percent_value_length", True)

    node_ignore = Tree("percent_value_length", ["txt", "50", "True", "==", "2"])
    assert qc._apply_check([wrap_check(node_ignore)]) == ("percent_value_length", False)


def test_apply_check_unknown_check_raises_genie_exception():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException, match="not implemented in the DSL"):
        qc._apply_check([wrap_check(Tree("mystery_check", ["a"]))])


def test_apply_check_keyerror_returns_false_tuple():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    result = qc._apply_check([wrap_check(Tree("not_null", ["missing"]))])
    assert result == ("Key error: not_null for missing", False)


def test_apply_predicates_reverses_results_order():
    qc = make_qc(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    block = code_block(
        wrap_check(Tree("row_count", [">", "1"])),
        wrap_check(Tree("is_unique", ["a"])),
    )
    assert qc._apply_predicates([block]) == [("is_unique", True), ("row_count", True)]


def test_run_executes_predicates_tree():
    qc = make_qc(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    ast = program(
        code_block(
            wrap_check(Tree("row_count", [">", "1"])),
            wrap_check(Tree("not_null", ["a"])),
        )
    )
    assert qc._run(ast) == [("not_null", True), ("row_count", True)]


def test_run_raises_on_invalid_top_level_tree_shape():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException, match="How is this possible"):
        qc._run(Tree("program", []))


def test_run_raises_on_unknown_instruction():
    qc = make_qc(pd.DataFrame({"a": [1]}))
    with pytest.raises(GenieException, match="Unknown instruction"):
        qc._run(Tree("program", [Tree("something_else", [])]))
