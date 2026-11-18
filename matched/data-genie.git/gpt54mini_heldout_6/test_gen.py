import pandas as pd
import pytest
from lark import Tree

from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException


def make_df():
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "x"],
            "C": ["2020-01-01", "2020-02-02", "2020-03-03"],
            "D": ["aa", "bb", "cc"],
            "E": [10, 10, 20],
        }
    )


def ctree(data, *children):
    return Tree(data, list(children))


def check_tree(inner):
    return Tree("check", [inner])


def predicate_tree(*checks):
    return Tree("code_block", list(checks))


def test_validate_rejects_empty_dataframe():
    df = pd.DataFrame({"a": [None], "b": [None]})
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        _ = df.dqc


def test_comparator_to_fn_greater_than():
    fn = QualityChecker._comparator_to_fn(">", 3)
    assert fn(4) is True
    assert fn(3) is False


def test_comparator_to_fn_equality():
    fn = QualityChecker._comparator_to_fn("==", 10)
    assert fn(10) is True
    assert fn(11) is False


def test_comparator_to_fn_unknown_returns_false():
    fn = QualityChecker._comparator_to_fn("!=", 10)
    assert fn(10) is False
    assert fn(100) is False


def test_treat_column_name_respects_ignore_case_flag():
    qc = make_df().dqc
    assert qc._treat_column_name("ABC") == "ABC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("ABC") == "abc"


def test_apply_check_row_count_passes():
    qc = make_df().dqc
    inner = ctree("row_count", ">", "2")
    result = qc._apply_check([check_tree(inner)])
    assert result == ("row_count", True)


def test_apply_check_has_columns_with_ignore_case():
    df = pd.DataFrame({"Name": [1], "Age": [2]})
    qc = df.dqc
    inner = ctree(
        "has_columns",
        ctree("column_names", Tree("quoted", ['"name"']), Tree("quoted", ['"AGE"'])),
        "True",
    )
    result = qc._apply_check([check_tree(inner)])
    assert result == ("has_columns", True)


def test_apply_check_is_unique():
    df = pd.DataFrame({"id": [1, 2, 3]})
    qc = df.dqc
    inner = ctree("is_unique", "id")
    assert qc._apply_check([check_tree(inner)]) == ("is_unique", True)


def test_apply_check_not_null_false_when_missing_values_present():
    df = pd.DataFrame({"id": [1, None, 3]})
    qc = df.dqc
    inner = ctree("not_null", "id")
    assert qc._apply_check([check_tree(inner)]) == ("not_null", False)


def test_apply_check_is_positive():
    df = pd.DataFrame({"score": [1, 2, 3]})
    qc = df.dqc
    inner = ctree("is_positive", "score")
    assert qc._apply_check([check_tree(inner)]) == ("is_positive", True)


def test_apply_check_has_one_of_case_sensitive_fails_when_case_differs():
    df = pd.DataFrame({"color": ["Red", "Blue"]})
    qc = df.dqc
    inner = ctree(
        "has_one_of",
        "color",
        ctree("allowed", Tree("quoted", ['"red"']), Tree("quoted", ['"blue"'])),
    )
    assert qc._apply_check([check_tree(inner)]) == ("has_one_of", False)


def test_apply_check_has_one_of_ignore_case_passes():
    df = pd.DataFrame({"color": ["Red", "Blue"]})
    qc = df.dqc
    inner = ctree(
        "has_one_of",
        "color",
        ctree("allowed", Tree("quoted", ['"red"']), Tree("quoted", ['"blue"'])),
        "True",
    )
    assert qc._apply_check([check_tree(inner)]) == ("has_one_of", True)


def test_apply_check_quantile_passes():
    df = pd.DataFrame({"v": [1, 2, 3, 4]})
    qc = df.dqc
    inner = ctree("quantile", "v", "0.5", ">", "2")
    assert qc._apply_check([check_tree(inner)]) == ("quantile", True)


def test_apply_check_is_date_passes_for_parseable_dates():
    qc = make_df().dqc
    inner = ctree("is_date", "C")
    assert qc._apply_check([check_tree(inner)]) == ("is_date", True)


def test_apply_check_is_date_returns_error_for_unparseable_dates():
    df = pd.DataFrame({"C": ["not-a-date", "still-not-a-date"]})
    qc = df.dqc
    inner = ctree("is_date", "C")
    msg, ok = qc._apply_check([check_tree(inner)])
    assert ok is False
    assert msg.startswith("Date parse error:")
    assert "is_date" in msg


def test_apply_check_value_length_ignores_nulls():
    df = pd.DataFrame({"txt": ["aa", "bb", None]})
    qc = df.dqc
    inner = ctree("value_length", "txt", "True", "==", "2")
    assert qc._apply_check([check_tree(inner)]) == ("value_length", True)


def test_apply_check_percent_value_length_passes():
    df = pd.DataFrame({"txt": ["aa", "bb", "c", None]})
    qc = df.dqc
    inner = ctree("percent_value_length", "txt", "50", "==", "2")
    assert qc._apply_check([check_tree(inner)]) == ("percent_value_length", True)


def test_apply_check_unknown_instruction_raises_genie_exception():
    qc = make_df().dqc
    inner = ctree("not_implemented", "A")
    with pytest.raises(GenieException, match="not_implemented seems to be not implemented"):
        qc._apply_check([check_tree(inner)])


def test_apply_check_key_error_returns_false():
    qc = make_df().dqc
    inner = ctree("is_unique", "missing_column")
    msg, ok = qc._apply_check([check_tree(inner)])
    assert ok is False
    assert msg == "Key error: is_unique for missing_column"


def test_apply_predicates_reverses_result_order():
    qc = make_df().dqc
    p = predicate_tree(
        check_tree(ctree("row_count", ">", "1")),
        check_tree(ctree("is_unique", "A")),
    )
    results = qc._apply_predicates([p])
    assert results == [("is_unique", True), ("row_count", True)]


def test_run_with_monkeypatched_parse_spec_and_ignore_column_case():
    df = pd.DataFrame({"Name": [1, 2], "Age": [3, 4]})
    qc = df.dqc

    ast = Tree(
        "start",
        [
            Tree(
                "predicates",
                [
                    predicate_tree(
                        check_tree(
                            ctree(
                                "has_columns",
                                ctree("column_names", Tree("quoted", ['"name"']), Tree("quoted", ['"age"'])),
                                "True",
                            )
                        )
                    )
                ],
            )
        ],
    )

    qc._parse_spec = staticmethod(lambda spec: ast)
    results = qc.run("ignored-spec", ignore_column_case=True)
    assert results == [("has_columns", True)]
    assert list(qc._obj.columns) == ["name", "age"]


def test_validate_spec_uses_parse_errors(monkeypatch):
    def bad_parse(_):
        raise ValueError("boom")

    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(bad_parse))
    msg, ok = QualityChecker.validate_spec("anything")
    assert ok is False
    assert "boom" in msg
