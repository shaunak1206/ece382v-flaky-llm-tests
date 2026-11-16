import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["aaa", "bbb", "ccc", "ddd"],
        "color": ["red", "green", "red", "blue"],
    })


def tok(value):
    return Token("STRING", value)


def check(node):
    """Run a single check node through _apply_check (expects list of nodes)."""
    return [node]


def test_validate_rejects_empty_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_validate_rejects_all_null_columns():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_init_defaults():
    qc = QualityChecker(make_df())
    assert qc.ignore_column_case is False


@pytest.mark.parametrize("comparator,lhs,rhs,expected", [
    (">", 5, 3, True),
    (">", 3, 5, False),
    ("==", 3, 3, True),
    ("==", 3, 4, False),
    ("<", 1, 2, True),
    ("<", 2, 1, False),
    ("!=", 1, 2, False),
])
def test_comparator_to_fn(comparator, lhs, rhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert fn(lhs) == expected


def test_treat_column_name_respects_flag():
    qc = QualityChecker(make_df())
    assert qc._treat_column_name("MyCol") == "MyCol"
    qc.ignore_column_case = True
    assert qc._treat_column_name("MyCol") == "mycol"


def test_row_count_check():
    qc = QualityChecker(make_df())
    name, ok = qc._apply_check(check(Tree("row_count", [">", "3"])))
    assert name == "row_count"
    assert bool(ok) is True
    name, ok = qc._apply_check(check(Tree("row_count", ["<", "3"])))
    assert bool(ok) is False


def test_is_unique_check():
    df = make_df()
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("is_unique", ["id"]))) == ("is_unique", True)
    assert qc._apply_check(check(Tree("is_unique", ["color"]))) == ("is_unique", False)


def test_not_null_check():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]})
    qc = QualityChecker(df)
    name, ok = qc._apply_check(check(Tree("not_null", ["a"])))
    assert name == "not_null" and bool(ok) is True
    name, ok = qc._apply_check(check(Tree("not_null", ["b"])))
    assert bool(ok) is False


def test_is_positive_check():
    df = pd.DataFrame({"pos": [1, 2, 3], "neg": [1, -2, 3]})
    qc = QualityChecker(df)
    assert bool(qc._apply_check(check(Tree("is_positive", ["pos"])))[1]) is True
    assert bool(qc._apply_check(check(Tree("is_positive", ["neg"])))[1]) is False


def test_missing_column_returns_key_error_message():
    qc = QualityChecker(make_df())
    msg, ok = qc._apply_check(check(Tree("not_null", ["missing"])))
    assert ok is False
    assert msg == "Key error: not_null for missing"


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._apply_check(check(Tree("bogus_check", ["id"])))


def test_has_columns_check():
    qc = QualityChecker(make_df())
    cols = Tree("columns", [tok('"id"'), tok('"name"')])
    assert qc._apply_check(check(Tree("has_columns", [cols]))) == ("has_columns", True)
    cols = Tree("columns", [tok('"id"'), tok('"nope"')])
    assert qc._apply_check(check(Tree("has_columns", [cols]))) == ("has_columns", False)


def test_has_columns_ignore_case():
    qc = QualityChecker(make_df())
    cols = Tree("columns", [tok('"ID"'), tok('"Name"')])
    assert qc._apply_check(check(Tree("has_columns", [cols])))[1] is False
    assert qc._apply_check(check(Tree("has_columns", [cols, "True"])))[1] is True


def test_has_one_of_check():
    qc = QualityChecker(make_df())
    allowed = Tree("values", [tok('"red"'), tok('"green"'), tok('"blue"')])
    assert qc._apply_check(check(Tree("has_one_of", ["color", allowed]))) == ("has_one_of", True)
    allowed = Tree("values", [tok('"red"'), tok('"green"')])
    assert qc._apply_check(check(Tree("has_one_of", ["color", allowed])))[1] is False


def test_has_one_of_ignore_case():
    df = pd.DataFrame({"color": ["Red", "GREEN"]})
    qc = QualityChecker(df)
    allowed = Tree("values", [tok('"red"'), tok('"green"')])
    assert qc._apply_check(check(Tree("has_one_of", ["color", allowed])))[1] is False
    assert qc._apply_check(check(Tree("has_one_of", ["color", allowed, "True"])))[1] is True


def test_quantile_check():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    name, ok = qc._apply_check(check(Tree("quantile", ["v", "0.5", "==", "3"])))
    assert name == "quantile" and bool(ok) is True
    name, ok = qc._apply_check(check(Tree("quantile", ["v", "0.5", ">", "4"])))
    assert bool(ok) is False


def test_is_date_check():
    good = QualityChecker(pd.DataFrame({"d": ["2020-01-01", "2021-05-06"]}))
    assert good._apply_check(check(Tree("is_date", ["d"]))) == ("is_date", True)

    bad = QualityChecker(pd.DataFrame({"d": ["not a date", "also bad"]}))
    msg, ok = bad._apply_check(check(Tree("is_date", ["d"])))
    assert ok is False
    assert msg == "Date parse error: is_date for d"


def test_value_length_check():
    qc = QualityChecker(make_df())
    # all names have length 3
    assert qc._apply_check(check(Tree("value_length", ["name", "==", "3"]))) == ("value_length", True)
    assert qc._apply_check(check(Tree("value_length", ["name", "==", "4"])))[1] is False
    # with explicit ignore_nulls flag
    assert qc._apply_check(check(Tree("value_length", ["name", "False", "==", "3"])))[1] is True


def test_value_length_mixed_lengths_fails_and_ignore_nulls_helps():
    df = pd.DataFrame({"s": ["abc", "def", None, " "]})
    qc = QualityChecker(df)
    assert qc._apply_check(check(Tree("value_length", ["s", "==", "3"])))[1] is False
    assert qc._apply_check(check(Tree("value_length", ["s", "True", "==", "3"])))[1] is True


def test_percent_value_length_check():
    df = pd.DataFrame({"s": ["abc", "def", "ghij", "klm"]})
    qc = QualityChecker(df)
    # 75% of values have length 3
    assert qc._apply_check(check(Tree("percent_value_length", ["s", "75", "==", "3"])))[1] is True
    assert qc._apply_check(check(Tree("percent_value_length", ["s", "80", "==", "3"])))[1] is False


def test_run_predicates_results_in_reverse_order():
    qc = QualityChecker(make_df())
    block = Tree("code_block", [
        Tree("check", [Tree("row_count", ["==", "4"])]),
        Tree("check", [Tree("is_unique", ["color"])]),
    ])
    tree = Tree("start", [Tree("predicates", [block])])
    results = qc._run(tree)
    assert [r[0] for r in results] == ["is_unique", "row_count"]
    assert bool(results[0][1]) is False
    assert bool(results[1][1]) is True


def test_run_rejects_bad_trees():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("something_else", [])]))


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"ID": [1, 2, 3], "Name": ["a", "b", "c"]})
    qc = QualityChecker(df)
    tree = Tree("start", [Tree("predicates", [
        Tree("code_block", [Tree("check", [Tree("is_unique", ["ID"])])])
    ])])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    results = qc.run("ignored", ignore_column_case=True)
    assert list(df.columns) == ["id", "name"]
    assert results == [("is_unique", True)]


def test_validate_spec_rejects_garbage():
    result = QualityChecker.validate_spec("@@@ this is definitely !!! not a spec ###")
    assert result[1] is False
    assert isinstance(result[0], str)
