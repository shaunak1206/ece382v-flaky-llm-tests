import pandas as pd
import pytest
from lark import Tree, Token

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def make_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["ab", "cd", "ef", "gh"],
        "score": [1.0, 2.0, 3.0, 4.0],
    })


def check_of(name, children):
    return [Tree(name, children)]


def str_list(tree_name, values):
    return Tree(tree_name, [Token("STRING", '"{}"'.format(v)) for v in values])


def test_empty_dataframe_raises_attribute_error():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_all_nan_dataframe_raises_attribute_error():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_valid_dataframe_constructs_with_default_case_flag():
    qc = QualityChecker(make_df())
    assert qc.ignore_column_case is False


def test_comparator_to_fn_all_operators():
    fn = QualityChecker._comparator_to_fn
    assert fn(">", 3)(5) is True
    assert fn(">", 3)(2) is False
    assert fn("==", 3)(3) is True
    assert fn("==", 3)(4) is False
    assert fn("<", 3)(2) is True
    assert fn("<", 3)(3) is False
    assert fn("!!", 3)(3) is False


def test_treat_column_name_respects_flag():
    qc = QualityChecker(make_df())
    assert qc._treat_column_name("AbC") == "AbC"
    qc.ignore_column_case = True
    assert qc._treat_column_name("AbC") == "abc"


def test_row_count_check():
    qc = QualityChecker(make_df())
    assert qc._apply_check(check_of("row_count", [">", "3"])) == ("row_count", True)
    assert qc._apply_check(check_of("row_count", ["==", "3"])) == ("row_count", False)
    assert qc._apply_check(check_of("row_count", ["<", "10"])) == ("row_count", True)


def test_is_unique_check():
    df = pd.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]})
    qc = QualityChecker(df)
    name, ok = qc._apply_check(check_of("is_unique", ["u"]))
    assert name == "is_unique" and bool(ok) is True
    name, ok = qc._apply_check(check_of("is_unique", ["d"]))
    assert name == "is_unique" and bool(ok) is False


def test_not_null_check():
    df = pd.DataFrame({"full": [1, 2, 3], "holes": [1, None, 3]})
    qc = QualityChecker(df)
    assert bool(qc._apply_check(check_of("not_null", ["full"]))[1]) is True
    assert bool(qc._apply_check(check_of("not_null", ["holes"]))[1]) is False


def test_is_positive_check():
    df = pd.DataFrame({"pos": [1, 2, 3], "mixed": [1, 0, -3]})
    qc = QualityChecker(df)
    assert bool(qc._apply_check(check_of("is_positive", ["pos"]))[1]) is True
    assert bool(qc._apply_check(check_of("is_positive", ["mixed"]))[1]) is False


def test_missing_column_returns_key_error_message():
    qc = QualityChecker(make_df())
    msg, ok = qc._apply_check(check_of("is_unique", ["nope"]))
    assert ok is False
    assert msg.startswith("Key error")
    assert "is_unique" in msg
    assert "nope" in msg


def test_unknown_check_raises_genie_exception():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._apply_check(check_of("made_up_check", []))


def test_has_columns_case_sensitive_and_insensitive():
    qc = QualityChecker(make_df())
    ok_tree = Tree("has_columns", [str_list("columns", ["id", "name"])])
    assert qc._apply_check([ok_tree]) == ("has_columns", True)

    upper = Tree("has_columns", [str_list("columns", ["ID"])])
    assert qc._apply_check([upper]) == ("has_columns", False)

    upper_ic = Tree("has_columns", [str_list("columns", ["ID"]), Token("BOOL", "True")])
    assert qc._apply_check([upper_ic]) == ("has_columns", True)

    missing = Tree("has_columns", [str_list("columns", ["id", "zzz"])])
    assert qc._apply_check([missing]) == ("has_columns", False)


def test_has_one_of_check():
    df = pd.DataFrame({"color": ["red", "Green", "red"]})
    qc = QualityChecker(df)

    ok = Tree("has_one_of", ["color", str_list("values", ["red", "Green"])])
    assert qc._apply_check([ok]) == ("has_one_of", True)

    bad = Tree("has_one_of", ["color", str_list("values", ["red", "blue"])])
    assert qc._apply_check([bad]) == ("has_one_of", False)

    case_sensitive = Tree("has_one_of", ["color", str_list("values", ["red", "green"]),
                                         Token("BOOL", "False")])
    assert qc._apply_check([case_sensitive]) == ("has_one_of", False)

    ignore_case = Tree("has_one_of", ["color", str_list("values", ["red", "green"]),
                                      Token("BOOL", "True")])
    assert qc._apply_check([ignore_case]) == ("has_one_of", True)


def test_quantile_check():
    qc = QualityChecker(make_df())
    # median of 1..4 is 2.5
    assert qc._apply_check(check_of("quantile", ["score", "0.5", ">", "2"])) == ("quantile", True)
    assert qc._apply_check(check_of("quantile", ["score", "0.5", "<", "2"])) == ("quantile", False)
    assert qc._apply_check(check_of("quantile", ["score", "0.5", "==", "2.5"])) == ("quantile", True)


def test_is_date_check_valid_and_invalid():
    df = pd.DataFrame({"good": ["2020-01-01", "2021-05-06"], "bad": ["not a date", "also not"]})
    qc = QualityChecker(df)
    assert qc._apply_check(check_of("is_date", ["good"])) == ("is_date", True)
    msg, ok = qc._apply_check(check_of("is_date", ["bad"]))
    assert ok is False
    assert msg.startswith("Date parse error")


def test_value_length_check():
    df = pd.DataFrame({"c": ["abc", "def", ""]})
    qc = QualityChecker(df)

    # without ignoring nulls: lengths {3, 0} -> not uniform
    assert qc._apply_check(check_of("value_length", ["c", "==", "3"])) == ("value_length", False)

    # ignoring nulls (blank strings): only length 3 remains
    assert qc._apply_check(
        check_of("value_length", ["c", Token("BOOL", "True"), "==", "3"])) == ("value_length", True)
    assert qc._apply_check(
        check_of("value_length", ["c", Token("BOOL", "True"), "<", "3"])) == ("value_length", False)

    uniform = QualityChecker(pd.DataFrame({"c": ["ab", "cd"]}))
    assert uniform._apply_check(check_of("value_length", ["c", ">", "1"])) == ("value_length", True)


def test_percent_value_length_check():
    df = pd.DataFrame({"c": ["ab", "cd", "e", "fg"]})
    qc = QualityChecker(df)
    assert qc._apply_check(
        check_of("percent_value_length", ["c", "75", "%", "2"])) == ("percent_value_length", True)

    qc = QualityChecker(pd.DataFrame({"c": ["ab", "cd", "e", "fg"]}))
    assert qc._apply_check(
        check_of("percent_value_length", ["c", "100", "%", "2"])) == ("percent_value_length", False)


def test_percent_value_length_ignore_nulls():
    df = pd.DataFrame({"c": ["ab", "", "cd", "e"]})
    qc = QualityChecker(df)
    node = check_of("percent_value_length", ["c", "50", Token("BOOL", "True"), "%", "2"])
    assert qc._apply_check(node) == ("percent_value_length", True)


def test_run_internal_rejects_bad_tree_shapes():
    qc = QualityChecker(make_df())
    with pytest.raises(GenieException):
        qc._run(Tree("start", []))
    with pytest.raises(GenieException):
        qc._run(Tree("start", [Tree("other", [])]))


def test_run_internal_returns_results_in_reverse_order():
    qc = QualityChecker(make_df())
    block = Tree("code_block", [
        Tree("check", [Tree("row_count", ["==", "4"])]),
        Tree("check", [Tree("is_unique", ["id"])]),
    ])
    tree = Tree("start", [Tree("predicates", [block])])
    results = qc._run(tree)
    assert [r[0] for r in results] == ["is_unique", "row_count"]
    assert bool(results[0][1]) is True
    assert bool(results[1][1]) is True


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    df = pd.DataFrame({"ID": [1, 2], "Name": ["a", "b"]})
    qc = QualityChecker(df)
    tree = Tree("start", [Tree("predicates", [
        Tree("code_block", [Tree("check", [Tree("is_unique", ["ID"])])])
    ])])
    monkeypatch.setattr(QualityChecker, "_parse_spec", staticmethod(lambda spec: tree))
    results = qc.run("ignored", ignore_column_case=True)
    assert qc.ignore_column_case is True
    assert list(qc._obj.columns) == ["id", "name"]
    assert results[0][0] == "is_unique"
    assert bool(results[0][1]) is True


def test_validate_spec_rejects_garbage():
    msg, ok = QualityChecker.validate_spec("@@@ this is definitely !!! not a valid spec ###")
    assert ok is False
    assert isinstance(msg, str) and msg
