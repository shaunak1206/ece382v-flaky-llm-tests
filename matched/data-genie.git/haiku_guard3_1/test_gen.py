import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _check(name, *children):
    """Build a check node the way the parser would produce it."""
    return Tree(name, list(children))


def _apply(qc, node):
    """_apply_check receives the list of child nodes of a check wrapper."""
    return qc._apply_check([node])


def _values(*items):
    """Build a 'values' node whose children expose .value like lark tokens."""
    return Tree("values", [Token("STRING", '"{}"'.format(i)) for i in items])


def _columns(*items):
    """Build a 'columns' node whose children expose .value like lark tokens."""
    return Tree("columns", [Token("STRING", '"{}"'.format(i)) for i in items])


def test_comparator_greater_than():
    fn = QualityChecker._comparator_to_fn(">", 2)
    assert bool(fn(3)) is True
    assert bool(fn(2)) is False
    assert bool(fn(1)) is False


def test_comparator_equal_and_less_than():
    eq = QualityChecker._comparator_to_fn("==", 5)
    lt = QualityChecker._comparator_to_fn("<", 5)
    assert bool(eq(5)) is True
    assert bool(eq(4)) is False
    assert bool(lt(4)) is True
    assert bool(lt(5)) is False


def test_comparator_unknown_always_false():
    fn = QualityChecker._comparator_to_fn("!=", 1)
    assert fn(1) is False
    assert fn(2) is False
    assert fn(-100) is False


def test_init_rejects_all_null_frame():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_init_keeps_frame_and_default_case_flag():
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)
    assert qc._obj is df
    assert qc.ignore_column_case is False


def test_treat_column_name_respects_flag():
    qc = QualityChecker(pd.DataFrame({"Name": [1]}))
    assert qc._treat_column_name("Name") == "Name"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Name") == "name"


def test_row_count_check():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    name, ok = _apply(qc, _check("row_count", ">", "2"))
    assert name == "row_count"
    assert bool(ok) is True

    name, ok = _apply(qc, _check("row_count", "==", "4"))
    assert name == "row_count"
    assert bool(ok) is False


def test_is_unique_check():
    unique_qc = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    dup_qc = QualityChecker(pd.DataFrame({"a": [1, 2, 2]}))
    assert bool(_apply(unique_qc, _check("is_unique", "a"))[1]) is True
    assert bool(_apply(dup_qc, _check("is_unique", "a"))[1]) is False


def test_not_null_check():
    clean = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    holey = QualityChecker(pd.DataFrame({"a": [1, None]}))
    assert bool(_apply(clean, _check("not_null", "a"))[1]) is True
    assert bool(_apply(holey, _check("not_null", "a"))[1]) is False


def test_is_positive_check():
    good = QualityChecker(pd.DataFrame({"a": [1, 2, 3]}))
    bad = QualityChecker(pd.DataFrame({"a": [0, 1, 2]}))
    assert bool(_apply(good, _check("is_positive", "a"))[1]) is True
    assert bool(_apply(bad, _check("is_positive", "a"))[1]) is False


def test_has_one_of_case_sensitive_and_insensitive():
    df = pd.DataFrame({"c": ["a", "b", "B"]})
    qc = QualityChecker(df)

    strict = _check("has_one_of", "c", _values("a", "b"))
    name, ok = _apply(qc, strict)
    assert name == "has_one_of"
    assert bool(ok) is False

    relaxed = _check("has_one_of", "c", _values("a", "b"), "True")
    name, ok = _apply(qc, relaxed)
    assert name == "has_one_of"
    assert bool(ok) is True


def test_quantile_check():
    qc = QualityChecker(pd.DataFrame({"x": [1, 2, 3, 4, 5]}))
    # median of 1..5 is 3.0
    name, ok = _apply(qc, _check("quantile", "x", "0.5", ">", "2"))
    assert name == "quantile"
    assert bool(ok) is True

    name, ok = _apply(qc, _check("quantile", "x", "0.5", "<", "2"))
    assert name == "quantile"
    assert bool(ok) is False


def test_value_length_check():
    uniform = QualityChecker(pd.DataFrame({"s": ["abc", "def"]}))
    mixed = QualityChecker(pd.DataFrame({"s": ["abc", "de"]}))

    name, ok = _apply(uniform, _check("value_length", "s", "==", "3"))
    assert name == "value_length"
    assert bool(ok) is True

    name, ok = _apply(mixed, _check("value_length", "s", "==", "3"))
    assert name == "value_length"
    assert bool(ok) is False


def _percent_frame():
    # lengths are [3, 2, 3, 2]: exactly half have length 3
    return pd.DataFrame({"s": ["abc", "de", "fgh", "ij"]})


def test_percent_value_length_check():
    qc = QualityChecker(_percent_frame())
    name, ok = _apply(qc, _check("percent_value_length", "s", "50", "==", "3"))
    assert name == "percent_value_length"
    assert bool(ok) is True

    qc = QualityChecker(_percent_frame())
    name, ok = _apply(qc, _check("percent_value_length", "s", "51", "==", "3"))
    assert name == "percent_value_length"
    assert bool(ok) is False


def test_has_columns_case_sensitive():
    qc = QualityChecker(pd.DataFrame({"A": [1], "B": [2]}))
    name, ok = _apply(qc, _check("has_columns", _columns("A")))
    assert name == "has_columns"
    assert bool(ok) is True

    name, ok = _apply(qc, _check("has_columns", _columns("A", "C")))
    assert name == "has_columns"
    assert bool(ok) is False

    # Case mismatch must fail when case-insensitivity is not requested.
    name, ok = _apply(qc, _check("has_columns", _columns("a")))
    assert bool(ok) is False


def test_has_columns_ignore_case():
    qc = QualityChecker(pd.DataFrame({"A": [1], "B": [2]}))
    name, ok = _apply(qc, _check("has_columns", _columns("a", "b"), "True"))
    assert name == "has_columns"
    assert bool(ok) is True


def test_is_date_valid_and_invalid():
    good = QualityChecker(pd.DataFrame({"d": ["2020-01-01", "2021-06-30"]}))
    name, ok = _apply(good, _check("is_date", "d"))
    assert name == "is_date"
    assert bool(ok) is True

    bad = QualityChecker(pd.DataFrame({"d": ["not a date"]}))
    message, ok = _apply(bad, _check("is_date", "d"))
    assert bool(ok) is False
    assert message.startswith("Date parse error:")
    assert "is_date" in message and "d" in message


def test_missing_column_returns_key_error_result():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    message, ok = _apply(qc, _check("is_unique", "missing"))
    assert bool(ok) is False
    assert message.startswith("Key error:")
    assert "missing" in message


def test_unimplemented_check_raises():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(GenieException):
        _apply(qc, _check("no_such_check", "a"))


def test_run_rejects_malformed_tree():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2]}))
    # Root must have exactly one child.
    with pytest.raises(GenieException):
        qc._run(Tree("root", [1, 2]))
    # Unknown top-level instruction must be rejected.
    with pytest.raises(GenieException):
        qc._run(Tree("root", [Tree("other", [])]))


def test_run_predicates_reverses_check_order():
    qc = QualityChecker(pd.DataFrame({"a": [1, 2, 2]}))
    tree = Tree("root", [
        Tree("predicates", [
            Tree("code_block", [
                Tree("check", [_check("row_count", ">", "0")]),
                Tree("check", [_check("is_unique", "a")]),
            ]),
        ]),
    ])
    results = qc._run(tree)
    # _apply_predicates inserts each result at position 0, so the list
    # ends up in reverse order of the checks.
    assert [name for name, _ in results] == ["is_unique", "row_count"]
    assert bool(results[0][1]) is False
    assert bool(results[1][1]) is True


def test_ignore_case_applies_to_column_lookup():
    df = pd.DataFrame({"name": [1, 2, 3]})
    qc = QualityChecker(df)
    qc.ignore_column_case = True
    # "NAME" is lowercased to "name" before lookup, so this must succeed.
    name, ok = _apply(qc, _check("is_unique", "NAME"))
    assert name == "is_unique"
    assert bool(ok) is True


def test_validate_spec_rejects_garbage():
    result = QualityChecker.validate_spec("this is definitely not a valid spec !!! @@@")
    message, ok = result
    assert ok is False
    assert isinstance(message, str)
    assert message != ""
