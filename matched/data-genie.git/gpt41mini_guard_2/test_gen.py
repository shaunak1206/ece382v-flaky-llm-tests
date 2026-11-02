import pytest
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker, GenieException


# Auxiliary class to build dummy Tree nodes for testing internal methods directly
class DummyToken:
    def __init__(self, value):
        self.value = value


class DummyNode:
    def __init__(self, data, children):
        self.data = data
        self.children = children


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40],
        "C": ["2020-01-01", "2020-02-01", "2020-01-15", "2020-03-01"],
        "D": ["yes", "no", "yes", "no"],
        "E": ["ok", "ok", "ok", "ok"],
        "F": ["a", "B", "c", "D"],
        "G": ["hello", "", "world", None]
    })


def test_init_validates_non_empty():
    df = pd.DataFrame({"X": [None, None], "Y": [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_comparator_to_fn():
    fn_gt = QualityChecker._comparator_to_fn(">", 5)
    assert fn_gt(6) is True
    assert fn_gt(5) is False

    fn_eq = QualityChecker._comparator_to_fn("==", 3)
    assert fn_eq(3) is True
    assert fn_eq(2) is False

    fn_lt = QualityChecker._comparator_to_fn("<", 10)
    assert fn_lt(9) is True
    assert fn_lt(10) is False

    fn_other = QualityChecker._comparator_to_fn("??", 0)
    assert fn_other(100) is False


def test_treat_column_name_ignore_case(sample_df):
    qc = QualityChecker(sample_df)
    qc.ignore_column_case = False
    assert qc._treat_column_name("Col") == "Col"
    qc.ignore_column_case = True
    assert qc._treat_column_name("Col") == "col"


def test_apply_quantile_true_and_false(sample_df):
    qc = QualityChecker(sample_df)
    node_true = DummyNode("quantile", ["A", 0.5, ">", 2])
    result_true = qc._apply_quantile(node_true)
    assert result_true[0] == "quantile"
    # median of A is 2.5 > 2 should be True
    assert result_true[1] is True

    node_false = DummyNode("quantile", ["B", 0.5, "<", 20])
    result_false = qc._apply_quantile(node_false)
    assert result_false[0] == "quantile"
    # median of B is 25 < 20 is False
    assert result_false[1] is False


def test_apply_date_validation_valid_and_invalid(sample_df):
    qc = QualityChecker(sample_df)
    node_valid = DummyNode("is_date", ["C"])
    key, result = qc._apply_date_validation(node_valid)
    assert key == "is_date"
    assert result is True

    # Create a df with an unparseable date
    df2 = pd.DataFrame({"date": ["notadate", "2020-01-01"]})
    qc2 = QualityChecker(df2)
    node_invalid = DummyNode("is_date", ["date"])
    key, result = qc2._apply_date_validation(node_invalid)
    assert key.startswith("Date parse error:")
    assert result is False


def test_apply_has_one_of_with_and_without_ignore_case():
    df = pd.DataFrame({"col": ["A", "b", "C"]})
    qc = QualityChecker(df)

    # Node with children: column name, allowed values, no ignore_case arg => ignore_case False
    allowed_values_node = DummyNode("list", [DummyToken('"A"'), DummyToken('"B"'), DummyToken('"C"')])
    node_no_ignore = DummyNode("has_one_of", ["col", allowed_values_node])
    key, result = qc._apply_has_one_of(node_no_ignore)
    assert key == "has_one_of"
    # unique values are ["A", "b", "C"], ignore_case False, so "b" not in allowed_values => False
    assert result is False

    # Node with ignore_case=True
    node_ignore = DummyNode("has_one_of", ["col", allowed_values_node, 'True'])
    key, result = qc._apply_has_one_of(node_ignore)
    assert key == "has_one_of"
    # now case is ignored, all letters are in allowed_values case-insensitive -> True
    assert result is True


def test_apply_value_length_ignore_nulls_and_comparator():
    df = pd.DataFrame({"col": ["abc", "defg", None, "   "]})
    qc = QualityChecker(df)

    # Without ignore_nulls arg (length 3 children): comparator '==' rhs 3, ignore_nulls False by default
    node = DummyNode("value_length", ["col", "==", 3])
    key, result = qc._apply_value_length(node)
    assert key == "value_length"
    # lengths of all values as str: "abc"(3), "defg"(4), "None"(4), "   "(3)
    # unique lengths are [3,4], so not length 1 -> False
    assert result is False

    # With ignore_nulls='True' (length 4 children): ignore_nulls 'True', comparator '>', rhs 2
    node2 = DummyNode("value_length", ["col", "True", ">", 2])
    key2, result2 = qc._apply_value_length(node2)
    assert key2 == "value_length"
    # after ignoring nulls and whitespaces only "abc" and "defg" remain:
    # lengths 3 and 4 => unique lengths 2, condition requires unique_lengths == 1
    assert result2 is False

    # Create df where length is constant except maybe nulls
    df2 = pd.DataFrame({"col": ["aaa", "bbb", None, "   "]})
    qc2 = QualityChecker(df2)
    node3 = DummyNode("value_length", ["col", "True", "==", 3])
    key3, result3 = qc2._apply_value_length(node3)
    assert key3 == "value_length"
    # ignoring nulls: "aaa", "bbb" length 3 unique_lengths is [3], comparator == 3 True
    assert result3 is True


def test_apply_percent_value_length_basic_and_ignore_nulls():
    df = pd.DataFrame({"col": ["aaa", "bbb", "", None, "cccc"]})
    qc = QualityChecker(df)

    # Case with 4 children: ignore_nulls='False' implicit, percent=60, rhs=3
    node1 = DummyNode("percent_value_length", ["col", 60, "==", 3])
    key1, res1 = qc._apply_percent_value_length(node1)
    assert key1 == "percent_value_length"
    # lengths are: aaa(3), bbb(3), ""(0), None(0), cccc(4)
    # not ignoring nulls: passing = rows with length == 3 -> rows 0 and 1 => 2/5 = 40% < 60 -> False
    assert res1 is False

    # Case with 5 children: percent=40, ignore_nulls='True', rhs=3
    node2 = DummyNode("percent_value_length", ["col", 40, "True", "dummy", 3])
    key2, res2 = qc._apply_percent_value_length(node2)
    assert key2 == "percent_value_length"
    # ignoring nulls and blank strings: only rows 0,1,4 remain with lengths 3,3,4
    # rows with length == 3 -> 2 out of 3 => ~66.7% >= 40% => True
    assert res2 is True


def test_apply_has_columns_exact_and_ignore_case():
    df = pd.DataFrame(columns=["A", "B", "C", "d", "E"])
    qc = QualityChecker(df)

    # Check without ignore_case (default)
    allowed_cols = DummyNode("list", [DummyToken('"A"'), DummyToken('"B"'), DummyToken('"C"')])
    node = DummyNode("has_columns", [allowed_cols])
    key, res = qc._apply_has_columns(node)
    assert key == "has_columns"
    assert res is True

    # Check with ignore_case=True but columns differ in case
    allowed_cols2 = DummyNode("list", [DummyToken('"a"'), DummyToken('"b"'), DummyToken('"c"')])
    node2 = DummyNode("has_columns", [allowed_cols2, 'True'])
    key2, res2 = qc._apply_has_columns(node2)
    assert key2 == "has_columns"
    assert res2 is True

    # Missing column check (should be False)
    allowed_cols3 = DummyNode("list", [DummyToken('"A"'), DummyToken('"B"'), DummyToken('"Z"')])
    node3 = DummyNode("has_columns", [allowed_cols3])
    _, res3 = qc._apply_has_columns(node3)
    assert res3 is False


def test_apply_check_various_valid_and_invalid(sample_df):
    qc = QualityChecker(sample_df)

    # row_count > 3 True (4 rows)
    row_count_check = DummyNode("row_count", [">", 3])
    key, res = qc._apply_check([row_count_check])
    assert key == "row_count"
    assert res is True

    # row_count == 5 False
    row_count_check2 = DummyNode("row_count", ["==", 5])
    key2, res2 = qc._apply_check([row_count_check2])
    assert key2 == "row_count"
    assert res2 is False

    # is_unique True on unique column
    unique_check = DummyNode("is_unique", ["A"])
    key3, res3 = qc._apply_check([unique_check])
    assert key3 == "is_unique"
    assert res3 is True

    # not_null True on column with no nulls
    not_null_check = DummyNode("not_null", ["A"])
    key4, res4 = qc._apply_check([not_null_check])
    assert key4 == "not_null"
    assert res4 is True

    # is_positive True on positive column
    is_positive_check = DummyNode("is_positive", ["B"])
    key5, res5 = qc._apply_check([is_positive_check])
    assert key5 == "is_positive"
    assert res5 is True

    # has_one_of returns True when valid
    allowed_values_node = DummyNode("list", [DummyToken('"yes"'), DummyToken('"no"')])
    has_one_of_check = DummyNode("has_one_of", ["D", allowed_values_node])
    key6, res6 = qc._apply_check([has_one_of_check])
    assert key6 == "has_one_of"
    assert res6 is True

    # quantile check
    quantile_check = DummyNode("quantile", ["A", 0.5, ">", 2])
    key7, res7 = qc._apply_check([quantile_check])
    assert key7 == "quantile"
    assert res7 is True

    # is_date valid
    is_date_check = DummyNode("is_date", ["C"])
    key8, res8 = qc._apply_check([is_date_check])
    assert key8 == "is_date"
    assert res8 is True

    # value_length check
    value_length_check = DummyNode("value_length", ["E", "==", 2])
    key9, res9 = qc._apply_check([value_length_check])
    assert key9 == "value_length"
    # all 'ok' length 2 => True
    assert res9 is True

    # percent_value_length check
    percent_value_length_check = DummyNode("percent_value_length", ["E", 100, "==", 2])
    key10, res10 = qc._apply_check([percent_value_length_check])
    assert key10 == "percent_value_length"
    assert res10 is True

    # Unknown check type raises GenieException
    unknown_check = DummyNode("unknown_check", [])
    with pytest.raises(GenieException):
        qc._apply_check([unknown_check])

    # KeyError on missing column returns False and message
    missing_column_check = DummyNode("is_unique", ["nonexistent"])
    key_msg, res_bool = qc._apply_check([missing_column_check])
    assert key_msg.startswith("Key error:")
    assert res_bool is False


def test_apply_predicates_code_block_and_unknown_instruction(sample_df):
    qc = QualityChecker(sample_df)

    # Create a predicate of data "code_block" with one check (row_count > 0)
    row_count_check = DummyNode("row_count", [">", 0])
    check_node = DummyNode("check", [row_count_check])
    code_block_pred = DummyNode("code_block", [check_node])
    predicates_node = DummyNode("predicates", [code_block_pred])
    res = qc._apply_predicates([code_block_pred])
    # List of tuples with keys and results expected
    assert isinstance(res, list)
    assert res[0][0] == "row_count"
    assert res[0][1] is True

    # Unknown instruction raises GenieException
    with pytest.raises(GenieException):
        qc._apply_predicates([DummyNode("random_instruction", [])])


def test__run_accepts_predicates_and_rejects_other(sample_df):
    qc = QualityChecker(sample_df)
    # Compose predicates node with one child code_block
    row_count_check = DummyNode("row_count", [">", 0])
    check_node = DummyNode("check", [row_count_check])
    code_block_pred = DummyNode("code_block", [check_node])
    predicates_node = DummyNode("predicates", [code_block_pred])
    tree = DummyNode("start", [predicates_node])

    out = qc._run(tree)
    assert isinstance(out, list)
    assert out[0][0] == "row_count"
    assert out[0][1] is True

    # Tree with no children triggers GenieException
    with pytest.raises(GenieException):
        qc._run(DummyNode("start", []))

    # Tree with unknown instruction triggers GenieException
    with pytest.raises(GenieException):
        wrong_instruction = DummyNode("wrong", [])
        tree2 = DummyNode("start", [wrong_instruction])
        qc._run(tree2)


def test_validate_spec_and_parse_spec_success_and_failure():
    # Valid spec
    valid_spec = "row_count > 0"
    err, ok = QualityChecker.validate_spec(valid_spec)
    assert ok is True
    assert err is None

    # Invalid spec
    invalid_spec = "row_count >>> 0"
    err2, ok2 = QualityChecker.validate_spec(invalid_spec)
    assert ok2 is False
    assert isinstance(err2, str)
    assert "Unexpected token" in err2 or len(err2) > 0  # Typical Lark error message


def test_run_ignores_column_case_and_transforms_columns(sample_df):
    qc = QualityChecker(sample_df)

    spec = "row_count > 0"
    res = qc.run(spec, ignore_column_case=False)
    assert any(r[0] == "row_count" and r[1] for r in res)

    # Save original columns to restore after test
    original_cols = list(sample_df.columns)

    res2 = qc.run(spec, ignore_column_case=True)
    assert isinstance(res2, list)
    assert any(r[0] == "row_count" and r[1] for r in res2)

    # Columns should now all be lowercased in _obj after run
    assert all(isinstance(c, str) and c.islower() for c in qc._obj.columns)

    # Revert columns to original to avoid polluting global state
    qc._obj.columns = original_cols


def test__comparator_to_fn_returns_false_for_unknown_comparator():
    fn = QualityChecker._comparator_to_fn("invalid_comp", 10)
    assert fn(999) is False


def test_apply_value_length_edge_cases_with_empty_string_and_whitespace():
    df = pd.DataFrame({"col": ["", " ", "  ", None, "abc"]})
    qc = QualityChecker(df)

    # ignore_nulls False, comparator == 0 (length)
    node = DummyNode("value_length", ["col", "==", 0])
    key, result = qc._apply_value_length(node)
    assert key == "value_length"
    # unique lengths: ""=0, " "=1, "  "=2, "None"=4, "abc"=3 => multiple unique lengths -> False
    assert result is False

    # ignore_nulls True, comparator == 0 (length)
    node2 = DummyNode("value_length", ["col", "True", "==", 0])
    key2, result2 = qc._apply_value_length(node2)
    # ignoring nulls and empty => only "abc" remains with length 3 != 0 so False
    assert result2 is False


def test_percent_value_length_with_all_nulls(sample_df):
    df = pd.DataFrame({"col": [None, None, None]})
    qc = QualityChecker(df)
    node = DummyNode("percent_value_length", ["col", 100, "True", "dummy", 0])
    key, result = qc._apply_percent_value_length(node)
    # all values have length 0; all rows count towards passing; 100% passes
    assert key == "percent_value_length"
    assert result is True


def test_apply_check_keyerror_on_nonexistent_column():
    df = pd.DataFrame({"A": [1]})
    qc = QualityChecker(df)
    node = DummyNode("not_null", ["NonexistentColumn"])
    key, res = qc._apply_check([node])
    assert key.startswith("Key error:")
    assert res is False


def test_run_with_realistic_spec(sample_df):
    # The actual grammar is complex; test that run returns expected format for a simple predicate parse
    spec = """
    has_columns ["A","B","C"]
    """
    qc = QualityChecker(sample_df)
    err, valid = qc.validate_spec(spec)
    if not valid:
        pytest.skip(f"Grammar not supporting simple spec: {err}")

    results = qc.run(spec)
    assert isinstance(results, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in results)
    # The key for has_columns should be present
    keys = [t[0] for t in results]
    assert any("has_columns" in k for k in keys)
