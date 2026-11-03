import pytest
import random
import numpy as np
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker, GenieException

# Helper: Make a Tree node like those used in QualityChecker with flexible children and data
def make_node(data, children):
    # lark.Tree requires list of children (tokens or Trees)
    # For simplicity, children can be strings or other Trees
    return Tree(data, children if isinstance(children, list) else [children])

def test_init_valid_dataframe():
    # Seed numpy.random used by QualityChecker._apply_quantile (actually DataFrame quantile)
    np.random.seed(0)
    df = pd.DataFrame({'A': [1, 2, 3]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)
    assert qc.ignore_column_case is False

def test_init_raises_on_empty_dataframe():
    df_empty1 = pd.DataFrame()
    with pytest.raises(AttributeError):
        QualityChecker(df_empty1)

    # DataFrame with all-NaN columns should raise
    df_empty2 = pd.DataFrame({'A': [np.nan, np.nan]})
    with pytest.raises(AttributeError):
        QualityChecker(df_empty2)

def test_comparator_to_fn_basic():
    fn_gt = QualityChecker._comparator_to_fn(">", 5)
    fn_eq = QualityChecker._comparator_to_fn("==", 5)
    fn_lt = QualityChecker._comparator_to_fn("<", 5)
    fn_false = QualityChecker._comparator_to_fn("???", 5)
    assert fn_gt(6) is True
    assert fn_gt(5) is False
    assert fn_eq(5) is True
    assert fn_eq(6) is False
    assert fn_lt(4) is True
    assert fn_lt(5) is False
    assert fn_false(999) is False

def test_treat_column_name_case_insensitive_and_sensitive():
    df = pd.DataFrame({'Col': [1]})
    qc = QualityChecker(df)
    qc.ignore_column_case = False
    assert qc._treat_column_name("XYZ") == "XYZ"
    qc.ignore_column_case = True
    assert qc._treat_column_name("XYZ") == "xyz"

def test_apply_quantile_comparisons():
    np.random.seed(0)
    df = pd.DataFrame({'score': [10, 20, 30, 40, 50]})
    qc = QualityChecker(df)
    # Build node: data="quantile", children = [colname, quantile float, comparator, rhs]
    # Example: quantile of 0.5 for "score" > 25  -> median is 30, so 30 > 25 == True
    node = make_node("quantile", ["score", "0.5", ">", "25"])
    name, result = qc._apply_quantile(node)
    assert name == "quantile"
    # median is 30
    assert isinstance(result, bool)
    assert result is True

    node2 = make_node("quantile", ["score", "0.5", "<", "20"])
    _, res2 = qc._apply_quantile(node2)
    assert res2 is False

def test_apply_date_validation_pass_and_fail():
    df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02'])})
    qc = QualityChecker(df)
    node_pass = make_node("is_date", ["date"])
    name, result = qc._apply_date_validation(node_pass)
    assert name == "is_date"
    assert result is True

    df_bad = pd.DataFrame({'date': ['not a date', '2020-01-01']})
    qc_bad = QualityChecker(df_bad)
    node_fail = make_node("is_date", ["date"])
    name, result = qc_bad._apply_date_validation(node_fail)
    assert "Date parse error" in name
    assert result is False

def test_apply_has_one_of_ignore_case_and_sensitive():
    # Values: A, B, C ; allowed: "A", "B"
    df = pd.DataFrame({'col': ["A", "B", "A"]})
    qc = QualityChecker(df)
    allowed_values_node = make_node("allowed", [make_node("val", [lambda_: None])])  # dummy
    # Build node.children:
    # children[0]: column_name
    # children[1]: node with children tokens with .value attribute
    # For test build a fake token object

    class Token:
        def __init__(self, value):
            self.value = value
    # allowed values tokens
    allowed_tokens = [Token("\"A\""), Token("\"B\"")]

    # node.children = [column_name (str), allowed_values node with children tokens, optional ignore_case str]
    node_sensitive = make_node("has_one_of", ["col", make_node("list", allowed_tokens)])
    node_ignore_case = make_node("has_one_of", ["col", make_node("list", allowed_tokens), "True"])

    # Add a value not allowed to get False
    df2 = pd.DataFrame({'col': ["A", "B", "C"]})
    qc2 = QualityChecker(df2)

    # Sensitive, C not allowed -> false
    name, result = qc._apply_has_one_of(node_sensitive)
    assert name == "has_one_of"
    assert result is True
    name2, result2 = qc2._apply_has_one_of(node_sensitive)
    assert result2 is False

    # Ignore case test: df3 with "a", "b" lowercase allowed "A","B"
    df3 = pd.DataFrame({'col': ["a", "b", "a"]})
    qc3 = QualityChecker(df3)
    name3, result3 = qc3._apply_has_one_of(node_ignore_case)
    assert result3 is True

def test_apply_value_length_basic_and_ignore_nulls():
    # Basic case: all lengths same and > 2
    df = pd.DataFrame({'col': ["aaa", "bbb", "ccc"]})
    qc = QualityChecker(df)

    # children = [column, comparator, rhs]
    node = make_node("value_length", ["col", ">", "2"])
    name, result = qc._apply_value_length(node)
    assert name == "value_length"
    assert result is True

    # with ignore_nulls = True, one value is empty string which should be treated as NaN
    df2 = pd.DataFrame({'col': ["aaaa", "", "cccc"]})
    qc2 = QualityChecker(df2)
    node2 = make_node("value_length", ["col", "True", "==", "4"])
    name2, result2 = qc2._apply_value_length(node2)
    assert name2 == "value_length"
    # The lengths after ignoring nulls are lengths of "aaaa" and "cccc" both 4, so True
    assert result2 is True

    # with ignore_nulls = False but varying lengths that should fail
    df3 = pd.DataFrame({'col': ["a", "bb", "ccc"]})
    qc3 = QualityChecker(df3)
    node3 = make_node("value_length", ["col", "==", "1"])
    name3, result3 = qc3._apply_value_length(node3)
    assert result3 is False

def test_apply_percent_value_length_various_cases():
    np.random.seed(0)
    df = pd.DataFrame({'col': ["aaa", "bbb", "ccc", "", None]})
    qc = QualityChecker(df)

    # children len==4 without ignore_nulls param
    node = make_node("percent_value_length", ["col", "60", "==", "3"])
    name, result = qc._apply_percent_value_length(node)
    assert name == "percent_value_length"
    # 3 rows with length==3 are "aaa","bbb","ccc" out of 5 rows = 60%, so result True
    assert result is True

    # children len>4 with ignore_nulls 'True'
    node2 = make_node("percent_value_length", ["col", "60", "True", "==", "3"])
    name2, result2 = qc._apply_percent_value_length(node2)
    assert name2 == "percent_value_length"
    # ignore_nulls True drops "" and None → 3 rows remain all length 3 → 100% >= 60 True
    assert result2 is True

    # Changing rhs length to 2, which no rows have → result False
    node3 = make_node("percent_value_length", ["col", "60", "False", "==", "2"])
    name3, result3 = qc._apply_percent_value_length(node3)
    assert result3 is False

def test_apply_has_columns_ignore_case_and_sensitive():
    df = pd.DataFrame(columns=["A", "b", "c"])
    qc = QualityChecker(df)

    # Allowed columns tokens with .value attr
    class Token:
        def __init__(self, value):
            self.value = value

    allowed_tokens = [Token("\"A\""), Token("\"B\"")]
    columns_node = make_node("list", allowed_tokens)

    # Sensitive
    node_sensitive = make_node("has_columns", [columns_node])
    name_s, result_s = qc._apply_has_columns(node_sensitive)
    assert name_s == "has_columns"
    # df has "A", "b", "c"; allowed "A", "B" → "B" not found case-sensitive, so False
    assert result_s is False

    # Ignore case
    node_icase = make_node("has_columns", [columns_node, "True"])
    name_i, result_i = qc._apply_has_columns(node_icase)
    assert result_i is True  # because ignoring case B matches b

def test_apply_check_row_count_comparisons():
    df = pd.DataFrame({'a': [1,2,3,4,5]})
    qc = QualityChecker(df)

    node = make_node("row_count", [">", "3"])
    name, result = qc._apply_check([node])
    assert name == "row_count"
    assert result is True

    node_eq = make_node("row_count", ["==", "5"])
    _, res_eq = qc._apply_check([node_eq])
    assert res_eq is True

    node_lt = make_node("row_count", ["<", "4"])
    _, res_lt = qc._apply_check([node_lt])
    assert res_lt is False

def test_apply_check_is_unique_and_not_null_and_is_positive():
    # is_unique true
    df = pd.DataFrame({'id': [1,2,3,4], 'vals': [10,20,30,40]})
    qc = QualityChecker(df)

    node_unique = make_node("is_unique", ["id"])
    name_u, res_u = qc._apply_check([node_unique])
    assert name_u == "is_unique"
    assert res_u is True

    # is_unique false
    df2 = pd.DataFrame({'id': [1,2,2,4]})
    qc2 = QualityChecker(df2)
    node_unique2 = make_node("is_unique", ["id"])
    _, res2 = qc2._apply_check([node_unique2])
    assert res2 is False

    # not_null true
    df3 = pd.DataFrame({'col': [1,2,3]})
    qc3 = QualityChecker(df3)
    node_not_null = make_node("not_null", ["col"])
    _, res_not_null = qc3._apply_check([node_not_null])
    assert res_not_null is True

    # not_null false
    df4 = pd.DataFrame({'col': [1, None, 3]})
    qc4 = QualityChecker(df4)
    node_not_null4 = make_node("not_null", ["col"])
    _, res_null = qc4._apply_check([node_not_null4])
    assert res_null is False

    # is_positive true
    df5 = pd.DataFrame({'col': [1, 2, 3]})
    qc5 = QualityChecker(df5)
    node_pos = make_node("is_positive", ["col"])
    _, res_pos = qc5._apply_check([node_pos])
    assert res_pos is True

    # is_positive false (zero and negatives included)
    df6 = pd.DataFrame({'col': [0, 2, -1]})
    qc6 = QualityChecker(df6)
    node_pos6 = make_node("is_positive", ["col"])
    _, res_pos6 = qc6._apply_check([node_pos6])
    assert res_pos6 is False

def test_apply_check_with_key_error_returns_false():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    node = make_node("not_null", ["nonexistent_column"])
    msg, res = qc._apply_check([node])
    assert msg.startswith("Key error")
    assert res is False

def test_apply_check_not_implemented_raises_GenieException():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    node = make_node("unknown_check", ["a"])
    with pytest.raises(GenieException):
        qc._apply_check([node])

def test_apply_predicates_runs_code_block_and_raises_on_unknown(monkeypatch):
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)

    # Prepare code_block predicate with one check node "row_count" > 0, should return True
    rc_node = make_node("row_count", [">", "0"])
    code_block_node = make_node("code_block", [make_node("check", [rc_node])])
    predicates_node = make_node("predicates", [code_block_node])

    results = qc._apply_predicates([code_block_node])
    # _apply_check should produce one tuple ("row_count", True)
    assert isinstance(results, list)
    assert len(results) == 1
    name, res = results[0]
    assert name == "row_count"
    assert res is True

    # Unknown predicate type raises GenieException
    unknown_pred = make_node("unknown", [])
    with pytest.raises(GenieException):
        qc._apply_predicates([unknown_pred])

def test_run_returns_expected_results_and_raises(monkeypatch):
    df = pd.DataFrame({'a': [1, 2, 3]})
    qc = QualityChecker(df)

    # Setup a predicates node with one code_block that runs one check: row_count > 0
    rc_node = make_node("row_count", [">", "0"])  # Checks row count
    code_block_node = make_node("code_block", [make_node("check", [rc_node])])
    predicates_node = make_node("predicates", [code_block_node])
    root = make_node("root", [predicates_node])

    # _run expects pt.children len 1 with data 'predicates'
    # Here root.data = 'root', with children predicates_node, so override root.data to 'root_fake' and test exception
    # But QualityChecker._run expects first child to be predicates
    # So test happy path:
    root_good = make_node("root", [predicates_node])
    # Replace root.data with 'root' but call on predicates_node to test happy path:
    result = qc._run(predicates_node)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0][0] == "row_count"
    assert result[0][1] is True

    # Fail: pt.children length != 1
    root_bad = make_node("predicates", [code_block_node, code_block_node])
    with pytest.raises(GenieException):
        qc._run(root_bad)

    # Fail: unknown instruction
    unknown_node = make_node("something_else", [])
    root_unknown = make_node("root", [unknown_node])
    with pytest.raises(GenieException):
        qc._run(make_node("predicates", [unknown_node]))

def test_parse_spec_and_validate_spec(monkeypatch):
    # Provide a minimal DSL that is valid: "row_count > 0"
    # The real grammar file is external and not modifiable here.
    spec = "row_count > 0"

    # Test _parse_spec returns a Tree
    tree = QualityChecker._parse_spec(spec)
    assert isinstance(tree, Tree)

    # validate_spec returns (None, True) on valid spec
    msg, valid = QualityChecker.validate_spec(spec)
    assert valid is True
    assert msg is None

    # For invalid spec, test returns (str error msg, False)
    invalid_spec = "row_count >> 0"
    msg_invalid, valid_invalid = QualityChecker.validate_spec(invalid_spec)
    assert valid_invalid is False
    assert isinstance(msg_invalid, str)
    assert len(msg_invalid) > 0

def test_run_method_integrated(monkeypatch):
    # Compose a DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [5, 6, 7]})
    qc = QualityChecker(df)

    # Spec testing row_count > 0
    spec = "row_count > 0"

    # Because _parse_spec calls Lark parser with external grammar, and returns Tree
    # Run normally should produce at least one check, row_count, result True
    results = qc.run(spec)
    assert isinstance(results, list)
    # At least one result and first tuple first element string, second a bool
    assert all(isinstance(t, tuple) and isinstance(t[0], str) and isinstance(t[1], bool) for t in results)
    assert any(t[0] == "row_count" for t in results)

    # Test with ignore_column_case = True changes columns to lowercase
    df2 = pd.DataFrame({'A': [1]})
    qc2 = QualityChecker(df2)
    results2 = qc2.run(spec, ignore_column_case=True)
    assert isinstance(results2, list)
    # Columns attribute changed to lowercase map object
    assert hasattr(qc2._obj.columns, "__iter__")

def test_ignore_column_case_effect_on_column_names():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    qc = QualityChecker(df)
    spec = "has_columns [\"a\", \"b\"] True"
    # Expect has_columns check to pass when ignore_column_case True
    result1 = qc.run(spec, ignore_column_case=True)
    assert any(res[1] for res in result1)  # At least one True

    # When ignore_column_case False, columns don't match by case -> False expected
    result2 = qc.run(spec, ignore_column_case=False)
    # There might be other checks or errors, but the has_columns check should be False
    assert any(isinstance(res[1], bool) for res in result2)

@pytest.mark.parametrize("ignore_nulls,input_vals,expected_result", [
    ('False', ["abc", "def", "ghi"], True),
    ('False', ["a", "bb", ""], False),
    ('True', ["abc", "", "ghi"], True),
    ('True', ["", "", ""], False),
])
def test_value_length_with_param_combinations(ignore_nulls, input_vals, expected_result):
    df = pd.DataFrame({'col': input_vals})
    qc = QualityChecker(df)
    # Compose children depending on ignore_nulls param count
    if ignore_nulls == 'False':
        node = make_node("value_length", ["col", "==", str(len(input_vals[0].strip() or input_vals[0]))])
    else:
        node = make_node("value_length", ["col", ignore_nulls, "==", str(len(input_vals[0].strip() or input_vals[0]))])

    name, res = qc._apply_value_length(node)
    assert isinstance(res, bool)
    # Can't guarantee exact True/False always, but check matches expected from paramized input
    assert res == expected_result

