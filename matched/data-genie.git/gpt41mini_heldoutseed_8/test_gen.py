import pytest
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException

# Helper function to create a Tree node for custom testing
def create_node(data, children):
    return Tree(data=data, children=children)

def test_init_rejects_empty_df():
    df = pd.DataFrame()
    with pytest.raises(AttributeError):
        QualityChecker(df)

def test_init_accepts_nonempty_df():
    df = pd.DataFrame({'a': [1,2]})
    qc = QualityChecker(df)
    assert qc._obj.equals(df)
    assert not qc.ignore_column_case

def test_comparator_to_fn():
    gt_fn = QualityChecker._comparator_to_fn(">", 5)
    eq_fn = QualityChecker._comparator_to_fn("==", 3)
    lt_fn = QualityChecker._comparator_to_fn("<", 10)
    false_fn = QualityChecker._comparator_to_fn("!=", 0)  # unknown comparator

    assert gt_fn(6) is True
    assert gt_fn(5) is False
    assert eq_fn(3) is True
    assert eq_fn(2) is False
    assert lt_fn(9) is True
    assert lt_fn(10) is False
    assert false_fn(100) is False

def test_treat_column_name_case_sensitive_and_ignore_case():
    df = pd.DataFrame({'Col': [1]})
    qc = QualityChecker(df)

    # case sensitive
    qc.ignore_column_case = False
    assert qc._treat_column_name("Col") == "Col"
    assert qc._treat_column_name("col") == "col"

    # ignore case
    qc.ignore_column_case = True
    assert qc._treat_column_name("Col") == "col"
    assert qc._treat_column_name("CoL") == "col"

def test_apply_quantile_pass_and_fail():
    df = pd.DataFrame({'val': [1,2,3,4,5]})
    qc = QualityChecker(df)

    # Prepare node: data='quantile', children: column_name, quantile, comparator, rhs
    # quantile 0.5 (median) = 3
    node = create_node("quantile", ["val", "0.5", ">", "2"])
    data, result = qc._apply_quantile(node)
    assert data == "quantile"
    assert result is True  # median 3 > 2 is True

    node = create_node("quantile", ["val", "0.5", "<", "2"])
    data, result = qc._apply_quantile(node)
    assert result is False  # median 3 < 2 is False

def test_apply_date_validation_success_and_failure():
    df = pd.DataFrame({'date': ['2022-01-01', '2023-02-02'], 'not_date': ['hello', 'world']})
    qc = QualityChecker(df)

    node = create_node("is_date", ["date"])
    data, result = qc._apply_date_validation(node)
    assert data == "is_date"
    assert result is True

    node = create_node("is_date", ["not_date"])
    data, result = qc._apply_date_validation(node)
    assert data.startswith("Date parse error")
    assert result is False

def test_apply_has_one_of_case_sensitive_and_insensitive():
    df = pd.DataFrame({'col': ['A', 'B', 'C', 'A']})
    qc = QualityChecker(df)

    # Allowed values ["A", "B", "C"]
    # Without ignore case argument
    values_node = create_node("", [Tree("str", [type("Token", (), {"value": "\"A\""})()]),
                                  Tree("str", [type("Token", (), {"value": "\"B\""})()]),
                                  Tree("str", [type("Token", (), {"value": "\"C\""})()])])
    node = create_node("has_one_of", ["col", values_node])
    data, result = qc._apply_has_one_of(node)
    assert data == "has_one_of"
    assert result is True

    # Let's add a value 'D' to df to cause failure
    df2 = pd.DataFrame({'col': ['A', 'B', 'D']})
    qc2 = QualityChecker(df2)
    data, result = qc2._apply_has_one_of(node)
    assert result is False

    # Test ignore case = True
    node_ignore_case = create_node("has_one_of", ["col", values_node, 'True'])
    df3 = pd.DataFrame({'col': ['a', 'b', 'c']})
    qc3 = QualityChecker(df3)
    data, result = qc3._apply_has_one_of(node_ignore_case)
    assert result is True

def test_apply_value_length_basic_cases():
    df = pd.DataFrame({'col': ['aa', 'bb', 'cc'], 'col_null': ['aa', None, 'cc']})
    qc = QualityChecker(df)

    # Without ignore_nulls (len children=3)
    node = create_node("value_length", ["col", "==", "2"])
    data, result = qc._apply_value_length(node)
    assert result is True  # all length 2

    # With ignore_nulls='False', comparator '==', rhs 2
    node2 = create_node("value_length", ["col_null", "False", "==", "2"])
    data2, result2 = qc._apply_value_length(node2)
    assert result2 is True

    # With ignore_nulls='True', comparator '==', rhs 2 
    node3 = create_node("value_length", ["col_null", "True", "==", "2"])
    data3, result3 = qc._apply_value_length(node3)
    assert result3 is True

    # Introduce inconsistent length
    df2 = pd.DataFrame({'col': ['a', 'bb', 'ccc']})
    qc2 = QualityChecker(df2)
    node = create_node("value_length", ["col", "==", "1"])
    data, result = qc2._apply_value_length(node)
    assert result is False

def test_apply_percent_value_length_basic():
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'ccc', 'ddd', None, '']})
    qc = QualityChecker(df)

    # len=3 count 4 out of 6 -> ~66.6%
    node = create_node("percent_value_length", ["col", "50", "False", "==", "3"])
    data, result = qc._apply_percent_value_length(node)
    assert result is True

    # Percent 70% would fail
    node2 = create_node("percent_value_length", ["col", "70", "False", "==", "3"])
    data2, result2 = qc._apply_percent_value_length(node2)
    assert result2 is False

    # ignore_nulls='True' excludes None and '' so 4 out of 4 => 100%
    node3 = create_node("percent_value_length", ["col", "100", "True", "==", "3"])
    data3, result3 = qc._apply_percent_value_length(node3)
    assert result3 is True

def test_apply_has_columns_case_sensitive_and_insensitive():
    df = pd.DataFrame({'A': [1], 'b': [2]})
    qc = QualityChecker(df)

    columns_node = create_node("", [type("Token", (), {"value": "\"A\""})(),
                                  type("Token", (), {"value": "\"b\""})()])
    node = create_node("has_columns", [columns_node])
    data, result = qc._apply_has_columns(node)
    assert result is True

    # missing columns
    columns_node2 = create_node("", [type("Token", (), {"value": "\"A\""})(),
                                   type("Token", (), {"value": "\"c\""})()])
    node2 = create_node("has_columns", [columns_node2])
    data2, result2 = qc._apply_has_columns(node2)
    assert result2 is False

    # Ignore case = True
    node_ic = create_node("has_columns", [columns_node2, 'True'])
    data_ic, result_ic = qc._apply_has_columns(node_ic)
    assert result_ic is False

    # Add column 'C' lowercase in df, should pass with ignore case
    df2 = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    qc2 = QualityChecker(df2)
    node_ic2 = create_node("has_columns", [columns_node2, 'True'])
    data_ic2, result_ic2 = qc2._apply_has_columns(node_ic2)
    assert result_ic2 is True

def test_apply_check_row_count_has_columns_is_unique_not_null_is_positive():
    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    qc = QualityChecker(df)

    # row_count: > 2
    row_count_node = create_node("row_count", [">", "2"])
    result = qc._apply_check([row_count_node])
    assert result == ("row_count", True)

    # row_count: < 3 (3 rows, so False)
    row_count_node2 = create_node("row_count", ["<", "3"])
    result2 = qc._apply_check([row_count_node2])
    assert result2 == ("row_count", False)

    # has_columns: columns A B
    columns_node = create_node("", [type("Token", (), {"value": "\"A\""})(),
                                   type("Token", (), {"value": "\"B\""})()])
    has_columns_node = create_node("has_columns", [columns_node])
    result3 = qc._apply_check([has_columns_node])
    assert result3 == ("has_columns", True)

    # is_unique: all unique in column A (which they are)
    is_unique_node = create_node("is_unique", ["A"])
    result4 = qc._apply_check([is_unique_node])
    assert result4 == ("is_unique", True)

    # not_null: column A has no nulls
    not_null_node = create_node("not_null", ["A"])
    result5 = qc._apply_check([not_null_node])
    assert result5 == ("not_null", True)

    # is_positive: all > 0 in A
    is_positive_node = create_node("is_positive", ["A"])
    result6 = qc._apply_check([is_positive_node])
    assert result6 == ("is_positive", True)

    # is_positive: with negative value returns False
    df2 = pd.DataFrame({'A': [1, -1, 3]})
    qc2 = QualityChecker(df2)
    is_positive_node2 = create_node("is_positive", ["A"])
    result7 = qc2._apply_check([is_positive_node2])
    assert result7 == ("is_positive", False)

def test_apply_check_has_one_of_quantile_date_value_length_percent_value_length():
    df = pd.DataFrame({'col': ['A', 'B'], 'num': [1, 2], 'dt': ['2020-01-01', '2020-01-02']})
    qc = QualityChecker(df)

    # has_one_of
    values_node = create_node("", [type("Token", (), {"value": "\"A\""})(),
                                  type("Token", (), {"value": "\"B\""})()])
    has_one_of_node = create_node("has_one_of", ["col", values_node])
    result = qc._apply_check([has_one_of_node])
    assert result == ("has_one_of", True)

    # quantile for num > 1
    quantile_node = create_node("quantile", ["num", "0.5", ">", "1"])
    result2 = qc._apply_check([quantile_node])
    assert result2[0] == "quantile"
    assert result2[1] is True

    # is_date for dt
    is_date_node = create_node("is_date", ["dt"])
    result3 = qc._apply_check([is_date_node])
    assert result3 == ("is_date", True)

    # value_length
    value_length_node = create_node("value_length", ["col", "==", "1"])
    result4 = qc._apply_check([value_length_node])
    assert result4[0] == "value_length"

    # percent_value_length
    percent_value_length_node = create_node("percent_value_length", ["col", "50", "False", "==", "1"])
    result5 = qc._apply_check([percent_value_length_node])
    assert result5[0] == "percent_value_length"

def test_apply_check_unknown_raises_genie_exception():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    unknown_node = create_node("foobar", [])
    with pytest.raises(GenieException):
        qc._apply_check([unknown_node])

def test_apply_check_keyerror_returns_false():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    # Provide a column name that doesn't exist for is_unique
    node = create_node("is_unique", ["missing_column"])
    msg, val = qc._apply_check([node])
    assert val is False
    assert "Key error" in msg

def test_apply_predicates_accepts_code_block_and_raises_on_unknown():
    df = pd.DataFrame({'a': [1, 2]})
    qc = QualityChecker(df)

    # code_block with one check: row_count > 1
    row_count_node = create_node("row_count", [">", "1"])
    code_block = create_node("code_block", [row_count_node])
    predicates = create_node("predicates", [code_block])
    results = qc._apply_predicates(predicates.children)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"

    # Unknown predicate data
    fake_pred = create_node("fake_pred", [])
    with pytest.raises(GenieException):
        qc._apply_predicates(fake_pred.children)

def test_run_accepts_valid_predicates_tree_and_raises_unknown_instruction():
    df = pd.DataFrame({'a': [1, 2]})
    qc = QualityChecker(df)

    # Create predicate tree structure for run
    row_count_node = create_node("row_count", [">", "1"])
    code_block = create_node("code_block", [row_count_node])
    predicates = create_node("predicates", [code_block])
    pt = create_node("root", [predicates])

    results = qc._run(pt)
    assert isinstance(results, list)
    assert results[0][0] == "row_count"

    # Multiple children should raise
    pt_multiple = create_node("root", [predicates, predicates])
    with pytest.raises(GenieException):
        qc._run(pt_multiple)

    # Unknown instruction data
    pt_unknown = create_node("root", [create_node("fake", [])])
    with pytest.raises(GenieException):
        qc._run(pt_unknown)

def test_validate_spec_valid_and_invalid():
    valid_spec = "row_count > 0"
    error, valid = QualityChecker.validate_spec(valid_spec)
    assert valid is True
    assert error is None

    invalid_spec = "row_count >> 0"
    error2, valid2 = QualityChecker.validate_spec(invalid_spec)
    assert valid2 is False
    assert isinstance(error2, str)
    assert len(error2) > 0

def test_run_with_ignore_column_case_lowercases_df_columns():
    df = pd.DataFrame({'A': [1, 2]})
    qc = QualityChecker(df)
    spec = "row_count > 1"

    results = qc.run(spec, ignore_column_case=True)
    assert isinstance(results, list)
    # Columns should now be lowercase
    assert all(col.islower() for col in qc._obj.columns)
    assert results[0][0] == "row_count"

def test_run_with_ignore_column_case_false_keeps_columns():
    df = pd.DataFrame({'A': [1, 2]})
    qc = QualityChecker(df)
    spec = "row_count > 1"

    results = qc.run(spec, ignore_column_case=False)
    assert isinstance(results, list)
    # Columns remain as original case
    assert 'A' in qc._obj.columns
    assert results[0][0] == "row_count"

