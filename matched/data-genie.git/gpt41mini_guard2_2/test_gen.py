import pytest
import numpy as np
import pandas as pd
import random
import torch
from lark import Tree
from genie_pkg.dqc import QualityChecker, GenieException


def make_node(data, children):
    return Tree(data, children)


def test_init_with_empty_dataframe_raises():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    with pytest.raises(AttributeError):
        _ = QualityChecker(df)


def test_comparator_to_fn_behaviour():
    fn_gt = QualityChecker._comparator_to_fn(">", 5)
    assert fn_gt(6) is True
    assert fn_gt(5) is False
    fn_eq = QualityChecker._comparator_to_fn("==", 3)
    assert fn_eq(3) is True
    assert fn_eq(4) is False
    fn_lt = QualityChecker._comparator_to_fn("<", 10)
    assert fn_lt(9) is True
    assert fn_lt(10) is False
    fn_other = QualityChecker._comparator_to_fn("unknown", 0)
    assert fn_other(100) is False


def test_treat_column_name_with_ignore_case(monkeypatch):
    df = pd.DataFrame({"A": [1]})
    qc = QualityChecker(df)
    # default ignore_column_case False, returns original
    assert qc._treat_column_name('COLUMN') == 'COLUMN'
    qc.ignore_column_case = True
    assert qc._treat_column_name('ColUMn') == 'column'


def test_apply_quantile_check_true_and_false():
    # Seed numpy because quantile uses it internally
    np.random.seed(42)
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    # Build node:
    # node.children = [column_name, quantile, comparator, rhs]
    node = make_node(
        "quantile",
        ["val", "0.5", ">", "2"]
    )
    result = qc._apply_quantile(node)
    # median is 3, 3 > 2 -> True
    assert result[0] == "quantile"
    assert result[1] is True

    node2 = make_node(
        "quantile",
        ["val", "0.5", "<", "2"]
    )
    result2 = qc._apply_quantile(node2)
    # median is 3, 3 < 2 -> False
    assert result2[0] == "quantile"
    assert result2[1] is False


def test_apply_date_validation_valid_and_invalid():
    df = pd.DataFrame({"dates": pd.to_datetime(["2020-01-01", "2021-02-02"])})
    qc = QualityChecker(df)
    node_valid = make_node("is_date", ["dates"])
    name, passed = qc._apply_date_validation(node_valid)
    assert name == "is_date"
    assert passed is True

    df_invalid = pd.DataFrame({"dates": ["notadate", "also not"]})
    qc2 = QualityChecker(df_invalid)
    node_invalid = make_node("is_date", ["dates"])
    name2, passed2 = qc2._apply_date_validation(node_invalid)
    assert name2.startswith("Date parse error:")
    assert passed2 is False


def test_apply_has_one_of_ignore_case_behaviour():
    df = pd.DataFrame({
        "col": ["A", "b", "C"]
    })
    qc = QualityChecker(df)

    # Allowed values ["A","B","C"], ignore_case=False (default)
    children_nodes = [make_node("str", []), make_node("container", [make_node("str", [])])]
    # We mock this by manually creating Tree with string values directly
    node_nc = Tree("has_one_of", ["col", make_node("values", [
        make_node("str", []),  # Will replace children next
        make_node("str", [])
    ])])

    # Since children are normally Tokens, fix to strings in our test:
    # We'll create node with children:
    node = Tree("has_one_of", [
        "col",
        Tree("values", [
            type("Token", (), {"value":"\"A\""})(),
            type("Token", (), {"value":"\"b\""})(),
            type("Token", (), {"value":"\"C\""})(),
        ])
    ])
    # no ignore_case argument (len(children)=2) means ignore_case=False

    # Values in col are "A", "b", "C"
    # allowed_values lowercased: "a","b","c"
    # But since ignore_case=False, no lower, so "A","b","C"
    # unique_values = ["A","b","C"]
    # Check all unique_values in allowed: True
    _, res = qc._apply_has_one_of(node)
    assert res is True

    # Now test ignore_case=True (len(children)==3, third child == 'True')
    node_ic = Tree("has_one_of", [
        "col",
        Tree("values", [
            type("Token", (), {"value":"\"A\""})(),
            type("Token", (), {"value":"\"b\""})(),
            type("Token", (), {"value":"\"c\""})(),
        ]),
        "True"
    ])
    _, res_ic = qc._apply_has_one_of(node_ic)
    assert res_ic is True

    # Test fail case: unique contains "D" which is not allowed
    df2 = pd.DataFrame({"col": ["A", "B", "D"]})
    qc2 = QualityChecker(df2)
    _, res_fail = qc2._apply_has_one_of(node_ic)
    assert res_fail is False


def test_apply_value_length_ignore_nulls_and_comparator():
    df = pd.DataFrame({
        "col": ["aaa", "bbb", "ccc"]
    })
    qc = QualityChecker(df)

    # Comparator '==', len(unique_lengths) ==1 and that length == 3
    node1 = Tree("value_length", ["col", "==", "3"])
    name1, res1 = qc._apply_value_length(node1)
    assert name1 == "value_length"
    assert res1 is True

    # Comparator '>', length > 2 true
    node2 = Tree("value_length", ["col", ">", "2"])
    name2, res2 = qc._apply_value_length(node2)
    assert res2 is True

    # Comparator '<', length < 3 false
    node3 = Tree("value_length", ["col", "<", "3"])
    _, res3 = qc._apply_value_length(node3)
    assert res3 is False

    # With ignore_nulls = 'True', some null values, length unique should be 1 after dropping nulls
    df2 = pd.DataFrame({"col": ["aaa", " ", None, "aaa"]})
    qc2 = QualityChecker(df2)
    node4 = Tree("value_length", ["col", "True", "==", "3"])
    _, res4 = qc2._apply_value_length(node4)
    assert res4 is True


def test_apply_percent_value_length_various_cases():
    df = pd.DataFrame({
        "col": ["a", "b", "cc", None, "dd", ""]
    })
    qc = QualityChecker(df)

    # 50% of rows satisfy length == 2 (cc, dd) out of total 6 rows
    node1 = Tree("percent_value_length", ["col", "33", "False", "==", "2"])
    name1, res1 = qc._apply_percent_value_length(node1)
    assert name1 == "percent_value_length"
    # 2/6*100=33.33 >= 33 True
    assert res1 is True

    # With ignore_nulls='True' => drop rows with empty "" or None before checking
    node2 = Tree("percent_value_length", ["col", "50", "True", "==", "2"])
    _, res2 = qc._apply_percent_value_length(node2)
    # Rows after dropping null/empty: a,b,cc,dd = 4 rows; 2 with length==2 => 2/4*100=50 >=50 True
    assert res2 is True

    # Fail case, require 70% threshold which is not met
    node3 = Tree("percent_value_length", ["col", "70", "True", "==", "2"])
    _, res3 = qc._apply_percent_value_length(node3)
    assert res3 is False


def test_apply_has_columns_ignore_case_behaviour():
    df = pd.DataFrame(columns=["A", "b", "C"])
    qc = QualityChecker(df)

    # Without ignore_case (default)
    node1 = Tree("has_columns", [Tree("columns", [
        type("Token", (), {"value": "\"A\""})(),
        type("Token", (), {"value": "\"b\""})(),
    ])])
    name1, res1 = qc._apply_has_columns(node1)
    assert name1 == "has_columns"
    assert res1 is True

    # With ignore_case=True, request lowercase columns, df columns uppercase/lowercase mix
    node2 = Tree("has_columns", [
        Tree("columns", [
            type("Token", (), {"value": "\"a\""})(),
            type("Token", (), {"value": "\"B\""})(),
        ]),
        "True"
    ])
    name2, res2 = qc._apply_has_columns(node2)
    assert res2 is True

    # Fail case: missing column "D"
    node3 = Tree("has_columns", [
        Tree("columns", [
            type("Token", (), {"value": "\"A\""})(),
            type("Token", (), {"value": "\"D\""})(),
        ])
    ])
    _, res3 = qc._apply_has_columns(node3)
    assert res3 is False


def test_apply_check_variants(monkeypatch):
    # Build df with various columns to test all _apply_check branches
    df = pd.DataFrame({
        "unique_col": [1, 2, 3],
        "not_null_col": [10, 20, 30],
        "positive_col": [5, 6, 7],
        "has_cols_check": [1],
        "val_len_col": ["aa", "bb", "cc"],
        "date_col": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "has_one_of_col": ["x", "y", "z"],
        "quant_col": [3, 4, 5]
    })
    qc = QualityChecker(df)

    # row_count: check True when row_count > 2
    node_rc = make_node("row_count", [">", "2"])
    result_rc = qc._apply_check([node_rc])
    assert result_rc == ("row_count", True)

    # has_columns: columns exist
    node_hc = make_node("has_columns", [Tree("columns", [type("Token", (), {"value":"\"unique_col\""})()])])
    result_hc = qc._apply_check([node_hc])
    assert result_hc[0] == "has_columns"
    assert result_hc[1] is True

    # is_unique: column unique_col is unique
    node_iu = make_node("is_unique", ["unique_col"])
    result_iu = qc._apply_check([node_iu])
    assert result_iu == ("is_unique", True)

    # not_null: not_null_col has no nulls
    node_nn = make_node("not_null", ["not_null_col"])
    result_nn = qc._apply_check([node_nn])
    assert result_nn == ("not_null", True)

    # is_positive: positive_col all positive >0
    node_ip = make_node("is_positive", ["positive_col"])
    result_ip = qc._apply_check([node_ip])
    assert result_ip == ("is_positive", True)

    # has_one_of: values x,y,z allowed? Test with correct allowed values
    node_ho = Tree("has_one_of", [
        "has_one_of_col",
        Tree("values", [
            type("Token", (), {"value":"\"x\""})(),
            type("Token", (), {"value":"\"y\""})(),
            type("Token", (), {"value":"\"z\""})(),
        ])
    ])
    result_ho = qc._apply_check([node_ho])
    assert result_ho == ("has_one_of", True)

    # quantile: quant_col median > 3
    node_q = Tree("quantile", ["quant_col", "0.5", ">", "3"])
    result_q = qc._apply_check([node_q])
    # median of [3,4,5] is 4 > 3 True
    assert result_q == ("quantile", True)

    # is_date: date_col valid dates
    node_id = make_node("is_date", ["date_col"])
    result_id = qc._apply_check([node_id])
    assert result_id == ("is_date", True)

    # value_length: val_len_col all length == 2
    node_vl = Tree("value_length", ["val_len_col", "==", "2"])
    result_vl = qc._apply_check([node_vl])
    assert result_vl == ("value_length", True)

    # percent_value_length: value length percent test for length 2
    node_pvl = Tree("percent_value_length", [
        "val_len_col", "100", "False", "==", "2"
    ])
    result_pvl = qc._apply_check([node_pvl])
    assert result_pvl == ("percent_value_length", True)

    # Unknown data attribute leads to GenieException
    unknown_node = make_node("unknown_check", [])
    with pytest.raises(GenieException):
        qc._apply_check([unknown_node])

    # Test KeyError in _apply_check returning error tuple
    node_missing_col = make_node("not_null", ["nonexistent_column"])
    name, res = qc._apply_check([node_missing_col])
    assert name.startswith("Key error:")
    assert res is False


def test_apply_predicates_with_single_code_block(monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    qc = QualityChecker(df)

    # Prepare a predicate of type code_block with one check -> row_count > 0
    check_node = make_node("row_count", [">", "0"])
    code_block_node = Tree("code_block", [Tree("row_count", [">", "0"])])
    predicates_node = Tree("predicates", [code_block_node])

    # Apply predicates: should process one check and return list with one tuple
    results = qc._apply_predicates(predicates_node.children)
    # The result list has the check result inserted at position 0, so should be length 1
    assert isinstance(results, list)
    assert len(results) == 1
    name, passed = results[0]
    assert name == "row_count"
    assert passed is True

    # If predicate data unknown raises GenieException
    unknown_predicate = make_node("unknown_predicate", [])
    with pytest.raises(GenieException):
        qc._apply_predicates([unknown_predicate])


def test__run_with_predicates_and_raises():
    df = pd.DataFrame({"a": [1]})
    qc = QualityChecker(df)

    # Build a Tree with one child of data = predicates with one code_block child
    check_node = Tree("row_count", [">", "0"])
    code_block = Tree("code_block", [check_node])
    predicates = Tree("predicates", [code_block])
    root = Tree("root", [predicates])

    results = qc._run(root)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0][0] == "row_count"
    assert results[0][1] is True

    # Test with children length not 1 raises
    bad_root = Tree("root", [predicates, code_block])
    with pytest.raises(GenieException):
        qc._run(bad_root)

    # Test unknown instruction raises
    unknown_inst = Tree("unknown", [])
    root2 = Tree("root", [unknown_inst])
    with pytest.raises(GenieException):
        qc._run(root2)


def test_parse_spec_and_validate_spec(monkeypatch):
    # A minimal valid grammar spec for row_count
    valid_spec = "row_count > 0"
    msg, valid = QualityChecker.validate_spec(valid_spec)
    assert valid is True
    assert msg is None

    # Invalid spec returns error message string and False
    invalid_spec = "this is not a spec!"
    msg2, valid2 = QualityChecker.validate_spec(invalid_spec)
    assert valid2 is False
    assert isinstance(msg2, str) and len(msg2) > 0


def test_run_method_ignore_column_case_changes_columns(monkeypatch):
    df = pd.DataFrame({
        "A": [1, 2],
        "b": [3, 4]
    })
    qc = QualityChecker(df.copy())
    initial_columns = list(qc._obj.columns)
    spec = "row_count > 0"
    results = qc.run(spec, ignore_column_case=True)
    # Columns changed to lowercase
    assert all(isinstance(c, str) and c.islower() for c in qc._obj.columns)
    # The results list must not be empty and be correct for row_count > 0
    assert any(name == "row_count" for name, _ in results)

    # Calling without ignore_column_case leaves columns unchanged
    qc2 = QualityChecker(df.copy())
    results2 = qc2.run(spec, ignore_column_case=False)
    assert list(qc2._obj.columns) == list(df.columns)


def test_run_method_raises_for_unknown_instruction(monkeypatch):
    df = pd.DataFrame({"x": [1]})
    qc = QualityChecker(df)

    # Patch _parse_spec to return tree with unknown data instruction
    def fake_parse_spec(_):
        return Tree("root", [Tree("unknown_instruction", [])])
    monkeypatch.setattr(QualityChecker, "_parse_spec", fake_parse_spec)

    with pytest.raises(GenieException) as e:
        qc.run("anything")
    assert "Unknown instruction" in str(e.value)
