import pytest
import pandas as pd
from lark import Tree
from genie_pkg.dqc import QualityChecker
from genie_pkg import GenieException
import numpy as np

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'x', 'y'],
        'C': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
        'D': ['apple', 'apple', 'apple', 'apple', 'apple'],
        'E': [10, None, 30, 40, 50],
        'F': ['a', '  ', '', 'b', None]
    })

@pytest.fixture
def accessor(simple_df):
    return simple_df.dqc

def test_empty_dataframe_raises():
    df = pd.DataFrame({'A': [None, None], 'B': [None, None]})
    with pytest.raises(AttributeError):
        _ = df.dqc

def test_comparator_to_fn_behaviors():
    cmp = QualityChecker._comparator_to_fn
    fn_gt = cmp(">", 3)
    fn_eq = cmp("==", 2)
    fn_lt = cmp("<", 5)
    fn_false = cmp("xxx", 10)
    assert fn_gt(4) is True
    assert fn_gt(3) is False
    assert fn_eq(2) is True
    assert fn_eq(3) is False
    assert fn_lt(4) is True
    assert fn_lt(5) is False
    assert fn_false(100) is False

def test_treat_column_name_case_sensitive(accessor):
    accessor.ignore_column_case = False
    assert accessor._treat_column_name("ABC") == "ABC"
    accessor.ignore_column_case = True
    assert accessor._treat_column_name("ABC") == "abc"

def make_node(data, children):
    # Helper to create dummy nodes imitating lark.Tree with .data and .children
    return Tree(data, children)

def test_apply_quantile_true_and_false(accessor):
    # Use column 'A' for quantile 0.5; median is 3.
    # Check > 2 (True), < 2 (False)
    node_true = make_node("quantile", ['A', '0.5', '>', '2'])
    node_false = make_node("quantile", ['A', '0.5', '<', '2'])
    label, result_true = accessor._apply_quantile(node_true)
    label, result_false = accessor._apply_quantile(node_false)
    assert label == "quantile"
    assert result_true is True
    assert result_false is False

def test_apply_date_validation_success_and_fail():
    df = pd.DataFrame({
        'good_dates': ['2021-01-01', '2021-02-02'],
        'bad_dates': ['xxx', '2021-03-03']
    })
    acc = df.dqc
    node_good = make_node("is_date", ['good_dates'])
    node_bad = make_node("is_date", ['bad_dates'])
    lbl_good, res_good = acc._apply_date_validation(node_good)
    lbl_bad, res_bad = acc._apply_date_validation(node_bad)
    assert lbl_good == "is_date"
    assert res_good is True
    assert "Date parse error" in lbl_bad
    assert res_bad is False

def test_apply_has_one_of_basic_and_ignore_case():
    df = pd.DataFrame({'col': ['Dog', 'Cat', 'dog']})
    acc = df.dqc
    # Node children: [column_name, allowed_values_tree, optional ignore_case str]
    allowed_tree = Tree('allowed_values', [
        Tree('VALUE', ['"Dog"']),
        Tree('VALUE', ['"Cat"']),
    ])
    node_no_ic = make_node("has_one_of", ['col', allowed_tree])
    node_ic_true = make_node("has_one_of", ['col', allowed_tree, 'True'])
    node_ic_false = make_node("has_one_of", ['col', allowed_tree, 'False'])
    label, result_no_ic = acc._apply_has_one_of(node_no_ic)
    label, result_ic_true = acc._apply_has_one_of(node_ic_true)
    label, result_ic_false = acc._apply_has_one_of(node_ic_false)
    assert label == "has_one_of"
    assert result_no_ic is False  # 'dog' not exactly in allowed (case sensitive)
    assert result_ic_true is True  # ignore case makes it True
    assert result_ic_false is False

def test_apply_value_length_ignore_nulls_and_comparators():
    df = pd.DataFrame({'col': ['aa', 'bb', None, '', 'ccc']})
    acc = df.dqc
    # Without ignore nulls: lengths unique = [2, 3, 4 (None -> 'None'), 0], so fails len(unique) == 1
    node1 = make_node("value_length", ['col', '==', '2'])
    label1, result1 = acc._apply_value_length(node1)
    # With ignore nulls ('True'), length unique should consider only 'aa', 'bb', 'ccc' -> lengths 2,2,3 => fails again
    node2 = make_node("value_length", ['col', 'True', '==', '2'])
    label2, result2 = acc._apply_value_length(node2)
    # Now let's make a df where length unique is exactly one and length == 2:
    df2 = pd.DataFrame({'col': ['aa', 'bb', 'cc']})
    acc2 = df2.dqc
    node3 = make_node("value_length", ['col', '==', '2'])
    label3, result3 = acc2._apply_value_length(node3)
    assert label1 == "value_length"
    assert result1 is False
    assert label2 == "value_length"
    assert result2 is False
    assert label3 == "value_length"
    assert result3 is True

def test_apply_percent_value_length_percentage_check():
    df = pd.DataFrame({'col': ['a', 'bb', '', None, 'ccc', 'dd']})
    acc = df.dqc
    # percent=50, length to check = 2, ignore_null False, rhs = 2
    node1 = make_node("percent_value_length", ['col', '50', '==', '2'])
    label1, result1 = acc._apply_percent_value_length(node1)
    # Since 3 rows have length 2: 'bb', 'dd', 'ccc'(3) not 2 so passing=2 rows -> 2/6*100=33.33 < 50 => False
    assert label1 == "percent_value_length"
    assert result1 is False

    # With ignore_nulls True (3rd child), percent=30, ignore_nulls='True', rhs=1 (meaning lengths of 1)
    node2 = make_node("percent_value_length", ['col', '30', 'True', '==', '1'])
    label2, result2 = acc._apply_percent_value_length(node2)
    # After replacement and dropna: only non-null non-empty length counts
    # Values: 'a'(1), 'bb'(2), 'ccc'(3), 'dd'(2)
    # length==1 rows = 1 row -> 1/6*100=16.66 < 30 False
    assert label2 == "percent_value_length"
    assert result2 is False

def test_apply_has_columns_with_and_without_ignore_case():
    df = pd.DataFrame(columns=['A', 'b', 'C'])
    acc = df.dqc
    # Create node children: [columns_tree, optional ignore_case]
    columns_tree = Tree('columns', [
        Tree('VALUE', ['"A"']),
        Tree('VALUE', ['"B"'])
    ])
    node_no_ic = make_node("has_columns", [columns_tree])
    node_ic_true = make_node("has_columns", [columns_tree, 'True'])
    node_ic_false = make_node("has_columns", [columns_tree, 'False'])
    label1, res1 = acc._apply_has_columns(node_no_ic)
    label2, res2 = acc._apply_has_columns(node_ic_true)
    label3, res3 = acc._apply_has_columns(node_ic_false)
    assert label1 == "has_columns"
    # Without ignore case, 'B' not in columns (has 'b' lowercase), so False
    assert res1 is False
    # With ignore case True, it matches 'b' with 'B' so True
    assert res2 is True
    assert res3 is False

def test_apply_check_row_count(accessor):
    # Create a node for row_count check: comparator '<', quantity 10, row count is 5
    node = make_node("row_count", ['<', '10'])
    label, res = accessor._apply_check([node])
    assert label == "row_count"
    assert res is True
    node2 = make_node("row_count", ['==', '5'])
    label2, res2 = accessor._apply_check([node2])
    assert label2 == "row_count"
    assert res2 is True
    node3 = make_node("row_count", ['>', '10'])
    label3, res3 = accessor._apply_check([node3])
    assert label3 == "row_count"
    assert res3 is False

def test_apply_check_has_columns_and_is_unique_and_not_null(accessor):
    # has_columns node
    columns_tree = Tree('columns', [Tree('VALUE', ['"A"'])])
    node_has_columns = make_node("has_columns", [columns_tree])
    label, res = accessor._apply_check([node_has_columns])
    assert label == "has_columns"
    assert res is True

    # is_unique true: column 'A' is unique
    node_unique = make_node("is_unique", ['A'])
    label2, res2 = accessor._apply_check([node_unique])
    assert label2 == "is_unique"
    assert res2 is True

    # not_null true on column 'A'
    node_not_null = make_node("not_null", ['A'])
    label3, res3 = accessor._apply_check([node_not_null])
    assert label3 == "not_null"
    assert res3 is True

    # not_null false on column 'E' (has None)
    node_not_null2 = make_node("not_null", ['E'])
    label4, res4 = accessor._apply_check([node_not_null2])
    assert label4 == "not_null"
    assert res4 is False

def test_apply_check_is_positive(accessor):
    # Add a column with positive and non-positive values
    accessor._obj['pos'] = [1, 2, 3, 4, 5]
    node_pos = make_node("is_positive", ['pos'])
    label, res = accessor._apply_check([node_pos])
    assert label == "is_positive"
    assert res is True

    accessor._obj['pos'][1] = 0
    label2, res2 = accessor._apply_check([node_pos])
    assert res2 is False

def test_apply_check_not_implemented_raises(accessor):
    node = make_node("unknown_check", ['A'])
    with pytest.raises(GenieException):
        accessor._apply_check([node])

def test_apply_check_key_error_returns_false():
    df = pd.DataFrame({'A': [1, 2]})
    acc = df.dqc
    node = make_node("not_null", ['B'])  # B not in df
    label, res = acc._apply_check([node])
    assert label.startswith("Key error:")
    assert res is False

def test_apply_predicates_calls_apply_check_in_reverse_order(accessor):
    # code_block node with two checks
    check1 = make_node("not_null", ['A'])
    check2 = make_node("is_unique", ['A'])
    code_block = make_node("code_block", [check1, check2])
    predicates = make_node("predicates", [code_block])
    results = accessor._apply_predicates(predicates.children)
    # Results inserted at front, so order reversed compared to children order
    assert len(results) == 2
    assert results[0][0] == "is_unique"
    assert results[1][0] == "not_null"

def test_apply_predicates_unknown_instruction_raises(accessor):
    unknown = make_node("unknown", [])
    predicates = make_node("predicates", [unknown])
    with pytest.raises(GenieException):
        accessor._apply_predicates(predicates.children)

def test_run_with_multiple_children_raises(accessor):
    node1 = make_node("predicates", [])
    node2 = make_node("predicates", [])
    tree = Tree("root", [node1, node2])
    with pytest.raises(GenieException):
        accessor._run(tree)

def test_run_with_unknown_instruction_raises(accessor):
    node = make_node("unknown", [])
    tree = Tree("root", [node])
    with pytest.raises(GenieException):
        accessor._run(tree)

def test_validate_spec_correct_and_incorrect():
    good_spec = "row_count > 0"
    bad_spec = "row_count >< 5"
    err_msg, valid = QualityChecker.validate_spec(good_spec)
    assert valid is True
    assert err_msg is None
    err_msg2, valid2 = QualityChecker.validate_spec(bad_spec)
    assert valid2 is False
    assert isinstance(err_msg2, str)

def test_run_with_ignore_column_case_changes_columns(simple_df):
    spec = "row_count > 0"
    acc = simple_df.dqc
    old_columns = list(acc._obj.columns)
    # Run and ignore case True: columns mapped to lower
    res = acc.run(spec, ignore_column_case=True)
    assert isinstance(res, list)
    # Columns should be mapped to lower
    assert all(c.islower() for c in acc._obj.columns)
    # Default False does not change columns
    acc2 = simple_df.copy().dqc
    _ = acc2.run(spec, ignore_column_case=False)
    assert list(acc2._obj.columns) == old_columns


# Extra helper: test the _parse_spec returns Tree
def test_parse_spec_returns_tree():
    tree = QualityChecker._parse_spec("row_count > 0")
    assert isinstance(tree, Tree)
    assert tree.data == "predicates" or hasattr(tree, 'data')

