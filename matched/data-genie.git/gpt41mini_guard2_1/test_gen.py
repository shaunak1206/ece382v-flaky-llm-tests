import pytest
import pandas as pd
import numpy as np
import random
import torch

from genie_pkg.dqc import QualityChecker, GenieException
from lark import Tree


def mk_tree(data, children):
    return Tree(data, children)


def test_init_with_empty_dataframe_raises():
    df = pd.DataFrame()
    with pytest.raises(AttributeError, match="DataFrame is empty"):
        QualityChecker(df)


def test_ignore_column_case_lowercases_columns(monkeypatch):
    # create sample df with uppercase columns
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    qc = QualityChecker(df)
    qc.ignore_column_case = True
    # simulate a node with column name uppercase
    name = qc._treat_column_name('A')
    assert name == 'a'


def test_comparator_functions_return_correct_results():
    # test >, ==, <, else fallback
    fn_gt = QualityChecker._comparator_to_fn(">", 10)
    assert fn_gt(11) is True
    assert fn_gt(10) is False

    fn_eq = QualityChecker._comparator_to_fn("==", 5)
    assert fn_eq(5) is True
    assert fn_eq(3) is False

    fn_lt = QualityChecker._comparator_to_fn("<", 0)
    assert fn_lt(-1) is True
    assert fn_lt(0) is False

    fn_other = QualityChecker._comparator_to_fn("!==", 0)
    assert fn_other(100) is False


def test_apply_quantile_returns_correct_tuple():
    df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
    qc = QualityChecker(df)
    # create node with children: column name, quantile q, comparator c, rhs
    node = mk_tree('quantile', ['col', '0.5', '>', '3'])
    result = qc._apply_quantile(node)
    # median is 3.0, comparator '>' 3 is False
    assert result[0] == 'quantile'
    assert result[1] is False

    node2 = mk_tree('quantile', ['col', '0.5', '==', '3'])
    result2 = qc._apply_quantile(node2)
    assert result2[1] is True


def test_apply_date_validation_parses_datetime_correctly():
    df = pd.DataFrame({'dates': ['2020-01-01', '2021-12-31']})
    qc = QualityChecker(df)
    node = mk_tree('is_date', ['dates'])
    name, passed = qc._apply_date_validation(node)
    assert name == 'is_date'
    assert passed is True


def test_apply_date_validation_returns_false_on_invalid_dates():
    df = pd.DataFrame({'bad_dates': ['not-a-date', 'also-bad']})
    qc = QualityChecker(df)
    node = mk_tree('is_date', ['bad_dates'])

    name, passed = qc._apply_date_validation(node)
    assert passed is False
    assert "Date parse error" in name


def test_apply_has_one_of_checks_values_ignore_case_false():
    df = pd.DataFrame({'col': ['A', 'B', 'C']})
    qc = QualityChecker(df)
    allowed_node_children = [
        mk_tree('value', []),
        mk_tree('value', [])
    ]
    # children[1] is a Tree with children containing 3 value leaves
    vals_node = mk_tree('values', [Tree('value', [type('Token', (), {'value': '"A"'})]),
                                   Tree('value', [type('Token', (), {'value': '"B"'})]),
                                   Tree('value', [type('Token', (), {'value': '"C"'})])])
    # Correct children format: [column_name str, a tree with children with Tokens, and optional ignore_case str]
    node = mk_tree('has_one_of', ['col', vals_node])
    result = qc._apply_has_one_of(node)
    assert result[0] == 'has_one_of'
    assert result[1] is True

    # test with a value not included
    vals_node2 = mk_tree('values', [Tree('value', [type('Token', (), {'value': '"A"'})]),
                                    Tree('value', [type('Token', (), {'value': '"B"'})])])
    node2 = mk_tree('has_one_of', ['col', vals_node2])
    result2 = qc._apply_has_one_of(node2)
    assert result2[1] is False


def test_apply_has_one_of_ignore_case_true():
    df = pd.DataFrame({'col': ['a', 'b', 'c']})
    qc = QualityChecker(df)

    vals_node = mk_tree('values', [Tree('value', [type('Token', (), {'value': '"A"'})]),
                                   Tree('value', [type('Token', (), {'value': '"B"'})]),
                                   Tree('value', [type('Token', (), {'value': '"C"'})])])
    node = mk_tree('has_one_of', ['col', vals_node, 'True'])
    res = qc._apply_has_one_of(node)
    assert res[1] is True


def test_apply_value_length_handles_ignore_nulls_false_and_true():
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'ccc']})
    qc = QualityChecker(df)
    # ignore_nulls False (len 3, comparator == 3)
    node = mk_tree('value_length', ['col', '==', '3'])
    res = qc._apply_value_length(node)
    assert res[1] is True

    # With different lengths, should be False
    df2 = pd.DataFrame({'col': ['a', 'bb', 'ccc']})
    qc2 = QualityChecker(df2)
    node2 = mk_tree('value_length', ['col', '==', '1'])
    res2 = qc2._apply_value_length(node2)
    assert res2[1] is False

    # ignore_nulls True, have one null value, it should be ignored
    df3 = pd.DataFrame({'col': ['aa', '', 'ccc', np.nan]})
    qc3 = QualityChecker(df3)
    node3 = mk_tree('value_length', ['col', 'True', '==', '2'])
    res3 = qc3._apply_value_length(node3)
    # The non-null trimmed have lengths 2 and 3 -> unique_lengths = 2 -> False for == 2
    assert res3[1] is False


def test_apply_percent_value_length_basic():
    # Prepare df where 50% have length == 3
    df = pd.DataFrame({'col': ['aaa', 'bbb', 'cccc', 'dddd']})
    qc = QualityChecker(df)
    # node format len 4: (col, percent, comparator?, rhs)
    node = mk_tree('percent_value_length', ['col', '50', 'False', '3'])
    res = qc._apply_percent_value_length(node)
    # (2/4)*100 = 50% >= 50% ⇒ True
    assert res[1] is True

    # test where percent threshold not met
    node2 = mk_tree('percent_value_length', ['col', '51', 'False', '3'])
    res2 = qc._apply_percent_value_length(node2)
    assert res2[1] is False


def test_apply_has_columns_ignore_case_false_and_true():
    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    qc = QualityChecker(df)

    vals_node = mk_tree('values', [type('Token', (), {'value': '"a"'})(),
                                   type('Token', (), {'value': '"B"'})()])

    # monkeypatch column names to preserve as list for test
    tokens = [type('Token', (), {'value': '"a"'})(), type('Token', (), {'value': '"b"'})()]

    vals_tree = mk_tree('values', tokens)

    node = mk_tree('has_columns', [vals_tree])
    res = qc._apply_has_columns(node)
    assert res[1] is False  # 'B' not present (case sensitive)

    node2 = mk_tree('has_columns', [vals_tree, 'True'])
    res2 = qc._apply_has_columns(node2)
    # ignore_case True -> 'b' matches 'B'
    assert res2[1] is True


def test_apply_check_coverage_various_checks():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [1, 1, 1],
        'c': [1, 2, 3],
        'd': ['2020-01-01', '2021-01-01', '2022-01-01'],
        'e': ['aa', 'bb', 'cc']
    })
    qc = QualityChecker(df)

    # Helper to construct check Tree with data attribute
    def check_node(data, children):
        return [mk_tree(data, children)]

    # row_count passes
    c1 = mk_tree('row_count', ['>', '2'])
    r1 = qc._apply_check([c1])
    assert r1 == ('row_count', True)

    # row_count fails
    c2 = mk_tree('row_count', ['>', '10'])
    r2 = qc._apply_check([c2])
    assert r2 == ('row_count', False)

    # has_columns true
    tokens = [type('Token', (), {'value': '"a"'})(), type('Token', (), {'value': '"b"'})()]
    vals_tree = mk_tree('values', tokens)
    c3 = mk_tree('has_columns', [vals_tree])
    r3 = qc._apply_check([c3])
    assert r3[1] is True

    # is_unique true in b (all 1's repeated? No, they are duplicates)
    # Actually, all equal means is_unique is False
    c4 = mk_tree('is_unique', ['b'])
    r4 = qc._apply_check([c4])
    assert r4[1] is False

    # not_null true (all columns non-null)
    c5 = mk_tree('not_null', ['a'])
    r5 = qc._apply_check([c5])
    assert r5[1] is True

    # is_positive true (a all > 0)
    c6 = mk_tree('is_positive', ['a'])
    r6 = qc._apply_check([c6])
    assert r6[1] is True

    # has_one_of true for column 'a' with correct values
    vals_node = mk_tree('values', [type('Token', (), {'value': '"1"'})(),
                                  type('Token', (), {'value': '"2"'})(),
                                  type('Token', (), {'value': '"3"'})()])
    c7 = mk_tree('has_one_of', ['a', vals_node])
    r7 = qc._apply_check([c7])
    assert r7[1] is True

    # quantile test
    c8 = mk_tree('quantile', ['a', '0.5', '>', '1'])
    r8 = qc._apply_check([c8])
    assert isinstance(r8, tuple)
    # median of a is 2, median > 1 is True
    assert r8[1] is True

    # is_date true
    c9 = mk_tree('is_date', ['d'])
    r9 = qc._apply_check([c9])
    assert r9[1] is True

    # value_length true (all lengths 2)
    c10 = mk_tree('value_length', ['e', '==', '2'])
    r10 = qc._apply_check([c10])
    assert r10[1] is True

    # percent_value_length true (e all length 2, so 100% have length 2)
    c11 = mk_tree('percent_value_length', ['e', '100', 'False', '2'])
    r11 = qc._apply_check([c11])
    assert r11[1] is True

    # unknown check triggers exception handled as GenieException
    unknown = mk_tree('not_implemented_check', [])
    with pytest.raises(GenieException):
        qc._apply_check([unknown])

    # keyerror in _apply_check returns error tuple
    # Pass column that doesn't exist for is_unique
    c_key_error = mk_tree('is_unique', ['nonexist'])
    name, res = qc._apply_check([c_key_error])
    assert res is False
    assert name.startswith("Key error")


def test_apply_predicates_runs_check_in_reverse_order():
    df = pd.DataFrame({'a': [1, 2, 3]})
    qc = QualityChecker(df)

    # Compose a code_block predicate with two checks
    c1 = mk_tree('row_count', ['>', '2'])
    c2 = mk_tree('is_positive', ['a'])
    code_block = mk_tree('code_block', [[c1], [c2]])
    predicates = mk_tree('predicates', [code_block])
    results = qc._apply_predicates(predicates.children)
    # Results are inserted at front, so the last check is first in results
    assert results[0][0] == 'is_positive'
    assert results[1][0] == 'row_count'


def test_run_raises_with_wrong_number_of_children():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    # construct a Tree with multiple children
    t = mk_tree('root', [mk_tree('predicates', []), mk_tree('predicates', [])])
    with pytest.raises(GenieException):
        qc._run(t)


def test_run_raises_on_unknown_instruction():
    df = pd.DataFrame({'a': [1]})
    qc = QualityChecker(df)
    t = mk_tree('root', [mk_tree('unknown_instruction', [])])
    with pytest.raises(GenieException):
        qc._run(t)


def test_validate_spec_returns_none_and_true_on_valid_spec():
    # Use a minimal correct DSL string for grammar parse
    valid_spec = "row_count > 0"
    err, valid = QualityChecker.validate_spec(valid_spec)
    assert valid is True
    assert err is None


def test_validate_spec_returns_error_and_false_on_invalid_spec():
    invalid_spec = "row_count >>> 0"
    err, valid = QualityChecker.validate_spec(invalid_spec)
    assert valid is False
    assert isinstance(err, str)
    assert len(err) > 0


def test_run_with_realistic_spec_runs_and_respects_ignore_column_case():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'b': [4, 5, 6],
        'c': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
    })
    qc = QualityChecker(df)

    # Provide a simple spec
    spec = """
    predicates {
      code_block {
        row_count > 2
        has_columns ["a", "b"] True
        is_date c
      }
    }
    """

    # Run ignoring column case, so 'A' matches 'a'
    results = qc.run(spec, ignore_column_case=True)
    # row_count > 2 true
    # has_columns 'a', 'b' True (case ignored, present)
    # is_date c is True
    assert isinstance(results, list)
    # check all True results for these three checks
    keys = [r[0] for r in results]
    assert "row_count" in keys
    assert "has_columns" in keys
    assert "is_date" in keys
    assert all(r[1] for r in results)


def test_run_seeds_randomness_independently(monkeypatch):
    # The code uses no random internally, but we test idempotency of run
    # Just call run twice and compare
    df = pd.DataFrame({'a': [1, 2, 3]})
    qc = QualityChecker(df)
    spec = "row_count > 0"
    results1 = qc.run(spec)
    results2 = qc.run(spec)
    assert results1 == results2


def test_ignore_column_case_modifies_df_columns(monkeypatch):
    df = pd.DataFrame({'A': [1]})
    qc = QualityChecker(df)
    _columns_before = df.columns.tolist()
    qc.run("row_count > 0", ignore_column_case=True)
    # columns are mapped to lowercase inside run
    assert all(isinstance(c, str) for c in qc._obj.columns)
    assert all(c.islower() for c in qc._obj.columns)
    # the original df columns are changed (by reference)
    assert list(df.columns) != _columns_before
