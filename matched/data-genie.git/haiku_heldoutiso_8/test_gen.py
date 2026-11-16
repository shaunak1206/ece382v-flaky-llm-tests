import numpy as np
import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def _values(*names):
    """Build a lark 'values' subtree holding quoted STRING tokens."""
    return Tree('values', [Token('STRING', '"%s"' % n) for n in names])


def _columns(*names):
    """Build a lark 'columns' subtree holding quoted STRING tokens."""
    return Tree('columns', [Token('STRING', '"%s"' % n) for n in names])


def _checker(df):
    return QualityChecker(df)


# ---------------------------------------------------------------------------
# Construction and accessor registration
# ---------------------------------------------------------------------------

def test_constructor_rejects_empty_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_constructor_rejects_dataframe_with_only_all_nan_columns():
    df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [None, None]})
    with pytest.raises(AttributeError):
        QualityChecker(df)


def test_constructor_accepts_valid_frame_and_defaults_ignore_case_off():
    df = pd.DataFrame({'a': [1, 2]})
    qc = QualityChecker(df)
    assert qc.ignore_column_case is False
    assert qc._obj is df


def test_dataframe_accessor_is_registered():
    df = pd.DataFrame({'a': [1]})
    assert isinstance(df.dqc, QualityChecker)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_comparator_to_fn_supports_operators_and_defaults_to_false():
    gt = QualityChecker._comparator_to_fn('>', 5)
    eq = QualityChecker._comparator_to_fn('==', 5)
    lt = QualityChecker._comparator_to_fn('<', 5)
    unknown = QualityChecker._comparator_to_fn('!=', 5)

    assert gt(6) and not gt(5)
    assert eq(5) and not eq(4)
    assert lt(4) and not lt(5)
    assert unknown(5) is False
    assert unknown(0) is False


def test_treat_column_name_lowercases_only_when_ignore_case_enabled():
    qc = _checker(pd.DataFrame({'A': [1]}))
    assert qc._treat_column_name('MixedCase') == 'MixedCase'
    qc.ignore_column_case = True
    assert qc._treat_column_name('MixedCase') == 'mixedcase'


# ---------------------------------------------------------------------------
# Individual DSL checks
# ---------------------------------------------------------------------------

def test_apply_quantile_passes_and_fails_as_expected():
    qc = _checker(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))  # median is 3.0

    name, ok = qc._apply_quantile(Tree('quantile', ['a', '0.5', '>', '2']))
    assert name == 'quantile'
    assert ok

    name, ok = qc._apply_quantile(Tree('quantile', ['a', '0.5', '<', '2']))
    assert name == 'quantile'
    assert not ok


def test_apply_date_validation_accepts_valid_and_reports_invalid():
    good = _checker(pd.DataFrame({'d': ['2020-01-01', '2021-06-30']}))
    name, ok = good._apply_date_validation(Tree('is_date', ['d']))
    assert name == 'is_date'
    assert ok

    bad = _checker(pd.DataFrame({'d': ['2020-01-01', 'not a date']}))
    message, ok = bad._apply_date_validation(Tree('is_date', ['d']))
    assert ok is False
    assert message == 'Date parse error: is_date for d'


def test_apply_has_one_of_strict_matching():
    qc = _checker(pd.DataFrame({'a': ['x', 'y', 'x']}))

    name, ok = qc._apply_has_one_of(Tree('has_one_of', ['a', _values('x', 'y')]))
    assert name == 'has_one_of'
    assert ok

    qc2 = _checker(pd.DataFrame({'a': ['x', 'z']}))
    _, ok = qc2._apply_has_one_of(Tree('has_one_of', ['a', _values('x', 'y')]))
    assert not ok


def test_apply_has_one_of_ignore_case_matching():
    df = pd.DataFrame({'a': ['X', 'y']})
    qc = _checker(df)

    # Case-insensitive: allowed ['x', 'y'] covers ['X', 'y'].
    _, ok = qc._apply_has_one_of(Tree('has_one_of', ['a', _values('x', 'Y'), 'True']))
    assert ok

    # Case-sensitive: 'X' is not in ['x', 'Y'].
    _, ok = qc._apply_has_one_of(Tree('has_one_of', ['a', _values('x', 'Y'), 'False']))
    assert not ok


def test_apply_value_length_exact_length_without_null_handling():
    qc = _checker(pd.DataFrame({'a': ['abc', 'xyz']}))
    name, ok = qc._apply_value_length(Tree('value_length', ['a', '==', '3']))
    assert name == 'value_length'
    assert ok

    qc2 = _checker(pd.DataFrame({'a': ['abc', 'abcd']}))
    _, ok = qc2._apply_value_length(Tree('value_length', ['a', '==', '3']))
    assert not ok


def test_apply_value_length_ignore_nulls_drops_blank_and_null_values():
    df = pd.DataFrame({'a': ['abc', '', None]})

    # Blanks and None are excluded: only 'abc' remains, length 3.
    qc = _checker(df.copy())
    _, ok = qc._apply_value_length(Tree('value_length', ['a', 'True', '==', '3']))
    assert ok

    # Nulls included: None -> 'None' (len 4) and '' (len 0) break uniformity.
    qc2 = _checker(df.copy())
    _, ok = qc2._apply_value_length(Tree('value_length', ['a', 'False', '==', '3']))
    assert not ok


def test_apply_percent_value_length_respects_threshold():
    # Lengths are 3, 2, 3, 2 -> exactly 50% have length 3.
    df = pd.DataFrame({'a': ['abc', 'de', 'fgh', 'ij']})

    qc = _checker(df.copy())
    name, ok = qc._apply_percent_value_length(Tree('percent_value_length', ['a', '50', 'x', '3']))
    assert name == 'percent_value_length'
    assert ok

    qc = _checker(df.copy())
    _, ok = qc._apply_percent_value_length(Tree('percent_value_length', ['a', '60', 'x', '3']))
    assert not ok


def test_apply_percent_value_length_ignore_nulls_changes_pass_rate():
    # With nulls ignored, rows containing blank/None are dropped; the
    # denominator is still the full row count (4), so 1/4 = 25%.
    df = pd.DataFrame({'a': ['abc', '', None, 'abcd']})

    qc = _checker(df.copy())
    _, ok = qc._apply_percent_value_length(
        Tree('percent_value_length', ['a', '20', 'True', '==', '3']))
    assert ok

    qc = _checker(df.copy())
    _, ok = qc._apply_percent_value_length(
        Tree('percent_value_length', ['a', '30', 'True', '==', '3']))
    assert not ok


def test_apply_has_columns_strict_matching():
    qc = _checker(pd.DataFrame({'A': [1], 'B': [2]}))

    name, ok = qc._apply_has_columns(Tree('has_columns', [_columns('A')]))
    assert name == 'has_columns'
    assert ok

    _, ok = qc._apply_has_columns(Tree('has_columns', [_columns('A', 'C')]))
    assert not ok


def test_apply_has_columns_ignore_case_matching():
    qc = _checker(pd.DataFrame({'A': [1], 'B': [2]}))

    _, ok = qc._apply_has_columns(Tree('has_columns', [_columns('a'), 'True']))
    assert ok

    _, ok = qc._apply_has_columns(Tree('has_columns', [_columns('a'), 'False']))
    assert not ok


# ---------------------------------------------------------------------------
# _apply_check dispatch
# ---------------------------------------------------------------------------

def test_apply_check_row_count_uses_comparator():
    qc = _checker(pd.DataFrame({'a': [1, 2, 3]}))

    name, ok = qc._apply_check([Tree('row_count', ['>', '2'])])
    assert name == 'row_count'
    assert ok

    _, ok = qc._apply_check([Tree('row_count', ['<', '2'])])
    assert not ok


def test_apply_check_is_unique_and_not_null():
    unique_df = _checker(pd.DataFrame({'a': [1, 2]}))
    name, ok = unique_df._apply_check([Tree('is_unique', ['a'])])
    assert name == 'is_unique'
    assert ok

    dup_df = _checker(pd.DataFrame({'a': [1, 1]}))
    _, ok = dup_df._apply_check([Tree('is_unique', ['a'])])
    assert not ok

    null_df = _checker(pd.DataFrame({'a': [1.0, np.nan]}))
    name, ok = null_df._apply_check([Tree('not_null', ['a'])])
    assert name == 'not_null'
    assert not ok


def test_apply_check_unknown_check_raises_genie_exception():
    qc = _checker(pd.DataFrame({'a': [1]}))
    with pytest.raises(GenieException):
        qc._apply_check([Tree('no_such_check', ['a'])])


def test_apply_check_missing_column_returns_key_error_message():
    qc = _checker(pd.DataFrame({'a': [1, 2]}))
    message, ok = qc._apply_check([Tree('is_unique', ['missing'])])
    assert ok is False
    assert message == 'Key error: is_unique for missing'


# ---------------------------------------------------------------------------
# Predicates and top-level run flow
# ---------------------------------------------------------------------------

def test_apply_predicates_collects_results_in_reverse_order():
    qc = _checker(pd.DataFrame({'a': [1, 2]}))
    predicates = [
        Tree('code_block', [
            Tree('check', [Tree('is_unique', ['a'])]),
            Tree('check', [Tree('not_null', ['a'])]),
        ])
    ]
    results = qc._apply_predicates(predicates)
    # Each result is inserted at index 0, so the last check comes first.
    assert [name for name, _ in results] == ['not_null', 'is_unique']
    assert all(ok for _, ok in results)


def test_run_dispatches_predicates_instruction():
    qc = _checker(pd.DataFrame({'a': [1, 2, 3]}))
    tree = Tree('start', [
        Tree('predicates', [
            Tree('code_block', [
                Tree('check', [Tree('row_count', ['>', '0'])]),
            ]),
        ]),
    ])
    results = qc._run(tree)
    assert len(results) == 1
    name, ok = results[0]
    assert name == 'row_count'
    assert ok


def test_run_rejects_tree_with_multiple_top_level_children():
    qc = _checker(pd.DataFrame({'a': [1]}))
    tree = Tree('start', [Tree('a', []), Tree('b', [])])
    with pytest.raises(GenieException):
        qc._run(tree)


def test_run_rejects_unknown_top_level_instruction():
    qc = _checker(pd.DataFrame({'a': [1]}))
    tree = Tree('start', [Tree('other', [])])
    with pytest.raises(GenieException):
        qc._run(tree)


def test_run_with_ignore_case_lowercases_columns_before_checks(monkeypatch):
    tree = Tree('start', [
        Tree('predicates', [
            Tree('code_block', [
                Tree('check', [Tree('is_unique', ['a'])]),
            ]),
        ]),
    ])
    monkeypatch.setattr(QualityChecker, '_parse_spec', staticmethod(lambda spec: tree))

    qc = _checker(pd.DataFrame({'A': [1, 2]}))
    results = qc.run('ignored spec', ignore_column_case=True)

    assert list(qc._obj.columns) == ['a']
    assert results == [('is_unique', True)]


def test_validate_spec_reports_parse_errors(monkeypatch):
    def _raise(spec):
        raise ValueError('boom')

    monkeypatch.setattr(QualityChecker, '_parse_spec', staticmethod(_raise))
    assert QualityChecker.validate_spec('anything') == ('boom', False)


def test_validate_spec_reports_success_for_parsable_spec(monkeypatch):
    monkeypatch.setattr(QualityChecker, '_parse_spec',
                        staticmethod(lambda spec: Tree('start', [])))
    assert QualityChecker.validate_spec('anything') == (None, True)
