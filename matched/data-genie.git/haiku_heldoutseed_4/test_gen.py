import numpy as np
import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def T(data, *children):
    """Build a lark Tree node concisely."""
    return Tree(data, list(children))


def V(value):
    """Build a lark Token carrying a quoted DSL string value."""
    return Token('STRING', value)


def apply_check(df, node, ignore_column_case=False):
    """Run a single check node through QualityChecker._apply_check."""
    checker = QualityChecker(df)
    checker.ignore_column_case = ignore_column_case
    return checker._apply_check([node])


def test_empty_and_all_nan_dataframes_are_rejected():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({'a': [np.nan, np.nan]}))


def test_valid_dataframe_is_stored_with_default_case_setting():
    df = pd.DataFrame({'a': [1, 2]})
    checker = QualityChecker(df)
    assert checker._obj is df
    assert checker.ignore_column_case is False


def test_comparator_to_fn_supports_gt_eq_lt():
    assert QualityChecker._comparator_to_fn('>', 2)(3) is True
    assert QualityChecker._comparator_to_fn('>', 2)(2) is False
    assert QualityChecker._comparator_to_fn('==', 2)(2) is True
    assert QualityChecker._comparator_to_fn('==', 2)(5) is False
    assert QualityChecker._comparator_to_fn('<', 2)(1) is True
    assert QualityChecker._comparator_to_fn('<', 2)(2) is False


def test_comparator_to_fn_unsupported_operator_always_false():
    fn = QualityChecker._comparator_to_fn('>=', 2)
    assert fn(100) is False
    assert fn(-100) is False


def test_treat_column_name_respects_ignore_case_flag():
    checker = QualityChecker(pd.DataFrame({'a': [1]}))
    assert checker._treat_column_name('AbC') == 'AbC'
    checker.ignore_column_case = True
    assert checker._treat_column_name('AbC') == 'abc'


def test_row_count_check():
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert apply_check(df, T('row_count', '>', '2'))[1] is True
    assert apply_check(df, T('row_count', '==', '5'))[1] is False
    assert apply_check(df, T('row_count', '<', '2'))[1] is False


def test_is_unique_check():
    assert apply_check(pd.DataFrame({'a': [1, 2, 3]}), T('is_unique', 'a'))[0] == 'is_unique'
    assert apply_check(pd.DataFrame({'a': [1, 2, 3]}), T('is_unique', 'a'))[1]
    assert not apply_check(pd.DataFrame({'a': [1, 2, 2]}), T('is_unique', 'a'))[1]


def test_not_null_check():
    assert apply_check(pd.DataFrame({'a': [1, 2]}), T('not_null', 'a'))[1]
    assert not apply_check(pd.DataFrame({'a': [1, None]}), T('not_null', 'a'))[1]


def test_is_positive_check():
    assert apply_check(pd.DataFrame({'a': [1, 2]}), T('is_positive', 'a'))[1]
    assert not apply_check(pd.DataFrame({'a': [1, 0]}), T('is_positive', 'a'))[1]
    assert not apply_check(pd.DataFrame({'a': [1, -3]}), T('is_positive', 'a'))[1]


def test_has_one_of_exact_match():
    df_ok = pd.DataFrame({'a': ['x', 'x']})
    df_bad = pd.DataFrame({'a': ['x', 'y']})
    node = T('has_one_of', 'a', T('vals', V('"x"')))
    assert apply_check(df_ok, node)[1]
    assert not apply_check(df_bad, node)[1]


def test_has_one_of_ignore_case():
    df = pd.DataFrame({'a': ['x', 'X']})
    node = T('has_one_of', 'a', T('vals', V('"x"')), 'True')
    assert apply_check(df, node)[1]
    strict = T('has_one_of', 'a', T('vals', V('"x"')), 'False')
    assert not apply_check(df, strict)[1]


def test_has_columns_requires_all_expected_columns():
    df = pd.DataFrame({'a': [1], 'b': [2]})
    assert apply_check(df, T('has_columns', T('cols', V('"a"'), V('"b"'))))[1]
    assert not apply_check(df, T('has_columns', T('cols', V('"a"'), V('"c"'))))[1]


def test_has_columns_ignore_case():
    df = pd.DataFrame({'A': [1], 'B': [2]})
    node = T('has_columns', T('cols', V('"a"'), V('"b"')), 'True')
    assert apply_check(df, node)[1]
    strict = T('has_columns', T('cols', V('"a"'), V('"b"')))
    assert not apply_check(df, strict)[1]


def test_quantile_check_uses_median_and_comparator():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})  # median is 3.0
    assert apply_check(df, T('quantile', 'x', '0.5', '>', '2'))[1] is True
    assert apply_check(df, T('quantile', 'x', '0.5', '<', '2'))[1] is False
    assert apply_check(df, T('quantile', 'x', '0.5', '>=', '2'))[1] is False


def test_is_date_check_accepts_dates_and_rejects_garbage():
    good = pd.DataFrame({'d': ['2020-01-01', '2021-12-31']})
    bad = pd.DataFrame({'d': ['2020-01-01', 'not a date']})
    assert apply_check(good, T('is_date', 'd'))[1] is True

    message, ok = apply_check(bad, T('is_date', 'd'))
    assert ok is False
    assert message.startswith('Date parse error')
    assert 'is_date' in message and 'd' in message


def test_value_length_without_ignore_nulls():
    df_ok = pd.DataFrame({'v': ['abc', 'def']})
    df_mixed = pd.DataFrame({'v': ['abc', 'de']})
    assert apply_check(df_ok, T('value_length', 'v', '==', '3'))[1]
    assert not apply_check(df_mixed, T('value_length', 'v', '==', '3'))[1]


def test_value_length_ignore_nulls_skips_blank_and_missing_values():
    df = pd.DataFrame({'v': ['abc', '', None]})
    # Without ignoring nulls, '' (len 0) and None (len 4) break uniformity.
    assert not apply_check(df, T('value_length', 'v', '==', '3'))[1]
    # Ignoring nulls leaves only 'abc', which has length 3.
    assert apply_check(df, T('value_length', 'v', 'True', '==', '3'))[1]


def test_percent_value_length_threshold():
    df = pd.DataFrame({'v': ['abc', 'def', 'gh']})  # 2 of 3 rows have length 3
    # 4-child form: [column, percent, unused, rhs]
    assert apply_check(df, T('percent_value_length', 'v', '60', '==', '3'))[1] is True
    assert apply_check(df, T('percent_value_length', 'v', '80', '==', '3'))[1] is False


def test_unknown_check_raises_genie_exception():
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(GenieException):
        apply_check(df, T('not_a_real_check', 'a'))


def test_missing_column_returns_key_error_result():
    df = pd.DataFrame({'a': [1, 2]})
    message, ok = apply_check(df, T('is_unique', 'missing'))
    assert ok is False
    assert message.startswith('Key error')
    assert 'missing' in message


def test_apply_predicates_inserts_results_in_reverse_order():
    df = pd.DataFrame({'a': [1, 1]})
    checker = QualityChecker(df)
    predicates = [
        T('code_block',
          T('stmt', T('row_count', '>', '0')),
          T('stmt', T('is_unique', 'a'))),
    ]
    results = checker._apply_predicates(predicates)
    assert [name for name, _ in results] == ['is_unique', 'row_count']
    assert results[0][1] is False
    assert results[1][1] is True


def test_run_rejects_malformed_tree_and_unknown_instruction():
    df = pd.DataFrame({'a': [1]})
    checker = QualityChecker(df)

    with pytest.raises(GenieException):
        checker._run(T('start', T('predicates', []), T('predicates', [])))

    with pytest.raises(GenieException):
        checker._run(T('start', T('something_else', [])))


def test_validate_spec_reports_invalid_grammar():
    message, ok = QualityChecker.validate_spec('@@@ definitely not valid ))) @@@')
    assert ok is False
    assert isinstance(message, str)
    assert message


def test_run_with_ignore_column_case_lowercases_columns(monkeypatch):
    tree = T('start',
             T('predicates',
               T('code_block',
                 T('stmt', T('is_unique', 'A')))))
    monkeypatch.setattr(QualityChecker, '_parse_spec', staticmethod(lambda spec: tree))

    df = pd.DataFrame({'A': [1, 2, 3]})
    checker = QualityChecker(df)
    results = checker.run('ignored spec', ignore_column_case=True)

    assert checker.ignore_column_case is True
    assert list(checker._obj.columns) == ['a']
    assert results == [('is_unique', True)] or [(n, bool(v)) for n, v in results] == [('is_unique', True)]
