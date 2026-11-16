import pandas as pd
import pytest
from lark import Token, Tree

from genie_pkg import GenieException
from genie_pkg.dqc import QualityChecker


def tok(value):
    return Token('STRING', value)


def make_checker(data):
    return QualityChecker(pd.DataFrame(data))


# --- construction / validation ---

def test_init_rejects_empty_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame())


def test_init_rejects_all_nan_dataframe():
    with pytest.raises(AttributeError):
        QualityChecker(pd.DataFrame({'a': [None, None]}))


def test_init_accepts_nonempty_dataframe_with_defaults():
    checker = make_checker({'a': [1, 2]})
    assert checker.ignore_column_case is False


def test_validate_spec_rejects_garbage_input():
    message, ok = QualityChecker.validate_spec("!!! not a valid spec !!!")
    assert ok is False
    assert isinstance(message, str)


# --- helpers ---

@pytest.mark.parametrize("comparator,rhs,lhs,expected", [
    (">", 2, 3, True),
    (">", 2, 2, False),
    ("==", 2, 2, True),
    ("==", 2, 3, False),
    ("<", 2, 1, True),
    ("<", 2, 2, False),
    ("!=", 2, 2, False),  # unsupported comparator always returns False
])
def test_comparator_to_fn(comparator, rhs, lhs, expected):
    fn = QualityChecker._comparator_to_fn(comparator, rhs)
    assert fn(lhs) is expected


def test_treat_column_name_preserves_case_by_default():
    checker = make_checker({'Abc': [1]})
    assert checker._treat_column_name('Abc') == 'Abc'


def test_treat_column_name_lowercases_when_ignoring_case():
    checker = make_checker({'Abc': [1]})
    checker.ignore_column_case = True
    assert checker._treat_column_name('Abc') == 'abc'


# --- individual checks via _apply_* methods ---

def test_apply_quantile_passes():
    checker = make_checker({'x': [1, 2, 3, 4, 5]})
    node = Tree('quantile', ['x', '0.5', '>', '2'])
    name, result = checker._apply_quantile(node)
    assert name == 'quantile'
    assert bool(result) is True


def test_apply_quantile_fails():
    checker = make_checker({'x': [1, 2, 3, 4, 5]})
    node = Tree('quantile', ['x', '0.5', '<', '2'])
    name, result = checker._apply_quantile(node)
    assert name == 'quantile'
    assert bool(result) is False


def test_apply_date_validation_valid_dates():
    checker = make_checker({'d': ['2020-01-01', '2021-06-15']})
    name, result = checker._apply_date_validation(Tree('is_date', ['d']))
    assert name == 'is_date'
    assert result is True


def test_apply_date_validation_invalid_dates_reports_error():
    checker = make_checker({'d': ['not a date']})
    message, result = checker._apply_date_validation(Tree('is_date', ['d']))
    assert result is False
    assert message.startswith('Date parse error')
    assert 'is_date' in message


def test_apply_has_one_of_all_allowed():
    checker = make_checker({'c': ['a', 'b', 'a']})
    node = Tree('has_one_of', ['c', Tree('values', [tok('"a"'), tok('"b"')])])
    name, result = checker._apply_has_one_of(node)
    assert name == 'has_one_of'
    assert result is True


def test_apply_has_one_of_disallowed_value():
    checker = make_checker({'c': ['a', 'z']})
    node = Tree('has_one_of', ['c', Tree('values', [tok('"a"'), tok('"b"')])])
    _, result = checker._apply_has_one_of(node)
    assert result is False


def test_apply_has_one_of_ignore_case():
    checker = make_checker({'c': ['A', 'b']})
    node = Tree('has_one_of', ['c', Tree('values', [tok('"a"'), tok('"B"')]), 'True'])
    _, result = checker._apply_has_one_of(node)
    assert result is True


def test_apply_value_length_fixed_length_passes():
    checker = make_checker({'c': ['abc', 'def']})
    node = Tree('value_length', ['c', '==', '3'])
    name, result = checker._apply_value_length(node)
    assert name == 'value_length'
    assert result is True


def test_apply_value_length_mismatch_fails():
    checker = make_checker({'c': ['abc', 'de']})
    node = Tree('value_length', ['c', '==', '3'])
    _, result = checker._apply_value_length(node)
    assert result is False


def test_apply_value_length_ignore_nulls_excludes_blank_values():
    # Whitespace-only and null values are dropped, leaving only 'ab' (length 2).
    checker = make_checker({'c': ['ab', '   ', None]})
    node = Tree('value_length', ['c', 'True', '>', '1'])
    _, result = checker._apply_value_length(node)
    assert result is True


def test_apply_percent_value_length_meets_threshold():
    checker = make_checker({'c': ['ab', 'abc', 'ab', 'x']})
    node = Tree('percent_value_length', ['c', '50', '==', '2'])
    name, result = checker._apply_percent_value_length(node)
    assert name == 'percent_value_length'
    assert bool(result) is True


def test_apply_percent_value_length_below_threshold():
    checker = make_checker({'c': ['ab', 'abc', 'abc', 'x']})
    node = Tree('percent_value_length', ['c', '75', '==', '2'])
    _, result = checker._apply_percent_value_length(node)
    assert bool(result) is False


def test_apply_has_columns_present():
    checker = make_checker({'a': [1], 'b': [2], 'c': [3]})
    node = Tree('has_columns', [Tree('values', [tok('"a"'), tok('"b"')])])
    name, result = checker._apply_has_columns(node)
    assert name == 'has_columns'
    assert result is True


def test_apply_has_columns_missing():
    checker = make_checker({'a': [1]})
    node = Tree('has_columns', [Tree('values', [tok('"a"'), tok('"z"')])])
    _, result = checker._apply_has_columns(node)
    assert result is False


def test_apply_has_columns_ignore_case():
    checker = make_checker({'a': [1]})
    node = Tree('has_columns', [Tree('values', [tok('"A"')]), 'True'])
    _, result = checker._apply_has_columns(node)
    assert result is True


# --- dispatch through _apply_check ---

def test_apply_check_row_count():
    checker = make_checker({'c': [1, 2, 3]})
    name, result = checker._apply_check([Tree('row_count', ['>', '2'])])
    assert name == 'row_count'
    assert result is True


@pytest.mark.parametrize("values,expected", [
    ([1, 2, 3], True),
    ([1, 1, 2], False),
])
def test_apply_check_is_unique(values, expected):
    checker = make_checker({'c': values})
    name, result = checker._apply_check([Tree('is_unique', ['c'])])
    assert name == 'is_unique'
    assert bool(result) is expected


def test_apply_check_not_null():
    checker = make_checker({'c': [1, None]})
    name, result = checker._apply_check([Tree('not_null', ['c'])])
    assert name == 'not_null'
    assert bool(result) is False


def test_apply_check_is_positive():
    checker = make_checker({'c': [1, 2, -3]})
    name, result = checker._apply_check([Tree('is_positive', ['c'])])
    assert name == 'is_positive'
    assert bool(result) is False


def test_apply_check_missing_column_returns_key_error_message():
    checker = make_checker({'c': [1]})
    message, result = checker._apply_check([Tree('is_unique', ['missing'])])
    assert result is False
    assert message.startswith('Key error')
    assert 'missing' in message


def test_apply_check_unknown_check_raises():
    checker = make_checker({'c': [1]})
    with pytest.raises(GenieException):
        checker._apply_check([Tree('not_a_real_check', ['c'])])


# --- top-level tree walking ---

def test_apply_predicates_orders_results_last_check_first():
    checker = make_checker({'c': [1, 1]})
    predicates = [
        Tree('code_block', [
            Tree('check', [Tree('row_count', ['==', '2'])]),
            Tree('check', [Tree('is_unique', ['c'])]),
        ])
    ]
    results = checker._apply_predicates(predicates)
    # Results are inserted at the front, so the last check appears first.
    assert [name for name, _ in results] == ['is_unique', 'row_count']
    assert bool(results[0][1]) is False
    assert bool(results[1][1]) is True


def test_run_evaluates_full_tree_without_grammar():
    checker = make_checker({'c': [1, 2, 3]})
    tree = Tree('start', [
        Tree('predicates', [
            Tree('code_block', [
                Tree('check', [Tree('row_count', ['==', '3'])]),
            ])
        ])
    ])
    results = checker._run(tree)
    assert results == [('row_count', True)]


def test_run_rejects_tree_with_multiple_children():
    checker = make_checker({'c': [1]})
    tree = Tree('start', [Tree('a', []), Tree('b', [])])
    with pytest.raises(GenieException):
        checker._run(tree)


def test_run_rejects_unknown_instruction():
    checker = make_checker({'c': [1]})
    tree = Tree('start', [Tree('not_predicates', [])])
    with pytest.raises(GenieException):
        checker._run(tree)
