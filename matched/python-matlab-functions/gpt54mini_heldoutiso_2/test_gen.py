import pytest
import numpy as np

from pymlfunc import sub2ind


def _call_sub2ind(shape, *subs):
    """
    Helper to accommodate small signature variations while keeping the tests
    focused on MATLAB-style sub2ind behavior.
    """
    # Most likely signatures:
    #   sub2ind(shape, i, j, ...)
    #   sub2ind(shape, (i, j, ...))
    try:
        return sub2ind(shape, *subs)
    except TypeError:
        if len(subs) == 1 and isinstance(subs[0], (tuple, list, np.ndarray)):
            return sub2ind(shape, subs[0])
        raise


def test_sub2ind_first_element_is_one_based():
    assert _call_sub2ind((3, 4), 1, 1) == 1


def test_sub2ind_advances_down_rows_before_columns():
    assert _call_sub2ind((3, 4), 2, 1) == 2
    assert _call_sub2ind((3, 4), 3, 1) == 3
    assert _call_sub2ind((3, 4), 1, 2) == 4
    assert _call_sub2ind((3, 4), 2, 2) == 5


def test_sub2ind_matches_matlab_column_major_order_for_entire_first_column():
    shape = (5, 2)
    expected = [1, 2, 3, 4, 5]
    actual = [_call_sub2ind(shape, r, 1) for r in range(1, 6)]
    assert actual == expected


def test_sub2ind_matches_matlab_column_major_order_for_second_column():
    shape = (5, 2)
    expected = [6, 7, 8, 9, 10]
    actual = [_call_sub2ind(shape, r, 2) for r in range(1, 6)]
    assert actual == expected


def test_sub2ind_last_element_of_matrix():
    assert _call_sub2ind((3, 4), 3, 4) == 12


def test_sub2ind_supports_three_dimensions():
    # MATLAB/Fortran-style linear indexing:
    # idx = i + (j-1)*m + (k-1)*m*n
    assert _call_sub2ind((2, 3, 4), 2, 3, 4) == 24
    assert _call_sub2ind((2, 3, 4), 1, 1, 2) == 7


def test_sub2ind_supports_higher_dimensions():
    assert _call_sub2ind((2, 2, 2, 2), 2, 2, 2, 2) == 16
    assert _call_sub2ind((2, 2, 2, 2), 1, 2, 1, 2) == 11


def test_sub2ind_accepts_tuple_or_list_of_subscripts():
    expected = 11
    assert _call_sub2ind((3, 4), (2, 4)) == expected
    assert _call_sub2ind((3, 4), [2, 4]) == expected


def test_sub2ind_accepts_numpy_array_of_subscripts():
    result = _call_sub2ind((3, 4), np.array([2, 4]))
    assert result == 11


def test_sub2ind_returns_python_int_for_scalar_inputs():
    result = _call_sub2ind((3, 4), 2, 4)
    assert isinstance(result, (int, np.integer))


def test_sub2ind_vectorized_row_inputs_produce_expected_linear_indices():
    shape = (3, 4)
    rows = np.array([1, 2, 3])
    cols = np.array([1, 2, 3])
    result = _call_sub2ind(shape, rows, cols)
    assert np.array_equal(np.asarray(result), np.array([1, 5, 9]))


def test_sub2ind_vectorized_column_inputs_produce_expected_linear_indices():
    shape = (3, 4)
    rows = np.array([1, 1, 1, 1])
    cols = np.array([1, 2, 3, 4])
    result = _call_sub2ind(shape, rows, cols)
    assert np.array_equal(np.asarray(result), np.array([1, 4, 7, 10]))


def test_sub2ind_raises_for_zero_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 0, 1)


def test_sub2ind_raises_for_negative_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), -1, 1)


def test_sub2ind_raises_for_out_of_bounds_row():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 4, 1)


def test_sub2ind_raises_for_out_of_bounds_column():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 5)


def test_sub2ind_raises_for_wrong_number_of_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1)

    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 2, 3)
