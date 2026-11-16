import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _call(shape, rows, cols):
    return np.asarray(sub2ind(shape, np.asarray(rows), np.asarray(cols)))


def test_origin_is_zero():
    assert _call((3, 4), [0], [0])[0] == 0


def test_second_column_first_row():
    assert _call((3, 4), [0], [1])[0] == 1


def test_second_row_first_column_row_major():
    assert _call((3, 4), [1], [0])[0] == 4


def test_last_element():
    assert _call((3, 4), [2], [3])[0] == 11


def test_multiple_points():
    res = _call((3, 4), [0, 1, 2], [0, 1, 2])
    np.testing.assert_array_equal(res, [0, 5, 10])


def test_returns_same_length_as_input():
    rows = np.array([0, 1, 2, 1, 0])
    cols = np.array([3, 2, 1, 0, 0])
    res = _call((3, 4), rows, cols)
    assert res.shape == rows.shape


def test_matches_numpy_ravel_multi_index():
    shape = (5, 7)
    rows = np.array([0, 4, 2, 3, 1])
    cols = np.array([6, 0, 3, 5, 2])
    expected = np.ravel_multi_index((rows, cols), shape)
    np.testing.assert_array_equal(_call(shape, rows, cols), expected)


def test_indexes_flattened_array_correctly():
    arr = np.arange(20).reshape(4, 5) * 3
    rows = np.array([0, 1, 3, 2])
    cols = np.array([4, 2, 0, 3])
    idx = _call(arr.shape, rows, cols)
    np.testing.assert_array_equal(arr.ravel()[idx], arr[rows, cols])


def test_square_matrix():
    res = _call((4, 4), [1, 3], [2, 3])
    np.testing.assert_array_equal(res, [6, 15])


def test_single_row_matrix():
    res = _call((1, 5), [0, 0, 0], [0, 2, 4])
    np.testing.assert_array_equal(res, [0, 2, 4])


def test_single_column_matrix():
    res = _call((5, 1), [0, 2, 4], [0, 0, 0])
    np.testing.assert_array_equal(res, [0, 2, 4])


def test_all_indices_unique_and_cover_range():
    shape = (3, 4)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    res = _call(shape, rr.ravel(), cc.ravel())
    np.testing.assert_array_equal(np.sort(res), np.arange(12))


def test_out_of_bounds_does_not_return_valid_index():
    # Out of range subscripts must not map to a valid linear index
    # (implementation either flags with -1 or raises).
    try:
        res = _call((3, 4), [3], [0])
    except (IndexError, ValueError):
        return
    assert res[0] < 0 or res[0] >= 12 or res[0] == -1


def test_roundtrip_with_ind2sub():
    shape = (4, 6)
    rows = np.array([0, 1, 3, 2])
    cols = np.array([5, 0, 2, 4])
    ind = sub2ind(shape, rows, cols)
    r, c = ind2sub(shape, np.asarray(ind))
    np.testing.assert_array_equal(np.asarray(r), rows)
    np.testing.assert_array_equal(np.asarray(c), cols)


def test_result_is_integer_dtype():
    res = _call((3, 4), [0, 1], [2, 3])
    assert np.issubdtype(res.dtype, np.integer)
