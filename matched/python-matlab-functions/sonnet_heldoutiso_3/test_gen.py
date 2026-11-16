import numpy as np
import pytest

from pymlfunc import sub2ind


def _arr(values):
    return np.array(values, dtype=int)


def test_origin_maps_to_zero():
    result = sub2ind((3, 4), _arr([0]), _arr([0]))
    assert np.array_equal(np.asarray(result), [0])


def test_last_element_maps_to_size_minus_one():
    result = sub2ind((3, 4), _arr([2]), _arr([3]))
    assert np.array_equal(np.asarray(result), [11])


def test_row_major_ordering_first_row():
    cols = _arr([0, 1, 2, 3])
    rows = _arr([0, 0, 0, 0])
    result = sub2ind((3, 4), rows, cols)
    assert np.array_equal(np.asarray(result), [0, 1, 2, 3])


def test_row_major_ordering_second_row_starts_at_ncols():
    result = sub2ind((3, 4), _arr([1]), _arr([0]))
    assert np.array_equal(np.asarray(result), [4])


def test_full_grid_matches_arange():
    shape = (3, 5)
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = np.asarray(sub2ind(shape, rows.ravel().copy(), cols.ravel().copy()))
    assert np.array_equal(result, np.arange(shape[0] * shape[1]))


def test_matches_numpy_ravel_multi_index():
    shape = (6, 7)
    rows = _arr([0, 1, 2, 3, 4, 5, 5, 2])
    cols = _arr([6, 0, 3, 2, 1, 6, 0, 5])
    expected = np.ravel_multi_index((rows, cols), shape)
    result = np.asarray(sub2ind(shape, rows.copy(), cols.copy()))
    assert np.array_equal(result, expected)


def test_single_row_matrix_index_equals_column():
    cols = _arr([0, 1, 2, 3, 4])
    rows = _arr([0, 0, 0, 0, 0])
    result = np.asarray(sub2ind((1, 5), rows, cols))
    assert np.array_equal(result, cols)


def test_single_column_matrix_index_equals_row():
    rows = _arr([0, 1, 2, 3])
    cols = _arr([0, 0, 0, 0])
    result = np.asarray(sub2ind((4, 1), rows, cols))
    assert np.array_equal(result, rows)


def test_square_matrix():
    result = np.asarray(sub2ind((3, 3), _arr([0, 1, 2, 2]), _arr([2, 1, 0, 2])))
    assert np.array_equal(result, [2, 4, 6, 8])


def test_result_length_matches_input_length():
    rows = _arr([0, 1, 2])
    cols = _arr([1, 1, 1])
    result = np.asarray(sub2ind((3, 3), rows, cols))
    assert result.shape == (3,)


def test_shape_as_list_gives_same_result_as_tuple():
    rows = _arr([1, 2])
    cols = _arr([2, 3])
    r_tuple = np.asarray(sub2ind((4, 5), rows.copy(), cols.copy()))
    r_list = np.asarray(sub2ind([4, 5], rows.copy(), cols.copy()))
    assert np.array_equal(r_tuple, r_list)


def test_inputs_values_preserved():
    rows = _arr([0, 1, 2])
    cols = _arr([3, 2, 1])
    rows_copy, cols_copy = rows.copy(), cols.copy()
    sub2ind((3, 4), rows, cols)
    assert np.array_equal(rows, rows_copy)
    assert np.array_equal(cols, cols_copy)


def test_indices_can_be_used_to_index_flattened_array():
    shape = (4, 5)
    data = np.arange(20).reshape(shape) * 10
    rows = _arr([0, 3, 2, 1])
    cols = _arr([4, 0, 2, 3])
    idx = np.asarray(sub2ind(shape, rows.copy(), cols.copy())).astype(int)
    assert np.array_equal(data.ravel()[idx], data[rows, cols])


def test_large_matrix_indices():
    shape = (1000, 2000)
    result = np.asarray(sub2ind(shape, _arr([999, 500]), _arr([1999, 10])))
    assert np.array_equal(result, [999 * 2000 + 1999, 500 * 2000 + 10])
