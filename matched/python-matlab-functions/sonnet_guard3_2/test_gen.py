import numpy as np
import pytest

from pymlfunc import sub2ind


def _as_array(values):
    return np.array(values, dtype=int)


def test_origin_maps_to_zero():
    result = sub2ind((3, 4), _as_array([0]), _as_array([0]))
    assert np.array_equal(np.asarray(result), np.array([0]))


def test_row_major_single_element():
    # index = row * ncols + col
    result = sub2ind((3, 4), _as_array([1]), _as_array([2]))
    assert np.array_equal(np.asarray(result), np.array([6]))


def test_last_element_maps_to_size_minus_one():
    result = sub2ind((3, 4), _as_array([2]), _as_array([3]))
    assert np.array_equal(np.asarray(result), np.array([11]))


def test_first_row_indices_equal_columns():
    cols = _as_array([0, 1, 2, 3])
    rows = _as_array([0, 0, 0, 0])
    result = sub2ind((3, 4), rows, cols)
    assert np.array_equal(np.asarray(result), cols)


def test_first_column_indices_are_multiples_of_ncols():
    rows = _as_array([0, 1, 2])
    cols = _as_array([0, 0, 0])
    result = sub2ind((3, 4), rows, cols)
    assert np.array_equal(np.asarray(result), np.array([0, 4, 8]))


def test_all_subscripts_give_consecutive_indices():
    shape = (3, 4)
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, np.arange(12))


def test_matches_numpy_ravel_multi_index():
    shape = (5, 7)
    rows = _as_array([0, 4, 2, 3, 1, 4])
    cols = _as_array([6, 0, 3, 5, 1, 6])
    expected = np.ravel_multi_index((rows, cols), shape)
    result = np.asarray(sub2ind(shape, rows.copy(), cols.copy()))
    assert np.array_equal(result, expected)


def test_indexing_flattened_array_matches_2d_indexing():
    np.random.seed(0)
    shape = (4, 6)
    data = np.arange(shape[0] * shape[1]).reshape(shape) * 3 + 1
    rows = _as_array([0, 1, 2, 3, 3, 0])
    cols = _as_array([5, 4, 3, 2, 0, 0])
    ind = np.asarray(sub2ind(shape, rows.copy(), cols.copy()))
    assert np.array_equal(data.ravel()[ind], data[rows, cols])


def test_square_matrix():
    result = sub2ind((3, 3), _as_array([0, 1, 2, 2]), _as_array([2, 1, 0, 2]))
    assert np.array_equal(np.asarray(result), np.array([2, 4, 6, 8]))


def test_single_column_matrix():
    rows = _as_array([0, 1, 2, 3])
    cols = _as_array([0, 0, 0, 0])
    result = sub2ind((4, 1), rows, cols)
    assert np.array_equal(np.asarray(result), rows)


def test_single_row_matrix():
    rows = _as_array([0, 0, 0])
    cols = _as_array([0, 1, 2])
    result = sub2ind((1, 3), rows, cols)
    assert np.array_equal(np.asarray(result), cols)


def test_output_length_matches_input_length():
    rows = _as_array([0, 1, 2, 1, 0])
    cols = _as_array([1, 1, 1, 0, 0])
    result = np.asarray(sub2ind((3, 2), rows, cols))
    assert result.shape == rows.shape


def test_two_dimensional_subscript_arrays_keep_shape():
    rows = _as_array([[0, 1], [2, 1]])
    cols = _as_array([[0, 1], [2, 0]])
    result = np.asarray(sub2ind((3, 3), rows, cols))
    assert result.shape == (2, 2)
    assert np.array_equal(result, np.array([[0, 4], [8, 3]]))


def test_inputs_are_not_modified():
    rows = _as_array([0, 1, 2])
    cols = _as_array([1, 2, 3])
    rows_copy = rows.copy()
    cols_copy = cols.copy()
    sub2ind((3, 4), rows, cols)
    assert np.array_equal(rows, rows_copy)
    assert np.array_equal(cols, cols_copy)


def test_shape_as_list_gives_same_result_as_tuple():
    rows = _as_array([0, 2, 1])
    cols = _as_array([3, 1, 2])
    r_tuple = np.asarray(sub2ind((3, 4), rows.copy(), cols.copy()))
    r_list = np.asarray(sub2ind([3, 4], rows.copy(), cols.copy()))
    assert np.array_equal(r_tuple, r_list)


def test_deterministic_repeated_calls():
    rows = _as_array([1, 0, 2])
    cols = _as_array([0, 3, 3])
    first = np.asarray(sub2ind((3, 4), rows.copy(), cols.copy()))
    second = np.asarray(sub2ind((3, 4), rows.copy(), cols.copy()))
    assert np.array_equal(first, second)
