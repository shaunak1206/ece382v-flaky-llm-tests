import numpy as np
import pytest

import pymlfunc
from pymlfunc import sub2ind


def _arr(values):
    return np.array(values, dtype=int)


def test_importable_from_package():
    assert callable(pymlfunc.sub2ind)


def test_basic_row_major_3x4():
    shape = (3, 4)
    rows = _arr([0, 1, 2])
    cols = _arr([0, 1, 2])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [0, 5, 10]


def test_origin_maps_to_zero():
    result = np.asarray(sub2ind((5, 6), _arr([0]), _arr([0])))
    assert result.tolist() == [0]


def test_last_element_maps_to_size_minus_one():
    shape = (4, 7)
    result = np.asarray(sub2ind(shape, _arr([3]), _arr([6])))
    assert result.tolist() == [shape[0] * shape[1] - 1]


def test_first_row_equals_column_indices():
    shape = (3, 5)
    cols = _arr([0, 1, 2, 3, 4])
    rows = _arr([0, 0, 0, 0, 0])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_single_row_matrix():
    shape = (1, 5)
    cols = _arr([4, 2, 0])
    rows = _arr([0, 0, 0])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [4, 2, 0]


def test_single_column_matrix():
    shape = (4, 1)
    rows = _arr([3, 1, 0])
    cols = _arr([0, 0, 0])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [3, 1, 0]


def test_square_matrix():
    shape = (3, 3)
    rows = _arr([0, 1, 2, 2])
    cols = _arr([2, 0, 1, 2])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [2, 3, 7, 8]


def test_shape_as_list():
    result = np.asarray(sub2ind([3, 4], _arr([1, 2]), _arr([3, 0])))
    assert result.tolist() == [7, 8]


def test_matches_numpy_ravel_multi_index_for_all_cells():
    shape = (4, 6)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    result = np.asarray(sub2ind(shape, rows, cols))
    expected = np.ravel_multi_index((rows, cols), shape)
    assert np.array_equal(result, expected)


def test_all_cells_give_unique_consecutive_indices():
    shape = (5, 3)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = np.asarray(sub2ind(shape, rr.ravel(), cc.ravel()))
    assert sorted(result.tolist()) == list(range(15))


def test_indexing_flattened_array_matches_2d_indexing():
    data = np.arange(35).reshape(5, 7) * 3
    rows = _arr([0, 2, 4, 3])
    cols = _arr([6, 1, 0, 5])
    ind = np.asarray(sub2ind(data.shape, rows, cols))
    assert np.array_equal(data.ravel()[ind], data[rows, cols])


def test_output_length_matches_input_length():
    rows = _arr([0, 1, 2, 3, 4, 0])
    cols = _arr([1, 1, 1, 1, 1, 1])
    result = np.asarray(sub2ind((5, 2), rows, cols))
    assert result.shape == rows.shape


def test_preserves_2d_input_shape():
    rows = _arr([[0, 1], [2, 3]])
    cols = _arr([[0, 1], [2, 3]])
    result = np.asarray(sub2ind((4, 4), rows, cols))
    assert result.shape == (2, 2)
    assert result.tolist() == [[0, 5], [10, 15]]


def test_result_is_integer_valued():
    result = np.asarray(sub2ind((3, 4), _arr([1, 2]), _arr([1, 2])))
    assert np.issubdtype(result.dtype, np.integer)


def test_large_shape():
    shape = (1000, 2000)
    rows = _arr([999, 500])
    cols = _arr([1999, 10])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.tolist() == [999 * 2000 + 1999, 500 * 2000 + 10]


def test_does_not_modify_inputs():
    rows = _arr([0, 1, 2])
    cols = _arr([3, 2, 1])
    rows_copy = rows.copy()
    cols_copy = cols.copy()
    sub2ind((3, 4), rows, cols)
    assert np.array_equal(rows, rows_copy)
    assert np.array_equal(cols, cols_copy)
