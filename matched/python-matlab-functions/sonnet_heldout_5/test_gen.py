import numpy as np
import pytest

from pymlfunc import sub2ind


def _arr(x):
    return np.asarray(x)


def test_origin_maps_to_zero():
    result = _arr(sub2ind((3, 4), np.array([0]), np.array([0])))
    assert result.tolist() == [0]


def test_first_row_indices_equal_columns():
    rows = np.array([0, 0, 0, 0])
    cols = np.array([0, 1, 2, 3])
    result = _arr(sub2ind((3, 4), rows, cols))
    assert result.tolist() == [0, 1, 2, 3]


def test_first_column_indices_step_by_ncols():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 0, 0])
    result = _arr(sub2ind((3, 4), rows, cols))
    assert result.tolist() == [0, 4, 8]


def test_last_element_maps_to_size_minus_one():
    result = _arr(sub2ind((3, 4), np.array([2]), np.array([3])))
    assert result.tolist() == [11]


def test_row_major_formula_single_point():
    result = _arr(sub2ind((5, 7), np.array([2]), np.array([3])))
    assert result.tolist() == [2 * 7 + 3]


def test_matches_numpy_ravel_multi_index_on_full_grid():
    shape = (4, 6)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    result = _arr(sub2ind(shape, rows.copy(), cols.copy()))
    expected = np.ravel_multi_index((rows, cols), shape)
    assert np.array_equal(result, expected)


def test_full_grid_gives_consecutive_indices():
    shape = (3, 5)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = _arr(sub2ind(shape, rr.ravel().copy(), cc.ravel().copy()))
    assert result.tolist() == list(range(15))


def test_output_length_matches_input_length():
    rows = np.array([0, 1, 2, 1, 0])
    cols = np.array([1, 2, 3, 0, 0])
    result = _arr(sub2ind((3, 4), rows, cols))
    assert result.shape == (5,)


def test_indexing_flattened_array_returns_same_element_as_2d_indexing():
    data = np.arange(30).reshape(5, 6) * 3
    rows = np.array([0, 2, 4, 3])
    cols = np.array([5, 1, 0, 3])
    idx = _arr(sub2ind(data.shape, rows.copy(), cols.copy())).astype(int)
    assert np.array_equal(data.ravel()[idx], data[rows, cols])


def test_square_array_transposed_subscripts_differ():
    a = _arr(sub2ind((3, 3), np.array([0]), np.array([2])))
    b = _arr(sub2ind((3, 3), np.array([2]), np.array([0])))
    assert a.tolist() == [2]
    assert b.tolist() == [6]


def test_single_column_array():
    rows = np.array([0, 1, 2, 3])
    cols = np.array([0, 0, 0, 0])
    result = _arr(sub2ind((4, 1), rows, cols))
    assert result.tolist() == [0, 1, 2, 3]


def test_single_row_array():
    rows = np.array([0, 0, 0])
    cols = np.array([0, 1, 2])
    result = _arr(sub2ind((1, 3), rows, cols))
    assert result.tolist() == [0, 1, 2]


def test_shape_as_list_equals_shape_as_tuple():
    rows = np.array([1, 2])
    cols = np.array([3, 0])
    r_tuple = _arr(sub2ind((3, 4), rows.copy(), cols.copy()))
    r_list = _arr(sub2ind([3, 4], rows.copy(), cols.copy()))
    assert np.array_equal(r_tuple, r_list)


def test_unique_subscripts_give_unique_indices():
    shape = (6, 5)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = _arr(sub2ind(shape, rr.ravel().copy(), cc.ravel().copy()))
    assert len(set(result.tolist())) == shape[0] * shape[1]
