import numpy as np
import pytest

import pymlfunc
from pymlfunc import sub2ind


def _arr(values):
    return np.array(values, dtype=int)


def test_importable_from_package_and_submodule():
    from pymlfunc import subind
    assert pymlfunc.sub2ind is subind.sub2ind
    assert callable(sub2ind)


def test_known_values_3x4():
    shape = (3, 4)
    rows = _arr([0, 0, 1, 2, 1])
    cols = _arr([0, 3, 0, 3, 2])
    result = sub2ind(shape, rows, cols)
    assert np.array_equal(np.asarray(result), np.array([0, 3, 4, 11, 6]))


def test_full_grid_gives_consecutive_indices():
    shape = (4, 5)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = np.asarray(sub2ind(shape, rr.ravel().astype(int), cc.ravel().astype(int)))
    assert np.array_equal(result, np.arange(20))


def test_matches_numpy_ravel_multi_index():
    shape = (6, 7)
    rows = _arr([0, 5, 2, 3, 4, 1])
    cols = _arr([6, 0, 3, 3, 1, 5])
    expected = np.ravel_multi_index((rows, cols), shape)
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, expected)


def test_seeded_random_subscripts_match_numpy():
    np.random.seed(12345)
    shape = (13, 17)
    rows = np.random.randint(0, shape[0], size=200)
    cols = np.random.randint(0, shape[1], size=200)
    expected = np.ravel_multi_index((rows, cols), shape)
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, expected)


def test_output_shape_matches_1d_input():
    shape = (5, 5)
    rows = _arr([0, 1, 2, 3])
    cols = _arr([4, 3, 2, 1])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.shape == (4,)


def test_output_shape_matches_2d_input():
    shape = (5, 6)
    rows = _arr([[0, 1, 2], [3, 4, 0]])
    cols = _arr([[5, 4, 3], [2, 1, 0]])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert result.shape == (2, 3)
    expected = np.ravel_multi_index((rows, cols), shape)
    assert np.array_equal(result, expected)


def test_single_column_matrix_index_equals_row():
    shape = (6, 1)
    rows = _arr([0, 1, 2, 3, 4, 5])
    cols = _arr([0, 0, 0, 0, 0, 0])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, rows)


def test_single_row_matrix_index_equals_column():
    shape = (1, 6)
    rows = _arr([0, 0, 0, 0])
    cols = _arr([5, 0, 3, 2])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, cols)


def test_square_matrix_corners():
    shape = (4, 4)
    rows = _arr([0, 0, 3, 3])
    cols = _arr([0, 3, 0, 3])
    result = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(result, np.array([0, 3, 12, 15]))


def test_indices_are_unique_for_unique_subscripts():
    shape = (7, 3)
    rr, cc = np.meshgrid(np.arange(7), np.arange(3), indexing="ij")
    result = np.asarray(sub2ind(shape, rr.ravel().astype(int), cc.ravel().astype(int)))
    assert len(np.unique(result)) == 21
    assert result.min() == 0
    assert result.max() == 20


def test_moving_one_column_adds_one_and_one_row_adds_ncols():
    shape = (5, 8)
    base = sub2ind(shape, _arr([2]), _arr([3]))
    right = sub2ind(shape, _arr([2]), _arr([4]))
    down = sub2ind(shape, _arr([3]), _arr([3]))
    assert int(np.asarray(right)[0]) - int(np.asarray(base)[0]) == 1
    assert int(np.asarray(down)[0]) - int(np.asarray(base)[0]) == 8


def test_inputs_are_not_modified():
    shape = (4, 4)
    rows = _arr([0, 1, 2, 3])
    cols = _arr([3, 2, 1, 0])
    rows_copy = rows.copy()
    cols_copy = cols.copy()
    sub2ind(shape, rows, cols)
    assert np.array_equal(rows, rows_copy)
    assert np.array_equal(cols, cols_copy)


def test_result_can_index_flattened_matrix():
    shape = (3, 5)
    matrix = np.arange(15).reshape(shape) * 10
    rows = _arr([0, 2, 1, 2])
    cols = _arr([1, 4, 0, 2])
    idx = np.asarray(sub2ind(shape, rows, cols)).astype(int)
    assert np.array_equal(matrix.ravel()[idx], matrix[rows, cols])
