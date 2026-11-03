import numpy as np
import pytest

from pymlfunc import sub2ind


def test_docstring_example():
    mat = np.arange(1, 31).reshape((5, 6))
    ind = sub2ind(mat.shape, np.array([0, 2]), np.array([1, 3]))
    np.testing.assert_array_equal(ind, np.array([1, 15]))


def test_matches_numpy_ravel_multi_index():
    shape = (5, 6)
    rows = np.array([0, 1, 2, 3, 4])
    cols = np.array([5, 4, 3, 2, 1])
    expected = np.ravel_multi_index((rows, cols), shape)
    np.testing.assert_array_equal(sub2ind(shape, rows.copy(), cols.copy()), expected)


def test_flat_index_selects_correct_element():
    mat = np.arange(100).reshape((10, 10)) * 3
    rows = np.array([0, 3, 9, 5])
    cols = np.array([0, 7, 9, 2])
    ind = sub2ind(mat.shape, rows, cols)
    np.testing.assert_array_equal(mat.ravel()[ind], mat[rows, cols])


def test_origin_is_zero():
    ind = sub2ind((4, 7), np.array([0]), np.array([0]))
    assert ind[0] == 0


def test_last_element_is_valid():
    shape = (4, 7)
    ind = sub2ind(shape, np.array([3]), np.array([6]))
    assert ind[0] == 27


def test_returns_numpy_array_same_length():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    ind = sub2ind((3, 3), rows, cols)
    assert isinstance(ind, np.ndarray)
    assert ind.shape == rows.shape


def test_diagonal_of_square_matrix():
    n = 5
    idx = np.arange(n)
    ind = sub2ind((n, n), idx, idx.copy())
    np.testing.assert_array_equal(ind, idx * (n + 1))


def test_negative_row_gives_minus_one():
    ind = sub2ind((5, 6), np.array([-1]), np.array([0]))
    assert ind[0] == -1


def test_negative_result_gives_minus_one():
    ind = sub2ind((5, 6), np.array([0]), np.array([-3]))
    assert ind[0] == -1


def test_row_out_of_range_gives_minus_one():
    ind = sub2ind((5, 6), np.array([5]), np.array([0]))
    assert ind[0] == -1


def test_large_row_out_of_range_gives_minus_one():
    ind = sub2ind((5, 6), np.array([100]), np.array([2]))
    assert ind[0] == -1


def test_mixed_valid_and_invalid():
    ind = sub2ind(
        (3, 4),
        np.array([0, -1, 2, 3, 1]),
        np.array([1, 0, 3, 0, 2]),
    )
    np.testing.assert_array_equal(ind, np.array([1, -1, 11, -1, 6]))


def test_single_row_matrix():
    ind = sub2ind((1, 5), np.array([0, 0, 0]), np.array([0, 2, 4]))
    np.testing.assert_array_equal(ind, np.array([0, 2, 4]))


def test_single_column_matrix():
    ind = sub2ind((5, 1), np.array([0, 2, 4]), np.array([0, 0, 0]))
    np.testing.assert_array_equal(ind, np.array([0, 2, 4]))


def test_empty_inputs():
    ind = sub2ind((3, 3), np.array([], dtype=int), np.array([], dtype=int))
    assert ind.shape == (0,)


def test_row_major_ordering_not_column_major():
    # (row 1, col 0) in a 3x4 matrix is flat index 4 in C order (would be 1 in Fortran order)
    ind = sub2ind((3, 4), np.array([1]), np.array([0]))
    assert ind[0] == 4


def test_all_positions_enumerate_in_order():
    shape = (3, 4)
    rr, cc = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    ind = sub2ind(shape, rr.ravel(), cc.ravel())
    np.testing.assert_array_equal(ind, np.arange(12))


def test_scalar_inputs_raise_type_error():
    with pytest.raises(TypeError):
        sub2ind((3, 3), 1, 1)
