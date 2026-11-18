import numpy as np
import pytest

from pymlfunc import sub2ind


def matlab_sub2ind_2d(shape, rows, cols):
    m, n = shape
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    return rows + (cols - 1) * m


def matlab_sub2ind_nd(shape, subs):
    # MATLAB/Fortran-order 1-based linear index
    shape = tuple(shape)
    subs = [np.asarray(s) for s in subs]
    idx0 = np.zeros_like(subs[0], dtype=int)
    stride = 1
    for dim, s in zip(shape, subs):
        idx0 = idx0 + (s - 1) * stride
        stride *= dim
    return idx0 + 1


def test_sub2ind_single_element_matrix_returns_one():
    assert sub2ind((1, 1), 1, 1) == 1


def test_sub2ind_first_column_indices_are_consecutive():
    assert sub2ind((4, 3), 1, 1) == 1
    assert sub2ind((4, 3), 2, 1) == 2
    assert sub2ind((4, 3), 3, 1) == 3
    assert sub2ind((4, 3), 4, 1) == 4


def test_sub2ind_second_column_starts_after_number_of_rows():
    assert sub2ind((4, 3), 1, 2) == 5
    assert sub2ind((4, 3), 2, 2) == 6
    assert sub2ind((4, 3), 4, 2) == 8


def test_sub2ind_matches_matlab_formula_for_2d_points():
    shape = (5, 7)
    rows = np.array([1, 2, 5, 3])
    cols = np.array([1, 4, 7, 2])

    expected = matlab_sub2ind_2d(shape, rows, cols)
    actual = sub2ind(shape, rows, cols)

    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_accepts_python_lists_for_2d_inputs():
    shape = (3, 4)
    expected = np.array([1, 6, 12])
    actual = sub2ind(shape, [1, 3, 3], [1, 2, 4])
    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_returns_scalar_for_scalar_inputs():
    result = sub2ind((6, 4), 4, 3)
    assert np.isscalar(result)
    assert result == 16


def test_sub2ind_3d_single_point_matches_column_major_formula():
    shape = (2, 3, 4)
    expected = matlab_sub2ind_nd(shape, (2, 3, 4))
    assert sub2ind(shape, 2, 3, 4) == expected


def test_sub2ind_3d_vectorized_inputs_match_formula():
    shape = (2, 3, 4)
    subs = (
        np.array([1, 2, 1, 2]),
        np.array([1, 3, 2, 2]),
        np.array([1, 1, 4, 3]),
    )

    expected = matlab_sub2ind_nd(shape, subs)
    actual = sub2ind(shape, *subs)

    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_2d_vectorized_inputs_match_known_values():
    shape = (3, 4)
    rows = np.array([1, 2, 3, 1])
    cols = np.array([1, 1, 1, 4])

    expected = np.array([1, 2, 3, 10])
    actual = sub2ind(shape, rows, cols)

    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_preserves_shape_of_numpy_array_inputs():
    shape = (4, 5)
    rows = np.array([[1, 2], [3, 4]])
    cols = np.array([[1, 1], [5, 5]])

    actual = sub2ind(shape, rows, cols)
    expected = np.array([[1, 2], [19, 20]])

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_handles_column_vector_like_inputs():
    shape = (3, 3)
    rows = np.array([[1], [2], [3]])
    cols = np.array([[1], [2], [3]])

    expected = np.array([[1], [5], [9]])
    actual = sub2ind(shape, rows, cols)

    np.testing.assert_array_equal(actual, expected)


def test_sub2ind_linear_index_increases_by_one_down_a_column():
    shape = (5, 2)
    indices = [sub2ind(shape, r, 2) for r in range(1, 6)]
    assert indices == [6, 7, 8, 9, 10]
