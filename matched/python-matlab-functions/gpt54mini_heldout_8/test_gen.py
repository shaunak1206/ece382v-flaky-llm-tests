import numpy as np
import pytest

from pymlfunc import sub2ind


def test_sub2ind_scalar_1d_first_element():
    assert sub2ind((5,), (1,)) == 1


def test_sub2ind_scalar_1d_last_element():
    assert sub2ind((5,), (5,)) == 5


def test_sub2ind_2d_column_major_basic():
    # MATLAB/Fortran-style column-major indexing
    assert sub2ind((3, 4), (1, 1)) == 1
    assert sub2ind((3, 4), (2, 1)) == 2
    assert sub2ind((3, 4), (3, 1)) == 3
    assert sub2ind((3, 4), (1, 2)) == 4
    assert sub2ind((3, 4), (2, 2)) == 5
    assert sub2ind((3, 4), (3, 4)) == 12


def test_sub2ind_2d_matches_manual_formula():
    dims = (4, 5)
    row, col = 3, 4
    expected = row + (col - 1) * dims[0]
    assert sub2ind(dims, (row, col)) == expected


def test_sub2ind_3d_column_major_formula():
    dims = (2, 3, 4)
    subs = (2, 3, 4)
    expected = 2 + (3 - 1) * dims[0] + (4 - 1) * dims[0] * dims[1]
    assert sub2ind(dims, subs) == expected


def test_sub2ind_accepts_numpy_scalar_inputs():
    dims = np.array([3, 4])
    subs = np.array([2, 3])
    assert sub2ind(dims, subs) == 8


def test_sub2ind_accepts_python_lists():
    assert sub2ind([3, 4], [2, 3]) == 8


def test_sub2ind_tuple_and_list_mixed_inputs():
    assert sub2ind((3, 4), [3, 2]) == 6


def test_sub2ind_multiple_subscripts_shape_2d():
    dims = (3, 4)
    rows = np.array([1, 2, 3])
    cols = np.array([1, 2, 4])
    result = sub2ind(dims, (rows, cols))
    assert np.all(result == np.array([1, 5, 12]))


def test_sub2ind_multiple_subscripts_shape_3d():
    dims = (2, 3, 4)
    subs = (
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([1, 4]),
    )
    result = sub2ind(dims, subs)
    expected = np.array([
        1 + (1 - 1) * 2 + (1 - 1) * 2 * 3,
        2 + (3 - 1) * 2 + (4 - 1) * 2 * 3,
    ])
    assert np.all(result == expected)


def test_sub2ind_returns_numpy_array_for_vectorized_inputs():
    result = sub2ind((3, 3), (np.array([1, 2]), np.array([1, 3])))
    assert isinstance(result, np.ndarray)
    assert result.dtype.kind in {"i", "u"}
    assert np.array_equal(result, np.array([1, 8]))


def test_sub2ind_raises_on_zero_index():
    with pytest.raises(Exception):
        sub2ind((3, 4), (0, 1))


def test_sub2ind_raises_on_negative_index():
    with pytest.raises(Exception):
        sub2ind((3, 4), (-1, 1))


def test_sub2ind_raises_on_out_of_bounds_index():
    with pytest.raises(Exception):
        sub2ind((3, 4), (4, 1))


def test_sub2ind_raises_on_dimension_mismatch():
    with pytest.raises(Exception):
        sub2ind((3, 4), (1, 2, 3))
