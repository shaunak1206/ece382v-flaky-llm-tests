import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def test_sub2ind_single_scalar_2d_first_element():
    assert sub2ind((3, 4), 0, 0) == 0


def test_sub2ind_single_scalar_2d_column_major_order():
    assert sub2ind((3, 4), 1, 0) == 1
    assert sub2ind((3, 4), 2, 0) == 2
    assert sub2ind((3, 4), 0, 1) == 3
    assert sub2ind((3, 4), 1, 1) == 4
    assert sub2ind((3, 4), 2, 1) == 5


def test_sub2ind_single_scalar_3d_column_major_order():
    # shape = (2, 3, 4), index formula:
    # ind = i + 2*j + 2*3*k
    assert sub2ind((2, 3, 4), 0, 0, 0) == 0
    assert sub2ind((2, 3, 4), 1, 0, 0) == 1
    assert sub2ind((2, 3, 4), 0, 1, 0) == 2
    assert sub2ind((2, 3, 4), 1, 1, 0) == 3
    assert sub2ind((2, 3, 4), 0, 0, 1) == 6
    assert sub2ind((2, 3, 4), 1, 2, 3) == 23


def test_sub2ind_accepts_shape_as_list():
    assert sub2ind([3, 4], 2, 1) == 5


def test_sub2ind_accepts_numpy_shape_array():
    shape = np.array([3, 4], dtype=int)
    assert sub2ind(shape, 2, 1) == 5


def test_sub2ind_1d_shape():
    assert sub2ind((5,), 0) == 0
    assert sub2ind((5,), 4) == 4


def test_sub2ind_vectorized_row_and_col_arrays():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    result = sub2ind((3, 3), rows, cols)
    expected = np.array([0, 4, 8])
    assert np.array_equal(result, expected)


def test_sub2ind_vectorized_mixed_broadcastable_inputs():
    rows = np.array([[0], [1], [2]])
    cols = np.array([0, 1, 2])
    result = sub2ind((3, 3), rows, cols)
    expected = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    assert np.array_equal(result, expected)


def test_sub2ind_vectorized_three_dimensions():
    i = np.array([0, 1, 0])
    j = np.array([0, 1, 2])
    k = np.array([0, 0, 1])
    result = sub2ind((2, 3, 4), i, j, k)
    expected = np.array([0, 3, 10])
    assert np.array_equal(result, expected)


def test_sub2ind_returns_numpy_array_for_array_inputs():
    result = sub2ind((3, 3), np.array([0, 1]), np.array([1, 2]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_sub2ind_matches_ind2sub_round_trip_2d_scalar():
    shape = (4, 5)
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = sub2ind(shape, i, j)
            out = ind2sub(shape, idx)
            assert out == (i, j)


def test_sub2ind_matches_ind2sub_round_trip_3d_scalar():
    shape = (2, 3, 4)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                idx = sub2ind(shape, i, j, k)
                out = ind2sub(shape, idx)
                assert out == (i, j, k)


def test_sub2ind_zero_sized_dimension_raises():
    with pytest.raises(Exception):
        sub2ind((0, 3), 0, 0)


def test_sub2ind_out_of_bounds_index_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 3, 0)
    with pytest.raises(Exception):
        sub2ind((3, 4), 0, 4)


def test_sub2ind_negative_index_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), -1, 0)
    with pytest.raises(Exception):
        sub2ind((3, 4), 0, -1)


def test_sub2ind_mismatched_number_of_indices_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 1)
    with pytest.raises(Exception):
        sub2ind((3, 4, 5), 1, 2)


def test_sub2ind_shape_and_indices_as_python_ints_and_numpy_ints():
    result = sub2ind((np.int64(3), np.int64(4)), np.int64(2), np.int64(1))
    assert result == 5


def test_sub2ind_output_is_stable_for_repeated_calls():
    shape = (3, 4, 5)
    args = (2, 1, 3)
    expected = sub2ind(shape, *args)
    for _ in range(10):
        assert sub2ind(shape, *args) == expected
