import pytest

from pymlfunc import sub2ind


def test_sub2ind_2d_scalar_basic_column_major_order():
    # MATLAB/Fortran order: linear index increases down rows first, then columns.
    assert sub2ind((3, 4), 1, 1) == 1
    assert sub2ind((3, 4), 2, 1) == 2
    assert sub2ind((3, 4), 3, 1) == 3
    assert sub2ind((3, 4), 1, 2) == 4
    assert sub2ind((3, 4), 2, 2) == 5


def test_sub2ind_2d_last_element():
    assert sub2ind((3, 4), 3, 4) == 12


def test_sub2ind_2d_middle_element():
    # (row, col) = (2, 3) in a 3x4 array -> 2 + (3-1)*3 = 8
    assert sub2ind((3, 4), 2, 3) == 8


def test_sub2ind_3d_scalar_basic():
    # For shape (2, 3, 4), MATLAB order linear index:
    # i + (j-1)*2 + (k-1)*2*3
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1
    assert sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert sub2ind((2, 3, 4), 1, 2, 1) == 3
    assert sub2ind((2, 3, 4), 1, 1, 2) == 7


def test_sub2ind_accepts_list_shape():
    assert sub2ind([3, 4], 2, 3) == 8


def test_sub2ind_accepts_tuple_shape():
    assert sub2ind((3, 4), 2, 3) == 8


def test_sub2ind_accepts_numpy_integer_arguments_if_available():
    np = pytest.importorskip("numpy")
    assert sub2ind((3, 4), np.int64(2), np.int32(3)) == 8


def test_sub2ind_vectorized_indices_return_sequence():
    result = sub2ind((3, 4), [1, 2, 3], [1, 1, 1])
    assert list(result) == [1, 2, 3]


def test_sub2ind_vectorized_mixed_columns():
    result = sub2ind((3, 4), [1, 2, 3], [2, 2, 2])
    assert list(result) == [4, 5, 6]


def test_sub2ind_vectorized_3d_indices():
    result = sub2ind((2, 3, 4), [1, 2], [1, 3], [1, 2])
    assert list(result) == [1, 12]


def test_sub2ind_out_of_bounds_row_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 4, 1)


def test_sub2ind_out_of_bounds_column_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 1, 5)


def test_sub2ind_zero_index_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 0, 1)


def test_sub2ind_negative_index_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), -1, 1)


def test_sub2ind_dimension_mismatch_raises():
    with pytest.raises(Exception):
        sub2ind((3, 4), 1, 1, 1)


def test_sub2ind_empty_shape_raises():
    with pytest.raises(Exception):
        sub2ind((), 1)
