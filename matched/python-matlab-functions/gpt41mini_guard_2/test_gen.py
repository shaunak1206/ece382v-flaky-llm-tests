import pytest
from pymlfunc import sub2ind


# Tests for sub2ind - converting subscript indices to linear indices.


def test_sub2ind_basic_2d_row_major():
    # For a 2x3 matrix, index (1, 1) should map to 0, (2,1) -> 1, (1,2) -> 2 in row-major order
    shape = (2, 3)
    assert sub2ind(shape, (1, 1)) == 0
    assert sub2ind(shape, (2, 1)) == 1
    assert sub2ind(shape, (1, 2)) == 2
    assert sub2ind(shape, (2, 3)) == 5


def test_sub2ind_basic_2d_column_major():
    # Use order='F' (Fortran/column-major), for 2x3 matrix:
    # (1,1) -> 0, (2,1) -> 1, (1,2) -> 2, (2,3) -> 7
    shape = (2, 3)
    assert sub2ind(shape, (1, 1), order='F') == 0
    assert sub2ind(shape, (2, 1), order='F') == 1
    assert sub2ind(shape, (1, 2), order='F') == 2
    assert sub2ind(shape, (2, 3), order='F') == 7


def test_sub2ind_3d_indices_row_major():
    shape = (2, 2, 2)
    # Check several indices manually
    assert sub2ind(shape, (1, 1, 1)) == 0
    assert sub2ind(shape, (2, 1, 1)) == 1
    assert sub2ind(shape, (1, 2, 1)) == 2
    assert sub2ind(shape, (1, 1, 2)) == 4
    assert sub2ind(shape, (2, 2, 2)) == 7


def test_sub2ind_3d_indices_column_major():
    shape = (2, 2, 2)
    # Column-major indexing (Fortran order)
    assert sub2ind(shape, (1, 1, 1), order='F') == 0
    assert sub2ind(shape, (2, 1, 1), order='F') == 1
    assert sub2ind(shape, (1, 2, 1), order='F') == 2
    assert sub2ind(shape, (1, 1, 2), order='F') == 4
    assert sub2ind(shape, (2, 2, 2), order='F') == 7


def test_sub2ind_vector_indices():
    shape = (10,)
    # For 1D, sub2ind is subscript - 1 in zero-based linear indexing
    for i in range(1, 11):
        assert sub2ind(shape, (i, )) == i - 1


def test_sub2ind_tuple_indices():
    shape = (3, 3)
    idx = (2, 3)
    # By default order 'C' means row major: linear index = (row-1)*cols + (col-1)
    expected = (2 - 1)*3 + (3 - 1)
    assert sub2ind(shape, idx) == expected


def test_sub2ind_list_indices():
    shape = (4, 5)
    idx = [4, 3]  # indices must be positive ints, here 4th row, 3rd col
    expected = (4 - 1)*5 + (3 - 1)
    assert sub2ind(shape, idx) == expected


def test_sub2ind_invalid_shape_type():
    with pytest.raises(TypeError):
        sub2ind(None, (1, 1))


def test_sub2ind_invalid_indices_type():
    with pytest.raises(TypeError):
        sub2ind((2, 2), None)


def test_sub2ind_wrong_number_of_indices():
    shape = (2, 2)
    # More indices than dimensions
    with pytest.raises(ValueError):
        sub2ind(shape, (1, 1, 1))
    # Fewer indices than dimensions
    with pytest.raises(ValueError):
        sub2ind(shape, (1,))


def test_sub2ind_index_out_of_bounds():
    shape = (3, 3)
    # Row subscript out of bounds
    with pytest.raises(IndexError):
        sub2ind(shape, (0, 1))
    with pytest.raises(IndexError):
        sub2ind(shape, (4, 1))
    # Column subscript out of bounds
    with pytest.raises(IndexError):
        sub2ind(shape, (1, 0))
    with pytest.raises(IndexError):
        sub2ind(shape, (1, 5))


def test_sub2ind_float_indices_raises():
    shape = (2, 2)
    with pytest.raises(TypeError):
        sub2ind(shape, (1.5, 1))
    with pytest.raises(TypeError):
        sub2ind(shape, (1, 2.0))


def test_sub2ind_negative_indices_raises():
    shape = (5, 5)
    with pytest.raises(IndexError):
        sub2ind(shape, (-1, 2))
    with pytest.raises(IndexError):
        sub2ind(shape, (3, -2))


def test_sub2ind_zero_indices_raises():
    # MATLAB style indices are 1-based and positive integers.
    shape = (4, 4)
    with pytest.raises(IndexError):
        sub2ind(shape, (0, 1))
    with pytest.raises(IndexError):
        sub2ind(shape, (1, 0))


def test_sub2ind_large_indices():
    shape = (1000, 1000)
    idx = (1000, 1000)
    result = sub2ind(shape, idx)
    # For row-major, linear index = (row-1)*ncols + col-1 = 999*1000 + 999 = 999999
    assert result == 999999


def test_sub2ind_order_parameter_invalid_value():
    shape = (2, 2)
    with pytest.raises(ValueError):
        sub2ind(shape, (1, 1), order='X')


def test_sub2ind_multiple_indices_input():
    shape = (3, 3)
    # If sub2ind supports multiple indices as multiple inputs (tuple vs multiple arguments)
    # but source suggests second argument is index tuple, so providing multiple indices as separate tuple elements will error
    with pytest.raises(TypeError):
        sub2ind(shape, 1, 1)


def test_sub2ind_for_tuple_input_coercion():
    shape = (2, 2)
    # Provide indices as tuple of length 2 (valid)
    indices = (2, 2)
    lin_idx = sub2ind(shape, indices)
    assert isinstance(lin_idx, int)


def test_sub2ind_for_list_input_coercion():
    shape = (2, 2)
    # Provide indices as list of length 2 (valid)
    indices = [2, 2]
    lin_idx = sub2ind(shape, indices)
    assert isinstance(lin_idx, int)
