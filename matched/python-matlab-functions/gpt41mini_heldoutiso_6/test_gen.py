import pytest
from pymlfunc import sub2ind

def test_sub2ind_basic_2d():
    # Simple 2D case: matrix 3x4, row=1, col=1 -> linear index 1
    idx = sub2ind((3,4), 1, 1)
    assert idx == 1

def test_sub2ind_basic_2d_last_element():
    # Last element in 3x4 matrix: row=3, col=4 → linear index = 12
    idx = sub2ind((3,4), 3, 4)
    assert idx == 12

def test_sub2ind_basic_2d_middle_element():
    # Middle element in 3x4: row=2, col=3 → linear index = 8
    idx = sub2ind((3,4), 2, 3)
    assert idx == 8

def test_sub2ind_3d():
    # 3D array of size 2x3x4, indices (2,3,4) → linear index
    idx = sub2ind((2,3,4), 2, 3, 4)
    # Computing expected value:
    # linear_index = (4-1)*2*3 + (3-1)*2 + 2 = 3*6 + 2*2 +2 = 18+4+2=24
    assert idx == 24

def test_sub2ind_3d_first_element():
    idx = sub2ind((2,3,4), 1, 1, 1)
    assert idx == 1

def test_sub2ind_1d():
    # 1D array of length 5
    idx = sub2ind((5,), 3)
    assert idx == 3

def test_sub2ind_index_out_of_bounds_row():
    # row index out of bounds (too high)
    with pytest.raises(IndexError):
        sub2ind((3,4), 4, 1)

def test_sub2ind_index_out_of_bounds_col():
    # column index out of bounds (too low)
    with pytest.raises(IndexError):
        sub2ind((3,4), 2, 0)

def test_sub2ind_index_out_of_bounds_3d():
    # One index out of bounds in 3D
    with pytest.raises(IndexError):
        sub2ind((2,3,4), 1, 4, 1)

def test_sub2ind_zero_dimension():
    # Dimension with size zero should raise an error or behave accordingly
    with pytest.raises(ValueError):
        sub2ind((0,3), 1, 1)

def test_sub2ind_negative_index():
    with pytest.raises(IndexError):
        sub2ind((3,4), -1, 2)

def test_sub2ind_float_indices():
    # Indices that are floats (not ints) should raise TypeError or ValueError
    with pytest.raises(TypeError):
        sub2ind((3,4), 1.5, 2)

def test_sub2ind_missing_indices():
    # Number of indices less than number of dimensions should raise error
    with pytest.raises(TypeError):
        sub2ind((3,4), 1)

def test_sub2ind_extra_indices():
    # Number of indices more than number of dimensions should raise error
    with pytest.raises(TypeError):
        sub2ind((3,4), 1, 2, 3)

def test_sub2ind_tuple_indices():
    # Indices passed as a tuple should work if unpacked
    idx = sub2ind((3,4), *(2,3))
    assert idx == 8

def test_sub2ind_large_indices():
    # Large shape, large indices
    shape = (100, 200, 300)
    idx = sub2ind(shape, 100, 200, 300)
    # Calculate expected linear index:
    # linear = (300-1)*100*200 + (200-1)*100 + 100
    exp = (299)*100*200 + (199)*100 + 100
    assert idx == exp

def test_sub2ind_one_dim_single_index_1():
    # Single dimension, index = 1
    idx = sub2ind((1,), 1)
    assert idx == 1

def test_sub2ind_one_dim_single_index_end():
    # Single dimension, index = max
    idx = sub2ind((10,), 10)
    assert idx == 10

def test_sub2ind_index_tuple_incorrect_types():
    # Passing non-integer types as indices should raise TypeError
    with pytest.raises(TypeError):
        sub2ind((3,3), "1", 2)

def test_sub2ind_shape_not_tuple():
    # Passing shape as list instead of tuple should work as long as it's iterable
    idx = sub2ind([3,4], 2, 3)
    assert idx == 8

def test_sub2ind_all_ones_shape_and_indices():
    # Edge case: shape (1,1,1) and all indices 1
    idx = sub2ind((1,1,1), 1, 1, 1)
    assert idx == 1
