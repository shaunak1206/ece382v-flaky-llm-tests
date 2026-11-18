import pytest
from pymlfunc import sub2ind

def test_sub2ind_1d_basic():
    # For a 1D array, sub2ind should return subscript - 1 (0-based index)
    shape = (10,)
    subs = (5,)
    idx = sub2ind(shape, *subs)
    assert idx == 4

def test_sub2ind_2d_first_element():
    shape = (5, 5)
    subs = (1, 1)
    idx = sub2ind(shape, *subs)
    assert idx == 0

def test_sub2ind_2d_last_element():
    shape = (5, 5)
    subs = (5, 5)
    idx = sub2ind(shape, *subs)
    assert idx == 24

def test_sub2ind_2d_middle_element():
    shape = (4, 6)
    subs = (3, 4)
    idx = sub2ind(shape, *subs)
    # Check calculation for zero-based: (4-1)*4 + (3-1) = 3*4 + 2 = 14
    # Actually, MATLAB order is column-major so: idx = (col-1)*num_rows + (row-1)
    # so idx = (4-1)*4 + (3-1) = 3*4 + 2 = 14
    assert idx == 14

def test_sub2ind_3d_element():
    shape = (3, 4, 5)
    subs = (2, 3, 4)
    # idx = ((z-1)*num_rows*num_cols) + ((y-1)*num_rows) + (x-1)
    # = (4-1)*3*4 + (3-1)*3 + (2-1)
    # = 3*12 + 2*3 + 1 = 36 + 6 + 1 = 43
    idx = sub2ind(shape, *subs)
    assert idx == 43

def test_sub2ind_out_of_bounds_row():
    shape = (4, 5)
    subs = (5, 3)  # row=5 > 4
    with pytest.raises(IndexError):
        sub2ind(shape, *subs)

def test_sub2ind_out_of_bounds_col():
    shape = (4, 5)
    subs = (3, 6)  # col=6 > 5
    with pytest.raises(IndexError):
        sub2ind(shape, *subs)

def test_sub2ind_zero_index():
    shape = (4, 5)
    subs = (0, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, *subs)

def test_sub2ind_dimension_mismatch_too_few_subs():
    shape = (4, 5, 6)
    subs = (1, 2)  # only 2 subscripts instead of 3
    with pytest.raises(TypeError):
        sub2ind(shape, *subs)

def test_sub2ind_dimension_mismatch_too_many_subs():
    shape = (4, 5)
    subs = (1, 2, 3)  # 3 subscripts instead of 2
    with pytest.raises(TypeError):
        sub2ind(shape, *subs)

def test_sub2ind_non_integer_subs():
    shape = (10, 10)
    subs = (3.5, 2)
    with pytest.raises(TypeError):
        sub2ind(shape, *subs)

def test_sub2ind_minimum_shape_1d_and_sub_indices_1():
    shape = (1,)
    subs = (1,)
    idx = sub2ind(shape, *subs)
    assert idx == 0

def test_sub2ind_large_shape_and_subs():
    shape = (100, 200, 300)
    subs = (50, 100, 150)
    idx = sub2ind(shape, *subs)
    # idx = ((150-1)*100*200) + ((100-1)*100) + (50-1)
    # = 149*20000 + 99*100 + 49
    expected = 149*20000 + 99*100 + 49
    assert idx == expected

def test_sub2ind_single_element_array():
    shape = (1, 1, 1)
    subs = (1, 1, 1)
    idx = sub2ind(shape, *subs)
    assert idx == 0

def test_sub2ind_negative_index():
    shape = (5, 5)
    subs = (-1, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, *subs)

def test_sub2ind_shape_not_tuple():
    # Shape must be a tuple - if list given, still should work if supported by sub2ind
    shape = [5, 6]  # list instead of tuple
    subs = (3, 4)
    idx = sub2ind(shape, *subs)
    assert idx == (4-1)*5 + (3-1)

def test_sub2ind_subs_not_tuple_unpacking():
    shape = (5, 5)
    subs = [2, 2]
    idx = sub2ind(shape, *subs)
    assert idx == (2-1)*5 + (2-1)

def test_sub2ind_with_tuple_of_subs_argument():
    shape = (5,5)
    subs = (3, 4)
    idx = sub2ind(shape, *subs)
    assert idx == (4-1)*5 + (3-1)

def test_sub2ind_all_ones_subs_in_multidimensional():
    shape = (3, 4, 5, 6)
    subs = (1, 1, 1, 1)
    idx = sub2ind(shape, *subs)
    assert idx == 0

def test_sub2ind_all_max_subs_in_multidimensional():
    shape = (3, 4, 5, 6)
    subs = (3, 4, 5, 6)
    size0, size1, size2, size3 = shape
    # idx = (6-1)*3*4*5 + (5-1)*3*4 + (4-1)*3 + (3-1)
    expected = (6-1)*3*4*5 + (5-1)*3*4 + (4-1)*3 + (3-1)
    idx = sub2ind(shape, *subs)
    assert idx == expected
