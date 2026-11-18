import pytest
from pymlfunc import sub2ind


def test_sub2ind_basic_2d():
    # 2D matrix, linear indexing for (1,1)
    dims = (3, 4)
    subs = (1, 1)
    assert sub2ind(dims, subs) == 0

def test_sub2ind_basic_2d_last_element():
    dims = (3, 4)
    subs = (3, 4)
    # Last element in a 3x4 matrix is index 11 (zero-based)
    assert sub2ind(dims, subs) == 11

def test_sub2ind_basic_3d():
    dims = (3, 4, 5)
    subs = (1, 1, 1)
    assert sub2ind(dims, subs) == 0

def test_sub2ind_middle_3d():
    dims = (3, 4, 5)
    subs = (2, 3, 4)
    # Manual computation:
    # zero based: (1, 2, 3)
    # index = 1 + 2*3 + 3*3*4 = 1 + 6 + 36 = 43
    assert sub2ind(dims, subs) == 43

def test_sub2ind_vector_indexing():
    dims = (5,)
    subs = (3,)
    # zero based index = 2
    assert sub2ind(dims, subs) == 2

def test_sub2ind_with_tuple_of_subscripts():
    dims = (2, 3)
    subs = (2, 2)
    # zero based (1,1) = index 4
    assert sub2ind(dims, subs) == 4

def test_sub2ind_zero_dimension():
    with pytest.raises(ValueError):
        sub2ind((), (1,))

def test_sub2ind_out_of_bounds_high():
    dims = (3, 4)
    subs = (4, 2)
    with pytest.raises(IndexError):
        sub2ind(dims, subs)

def test_sub2ind_out_of_bounds_low():
    dims = (3, 4)
    subs = (0, 3)
    with pytest.raises(IndexError):
        sub2ind(dims, subs)

def test_sub2ind_non_integer_subscripts():
    dims = (3, 3)
    subs = (1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(dims, subs)

def test_sub2ind_non_integer_dimensions():
    dims = (3.0, 4)
    subs = (1, 1)
    with pytest.raises(TypeError):
        sub2ind(dims, subs)

def test_sub2ind_string_subscripts():
    dims = (3, 4)
    subs = ('1', 2)
    with pytest.raises(TypeError):
        sub2ind(dims, subs)

def test_sub2ind_string_dimensions():
    dims = ('3', 4)
    subs = (1, 1)
    with pytest.raises(TypeError):
        sub2ind(dims, subs)

def test_sub2ind_single_dimension_multiple_subscripts():
    dims = (5,)
    subs = (1, 2)
    with pytest.raises(ValueError):
        sub2ind(dims, subs)

def test_sub2ind_multi_dimension_single_subscript():
    dims = (2, 2)
    subs = (1,)
    with pytest.raises(ValueError):
        sub2ind(dims, subs)

def test_sub2ind_large_dimensions():
    dims = (100, 100, 100)
    subs = (50, 50, 50)
    idx = sub2ind(dims, subs)
    assert isinstance(idx, int)
    assert 0 <= idx < 100*100*100

def test_sub2ind_minimum_subscript_1():
    dims = (3, 3)
    subs = (1, 1)
    assert sub2ind(dims, subs) == 0

def test_sub2ind_subscript_equal_dimension():
    dims = (3, 3)
    subs = (3, 3)
    # should be last index = 8
    assert sub2ind(dims, subs) == 8

def test_sub2ind_subscripts_tuple_and_list_equivalence():
    dims = (3, 3)
    subs_tuple = (2, 3)
    subs_list = [2, 3]
    assert sub2ind(dims, subs_tuple) == sub2ind(dims, subs_list)
