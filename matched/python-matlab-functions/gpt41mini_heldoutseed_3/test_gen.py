import pytest
from pymlfunc import sub2ind


def test_sub2ind_single_dimension():
    # 1D case: array size 10, indices 5 -> linear index should be 5 (0-based)
    size = (10,)
    subs = (5,)
    assert sub2ind(size, *subs) == 5


def test_sub2ind_two_dimensions_first_element():
    # 2D case: size 3x4, subscripts (0,0) -> linear index 0
    size = (3, 4)
    subs = (0, 0)
    assert sub2ind(size, *subs) == 0


def test_sub2ind_two_dimensions_middle_element():
    # 2D case: size 3x4, subscripts (1, 2) row 1 col 2
    size = (3, 4)
    subs = (1, 2)
    # linear index = 1 + 2*3 = 1 + 6 = 7
    assert sub2ind(size, *subs) == 7


def test_sub2ind_two_dimensions_last_element():
    # 2D case: size 3x4, last element (2, 3)
    size = (3, 4)
    subs = (2, 3)
    # linear index = 2 + 3*3 = 2 + 9 = 11
    assert sub2ind(size, *subs) == 11


def test_sub2ind_three_dimensions():
    # 3D case: size 2x3x4 subscripts (1, 2, 3)
    size = (2, 3, 4)
    subs = (1, 2, 3)
    # linear index = 1 + 2*2 + 3*2*3 = 1 + 4 + 18 = 23
    assert sub2ind(size, *subs) == 23


def test_sub2ind_zero_index_all_dimensions():
    # all zeros in subscripts always index 0
    size = (5, 5, 5)
    subs = (0, 0, 0)
    assert sub2ind(size, *subs) == 0


def test_sub2ind_index_error_too_few_subs():
    size = (3, 4)
    with pytest.raises(IndexError):
        # only one subscript provided instead of two
        sub2ind(size, 1)


def test_sub2ind_index_error_too_many_subs():
    size = (3, 4)
    with pytest.raises(IndexError):
        # three subscripts provided instead of two
        sub2ind(size, 1, 2, 3)


def test_sub2ind_index_out_of_bounds_negative():
    size = (4, 4)
    with pytest.raises(ValueError):
        # negative index invalid
        sub2ind(size, -1, 2)


def test_sub2ind_index_out_of_bounds_too_large():
    size = (4, 4)
    with pytest.raises(ValueError):
        # index 4 out of bounds (max index 3)
        sub2ind(size, 1, 4)


def test_sub2ind_large_dimensions():
    size = (10, 10, 10, 10)
    subs = (9, 0, 5, 2)
    # linear index = 9 + 0*10 + 5*10*10 + 2*10*10*10
    expected = 9 + 0 + 500 + 2000
    assert sub2ind(size, *subs) == expected


def test_sub2ind_single_dim_index_out_of_bounds():
    size = (3,)
    with pytest.raises(ValueError):
        sub2ind(size, 3)


def test_sub2ind_zero_size_dimension():
    size = (0, 5)
    # zero size dimension invalid, expect ValueError or IndexError when indexing
    with pytest.raises(ValueError):
        sub2ind(size, 0, 0)


def test_sub2ind_index_tuple_instead_of_args():
    size = (3, 3)
    # Some MATLAB sub2ind versions support tuple subscripts, test with unpacking
    subs = (1, 1)
    assert sub2ind(size, *subs) == 4  # 1 + 1*3 = 4


def test_sub2ind_index_with_numpy_ints():
    import numpy as np
    size = (3, 3)
    subs = (np.int32(2), np.int64(1))
    # linear index = 2 + 1*3 = 5
    assert sub2ind(size, *subs) == 5


def test_sub2ind_non_integer_indices_raise_typeerror():
    size = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(size, 1.5, 2)


def test_sub2ind_size_not_tuple_raises_typeerror():
    with pytest.raises(TypeError):
        sub2ind(5, 1)


def test_sub2ind_empty_size_raises_valueerror():
    with pytest.raises(ValueError):
        sub2ind((), 0)


def test_sub2ind_index_zero_with_size_one_dimension():
    size = (1,)
    subs = (0,)
    assert sub2ind(size, *subs) == 0


def test_sub2ind_index_at_max_boundary():
    size = (3, 3)
    subs = (2, 2)
    assert sub2ind(size, *subs) == 2 + 2 * 3  # 8


def test_sub2ind_index_non_tuple_size_raises():
    with pytest.raises(TypeError):
        sub2ind([3, 4], 2, 1)
