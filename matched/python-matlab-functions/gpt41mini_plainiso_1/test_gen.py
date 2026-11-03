import pytest
from pymlfunc import sub2ind

def test_sub2ind_1d_basic():
    # Simple 1D test: linear index is same as subscript -1
    dims = (10,)
    subs = (1,)
    idx = sub2ind(dims, *subs)
    assert idx == 0
    subs = (10,)
    idx = sub2ind(dims, *subs)
    assert idx == 9

def test_sub2ind_2d_basic():
    dims = (3, 4)
    # subscripts (row=2, col=3) -> index = (3-1)*3 + (2-1) = 1*3 + 1 = 4
    assert sub2ind(dims, 2, 3) == 4
    # first element
    assert sub2ind(dims, 1, 1) == 0
    # last element
    assert sub2ind(dims, 3, 4) == 11

def test_sub2ind_3d_basic():
    dims = (4, 3, 2)
    # subscripts (4,3,2)
    # index = (2-1)*4*3 + (3-1)*4 + (4-1) = 1*12 + 2*4 + 3 = 12 + 8 + 3 = 23
    assert sub2ind(dims, 4, 3, 2) == 23
    # first element
    assert sub2ind(dims, 1, 1, 1) == 0

def test_sub2ind_raises_on_too_many_subs():
    dims = (3, 3)
    with pytest.raises(TypeError):
        # too many subscripts means passing extra arguments to sub2ind
        sub2ind(dims, 1, 2, 3)

def test_sub2ind_raises_on_too_few_subs():
    dims = (3, 3, 3)
    with pytest.raises(TypeError):
        # too few subscripts means missing arguments
        sub2ind(dims, 1, 2)

def test_sub2ind_subscript_out_of_range_raises():
    dims = (3, 3)
    # subscript zero is invalid (MATLAB 1-based indexing)
    with pytest.raises(ValueError):
        sub2ind(dims, 0, 1)
    with pytest.raises(ValueError):
        sub2ind(dims, 1, 0)
    with pytest.raises(ValueError):
        sub2ind(dims, -1, 2)
    with pytest.raises(ValueError):
        sub2ind(dims, 1, 4)
    with pytest.raises(ValueError):
        sub2ind(dims, 4, 1)

def test_sub2ind_large_dims():
    dims = (100, 200, 50)
    subs = (100, 200, 50)
    # Check last element
    expected = (50-1)*100*200 + (200-1)*100 + (100-1)
    result = sub2ind(dims, *subs)
    assert result == expected

def test_sub2ind_first_and_last_elements_1d():
    dims = (5,)
    assert sub2ind(dims, 1) == 0
    assert sub2ind(dims, 5) == 4

def test_sub2ind_first_and_last_elements_2d():
    dims = (2, 3)
    assert sub2ind(dims, 1, 1) == 0
    assert sub2ind(dims, 2, 3) == 5

def test_sub2ind_scalar_input_dims_and_subs():
    dims = (1,)
    assert sub2ind(dims, 1) == 0

def test_sub2ind_with_dims_as_list():
    dims = [3, 3]
    assert sub2ind(dims, 1, 1) == 0
    assert sub2ind(dims, 3, 3) == 8

def test_sub2ind_with_subs_as_list():
    dims = (3, 3)
    subs = [2, 2]
    assert sub2ind(dims, *subs) == 4

def test_sub2ind_non_integer_subs_raises():
    dims = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(dims, 1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(dims, 2, "3")

def test_sub2ind_non_integer_dims_raises():
    dims = (3.5, 4)
    with pytest.raises(TypeError):
        sub2ind(dims, 1, 1)

def test_sub2ind_empty_dims():
    dims = ()
    with pytest.raises(TypeError):
        sub2ind(dims, 1)

def test_sub2ind_with_numpy_ints():
    import numpy as np
    dims = (np.int32(3), np.int64(4))
    assert sub2ind(dims, 1, 1) == 0
    assert sub2ind(dims, 3, 4) == 11

def test_sub2ind_with_one_dim_and_subs_as_tuple():
    dims = (10,)
    subs = (5,)
    assert sub2ind(dims, *subs) == 4

def test_sub2ind_multi_dim_row_major_vs_column_major_match():
    # MATLAB uses column-major order: verify calculation
    dims = (2, 3)
    # subscripts: (1,2)
    # index = (2-1)*2 + (1-1) = 2 + 0 = 2
    assert sub2ind(dims, 1, 2) == 2
    # verify scipy/numpy (row-major) would differ if interpreted differently
    # This test checks the internal logic is MATLAB style

def test_sub2ind_subs_are_exactly_dims():
    dims = (5, 5, 5)
    subs = (5, 5, 5)
    expected = (5-1)*5*5 + (5-1)*5 + (5-1)  # 4*25 + 4*5 + 4 = 100 + 20 + 4 = 124
    assert sub2ind(dims, *subs) == expected

def test_sub2ind_subs_all_ones():
    dims = (7, 8, 9)
    subs = (1, 1, 1)
    assert sub2ind(dims, *subs) == 0
