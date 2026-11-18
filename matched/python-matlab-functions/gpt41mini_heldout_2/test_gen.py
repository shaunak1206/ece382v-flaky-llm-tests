import pytest
from pymlfunc import sub2ind


def test_sub2ind_basic_2d():
    # 2D array shape (3,4)
    shape = (3, 4)
    # indices: row=1, col=2 (zero-based)
    idx = (1, 2)
    expected = 1 + 2*3  # row + col*rows (MATLAB col-major)
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_basic_3d():
    # 3D array shape (2,3,4)
    shape = (2, 3, 4)
    # indices: (1, 1, 1)
    idx = (1, 1, 1)
    # linear index = i + j*dim0 + k*dim0*dim1
    expected = 1 + 1*2 + 1*2*3
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_single_dimension():
    shape = (5,)
    idx = (3,)
    assert sub2ind(shape, *idx) == 3


def test_sub2ind_all_zeros():
    shape = (10, 10)
    idx = (0, 0)
    assert sub2ind(shape, *idx) == 0


def test_sub2ind_last_element_2d():
    shape = (4, 5)
    idx = (3, 4)
    expected = 3 + 4*4
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_last_element_3d():
    shape = (3, 3, 3)
    idx = (2, 2, 2)
    expected = 2 + 2*3 + 2*3*3
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_index_out_of_bounds_negative():
    shape = (3, 4)
    idx = (-1, 2)
    with pytest.raises(IndexError):
        sub2ind(shape, *idx)


def test_sub2ind_index_out_of_bounds_too_large():
    shape = (3, 4)
    idx = (3, 2)
    with pytest.raises(IndexError):
        sub2ind(shape, *idx)


def test_sub2ind_index_out_of_bounds_too_large_3d():
    shape = (2, 3, 4)
    idx = (1, 3, 2)
    with pytest.raises(IndexError):
        sub2ind(shape, *idx)


def test_sub2ind_non_integer_index():
    shape = (3, 3)
    idx = (1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(shape, *idx)


def test_sub2ind_no_indices():
    shape = (3, )
    with pytest.raises(TypeError):
        sub2ind(shape)


def test_sub2ind_wrong_number_of_indices():
    shape = (3, 4)
    idx = (1,)
    with pytest.raises(ValueError):
        sub2ind(shape, *idx)


def test_sub2ind_empty_shape():
    shape = ()
    idx = ()
    # 0D array, no indices, expect 0 maybe or error
    with pytest.raises(ValueError):
        sub2ind(shape, *idx)


def test_sub2ind_zero_dimension_in_shape():
    shape = (3, 0, 5)
    idx = (1, 0, 2)
    with pytest.raises(IndexError):
        sub2ind(shape, *idx)


def test_sub2ind_large_indices_valid():
    shape = (100, 100, 100)
    idx = (99, 99, 99)
    expected = 99 + 99*100 + 99*100*100
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_large_indices_invalid():
    shape = (100, 100, 100)
    idx = (100, 99, 99)
    with pytest.raises(IndexError):
        sub2ind(shape, *idx)


def test_sub2ind_float_indices_strict_type():
    shape = (5, 5)
    idx = (2, 3.0)
    with pytest.raises(TypeError):
        sub2ind(shape, *idx)


def test_sub2ind_index_as_list():
    shape = (4, 4)
    index = [2, 3]
    # Passing *index should work, so no error
    expected = 2 + 3*4
    assert sub2ind(shape, *index) == expected


def test_sub2ind_tuple_indices():
    shape = (4, 4)
    idx = (2, 3)
    expected = 2 + 3*4
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_multi_dimensional_with_ones():
    shape = (1, 1, 1)
    idx = (0, 0, 0)
    expected = 0
    assert sub2ind(shape, *idx) == expected


def test_sub2ind_negative_shape_dimension():
    shape = (3, -4)
    idx = (0, 0)
    with pytest.raises(ValueError):
        sub2ind(shape, *idx)
