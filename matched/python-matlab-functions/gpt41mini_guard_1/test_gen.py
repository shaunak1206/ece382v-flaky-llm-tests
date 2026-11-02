import pytest
from pymlfunc import sub2ind

def test_sub2ind_single_dim_index():
    # 1D case, size is a single integer
    assert sub2ind((10,), 5) == 5
    assert sub2ind((10,), 1) == 1
    assert sub2ind((10,), 10) == 10

def test_sub2ind_2d_basic():
    # 2D matrix 3x4, valid subscripts
    siz = (3, 4)
    assert sub2ind(siz, 1, 1) == 1
    # last element
    assert sub2ind(siz, 3, 4) == 12
    # middle element
    assert sub2ind(siz, 2, 3) == 8

def test_sub2ind_2d_vectorized():
    siz = (2, 3)
    rows = [1, 2, 1]
    cols = [1, 3, 2]
    expected = [1, 6, 3]
    # Passing lists for vectorized indexing
    assert list(sub2ind(siz, rows, cols)) == expected

def test_sub2ind_3d_basic():
    siz = (2, 2, 2)
    # Element at first subscript for each dim
    assert sub2ind(siz, 1, 1, 1) == 1
    # Element at last subscript for each dim
    assert sub2ind(siz, 2, 2, 2) == 8
    # Middle element
    assert sub2ind(siz, 2, 1, 2) == 6

def test_sub2ind_invalid_subscript_raises():
    siz = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(siz, 0, 1)  # subscripts are 1-based, 0 invalid
    with pytest.raises(IndexError):
        sub2ind(siz, 4, 2)
    with pytest.raises(IndexError):
        sub2ind(siz, 2, 4)

def test_sub2ind_return_type_int_for_scalar():
    # Should return int, not tuple or list for scalar input
    siz = (5, 5)
    idx = sub2ind(siz, 3, 4)
    assert isinstance(idx, int)

def test_sub2ind_input_as_tuple_subscripts():
    # Passing tuple for subscripts explicitly
    siz = (3, 3)
    # sub2ind expects subscripts as separate arguments, so tuple unpacked
    idx = sub2ind(siz, *(2, 2))
    assert idx == 5

def test_sub2ind_vectorized_return_type_list():
    siz = (4, 4)
    rows = [1, 2, 3]
    cols = [1, 4, 2]
    result = sub2ind(siz, rows, cols)
    # Should return list or at least iterable of integers
    assert all(isinstance(i, int) for i in result)

def test_sub2ind_extra_dimensions():
    siz = (2, 2, 2, 2)
    # subscripts for all 4 dims
    assert sub2ind(siz, 1, 1, 1, 1) == 1
    assert sub2ind(siz, 2, 2, 2, 2) == 16

def test_sub2ind_one_dimension_size():
    # siz with one dimension being 1
    siz = (1, 3)
    assert sub2ind(siz, 1, 1) == 1
    assert sub2ind(siz, 1, 3) == 3

def test_sub2ind_edge_case_empty_siz():
    with pytest.raises(TypeError):
        sub2ind((), 1)

def test_sub2ind_dimension_mismatch_raises():
    siz = (3, 3)
    # Passing too many subscripts
    with pytest.raises(TypeError):
        sub2ind(siz, 1, 1, 1)
    # Passing too few subscripts
    with pytest.raises(TypeError):
        sub2ind(siz, 1)

def test_sub2ind_negative_index_raises():
    siz = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(siz, -1, 2)

def test_sub2ind_float_subscripts_raises():
    # subscripts must be integers
    siz = (5, 5)
    with pytest.raises(TypeError):
        sub2ind(siz, 2.5, 3)

def test_sub2ind_large_indices():
    # Large size array, large subscripts within bounds
    siz = (1000, 1000)
    assert sub2ind(siz, 1000, 1000) == 1_000_000
    assert sub2ind(siz, 1, 1) == 1

def test_sub2ind_non_integer_siz_raises():
    with pytest.raises(TypeError):
        sub2ind((3.5, 4), 1, 1)

def test_sub2ind_tuple_of_lists_subscripts():
    siz = (3, 3)
    rows = (1, 2, 3)
    cols = (3, 2, 1)
    result = list(sub2ind(siz, rows, cols))
    expected = [7, 5, 3]
    assert result == expected
