import pytest
from pymlfunc import sub2ind

# sub2ind expects the shape first, then indices (can be multiple)
# We'll test various inputs and edge cases.

def test_sub2ind_simple_2d():
    shape = (3, 3)
    row = 2
    col = 3
    # Expected linear index is (col-1)*3 + (row-1) = (3-1)*3 + (2-1) = 2*3 + 1 = 7 (1-based)
    # So zero-based linear index is 6, so sub2ind should return 7.
    assert sub2ind(shape, row, col) == 7

def test_sub2ind_simple_3d():
    shape = (2, 3, 4)
    i, j, k = 1, 2, 3
    # MATLAB sub2ind linear index calculation:
    # idx = i + (j-1)*d1 + (k-1)*d1*d2
    # Here d1=2, d2=3, d3=4
    # idx = 1 + (2-1)*2 + (3-1)*2*3 = 1 + 2 + 12 = 15 (1-based)
    assert sub2ind(shape, i, j, k) == 15

def test_sub2ind_with_tuple_indices():
    shape = (4, 5)
    indices = (3, 4)
    assert sub2ind(shape, *indices) == sub2ind(shape, 3, 4)

def test_sub2ind_return_type_int():
    shape = (10, 10)
    result = sub2ind(shape, 5, 5)
    assert isinstance(result, int)

def test_sub2ind_all_ones_indices():
    shape = (3, 3, 3)
    indices = (1, 1, 1)
    # First element linear index is always 1 in MATLAB style (1-based)
    assert sub2ind(shape, *indices) == 1

def test_sub2ind_large_shape_and_indices():
    shape = (100, 100, 100)
    indices = (50, 20, 10)
    # Calculate expected index manually:
    d1, d2, d3 = shape
    i, j, k = indices
    expected = i + (j-1)*d1 + (k-1)*d1*d2
    assert sub2ind(shape, *indices) == expected

def test_sub2ind_indices_equal_shape():
    shape = (4, 4, 4)
    indices = (4, 4, 4)
    d1, d2, d3 = shape
    i, j, k = indices
    expected = i + (j-1)*d1 + (k-1)*d1*d2
    assert sub2ind(shape, *indices) == expected

def test_sub2ind_index_zero_should_fail():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 2)

def test_sub2ind_index_out_of_bounds_should_fail():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, 4, 2)
    with pytest.raises(IndexError):
        sub2ind(shape, 2, 4)

def test_sub2ind_negative_index_should_fail():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, -1, 2)

def test_sub2ind_incorrect_number_of_indices_should_fail():
    shape = (3, 3, 3)
    # Provide only 2 indices for 3D shape
    with pytest.raises(ValueError):
        sub2ind(shape, 1, 2)

def test_sub2ind_with_float_indices_should_fail():
    shape = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(shape, 1.5, 2)

def test_sub2ind_string_shape_should_fail():
    with pytest.raises(TypeError):
        sub2ind("34", 1, 2)

def test_sub2ind_string_index_should_fail():
    shape = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(shape, "1", 2)

def test_sub2ind_single_dimension():
    shape = (5,)
    for i in range(1, 6):
        assert sub2ind(shape, i) == i

def test_sub2ind_zero_dimensions_should_fail():
    with pytest.raises(ValueError):
        sub2ind((),)

def test_sub2ind_shape_with_zero_dimension_should_fail():
    with pytest.raises(ValueError):
        sub2ind((3, 0), 1, 1)

def test_sub2ind_indices_outside_shape_should_fail():
    shape = (2, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, 1, 4)

def test_sub2ind_shape_not_tuple_should_fail():
    with pytest.raises(TypeError):
        sub2ind([3, 3], 1, 1)

def test_sub2ind_indices_not_int_should_fail():
    shape = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(shape, 1, 1.1)

def test_sub2ind_indices_more_than_dimensions_should_fail():
    shape = (3, 3)
    with pytest.raises(ValueError):
        sub2ind(shape, 1, 1, 1)
