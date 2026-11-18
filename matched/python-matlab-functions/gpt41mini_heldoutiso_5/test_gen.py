import pytest
from pymlfunc import sub2ind

def test_sub2ind_single_index():
    rows, cols = 5, 4
    row_idx, col_idx = 3, 2
    # Expected linear index: (col_idx-1)*rows + row_idx = (2-1)*5 + 3 = 5 + 3 = 8
    assert sub2ind((rows, cols), row_idx, col_idx) == 8

def test_sub2ind_first_element():
    rows, cols = 10, 10
    assert sub2ind((rows, cols), 1, 1) == 1

def test_sub2ind_last_element():
    rows, cols = 7, 3
    # last element index = rows*cols = 21
    assert sub2ind((rows, cols), rows, cols) == rows * cols

def test_sub2ind_multiple_indices():
    rows, cols = 4, 5
    row_indices = [1, 2, 4]
    col_indices = [1, 3, 5]
    # expected = [(1,1), (2,3), (4,5)]
    # indices: (1-1)*4 +1=1; (3-1)*4 +2=10; (5-1)*4+4=20
    expected = [1, 10, 20]
    result = sub2ind((rows, cols), row_indices, col_indices)
    assert list(result) == expected

def test_sub2ind_with_single_row_array():
    rows, cols = 6, 6
    row_indices = [2]
    col_indices = [5]
    assert sub2ind((rows, cols), row_indices, col_indices) == 26

def test_sub2ind_out_of_bounds_row():
    rows, cols = 4, 4
    with pytest.raises(IndexError):
        sub2ind((rows, cols), 5, 2)

def test_sub2ind_out_of_bounds_col():
    rows, cols = 3, 3
    with pytest.raises(IndexError):
        sub2ind((rows, cols), 1, 4)

def test_sub2ind_zero_index_row():
    rows, cols = 4, 4
    with pytest.raises(IndexError):
        sub2ind((rows, cols), 0, 2)

def test_sub2ind_zero_index_col():
    rows, cols = 4, 4
    with pytest.raises(IndexError):
        sub2ind((rows, cols), 1, 0)

def test_sub2ind_negative_index_row():
    rows, cols = 3, 3
    with pytest.raises(IndexError):
        sub2ind((rows, cols), -1, 2)

def test_sub2ind_negative_index_col():
    rows, cols = 3, 3
    with pytest.raises(IndexError):
        sub2ind((rows, cols), 1, -2)

def test_sub2ind_non_integer_indices():
    rows, cols = 3, 3
    with pytest.raises(TypeError):
        sub2ind((rows, cols), 2.5, 1)

def test_sub2ind_shape_with_invalid_type():
    with pytest.raises(TypeError):
        sub2ind("invalid_shape", 1, 1)

def test_sub2ind_shape_with_too_few_dimensions():
    with pytest.raises(ValueError):
        sub2ind((4,), 1, 1)

def test_sub2ind_shape_with_too_many_dimensions():
    with pytest.raises(ValueError):
        sub2ind((4, 5, 6), 1, 1)

def test_sub2ind_row_and_col_of_different_lengths():
    rows, cols = 5, 5
    row_indices = [1, 2]
    col_indices = [1, 2, 3]
    with pytest.raises(ValueError):
        sub2ind((rows, cols), row_indices, col_indices)

def test_sub2ind_row_and_col_with_non_iterables():
    rows, cols = 6, 6
    # row_indices and col_indices both scalars should work
    assert sub2ind((rows, cols), 3, 4) == (4 - 1) * 6 + 3

def test_sub2ind_row_col_as_numpy_arrays():
    import numpy as np
    rows, cols = 7, 8
    row_indices = np.array([1, 7])
    col_indices = np.array([1, 8])
    expected = [1, 57]  # (1-1)*7 +1=1; (8-1)*7+7=49+7=56+1=57 (check math)
    # Actually (8-1)*7+7 = (7*7)+7=49+7=56, so expected is 56 not 57
    expected = [1, 56]
    result = sub2ind((rows, cols), row_indices, col_indices)
    assert list(result) == expected

def test_sub2ind_empty_indices():
    rows, cols = 3, 3
    assert sub2ind((rows, cols), [], []) == []

def test_sub2ind_with_large_indices():
    rows, cols = 1000, 1000
    r, c = 500, 500
    expected = (c - 1) * rows + r
    assert sub2ind((rows, cols), r, c) == expected
