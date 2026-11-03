import pytest
from pymlfunc import sub2ind


def test_sub2ind_basic():
    # 2D array, simple indices
    dims = (3, 4)
    rows = [1, 2]
    cols = [1, 3]
    expected = [0, 7]  # zero-based linear indices
    result = sub2ind(dims, rows, cols)
    assert result == expected


def test_sub2ind_single_index():
    # 1D array, single index
    dims = (5,)
    rows = [3]
    expected = [3]
    result = sub2ind(dims, rows)
    assert result == expected


def test_sub2ind_3d_indices():
    dims = (2, 3, 4)
    i = [1, 2, 1]
    j = [1, 2, 3]
    k = [1, 1, 4]
    # Manually calculating linear indices with zero-based:
    # ind = i-1 + (j-1)*dim1 + (k-1)*dim1*dim2
    # i,j,k in 1-based, convert
    expected = []
    for ii, jj, kk in zip(i, j, k):
        idx = (ii - 1) + (jj - 1) * dims[0] + (kk - 1) * dims[0] * dims[1]
        expected.append(idx)
    result = sub2ind(dims, i, j, k)
    assert result == expected


def test_sub2ind_multiple_points():
    # multiple points with mixed indices
    dims = (5, 5)
    row_indices = [1, 5, 3]
    col_indices = [5, 1, 3]
    expected = [(r - 1) + (c - 1) * dims[0] for r, c in zip(row_indices, col_indices)]
    result = sub2ind(dims, row_indices, col_indices)
    assert result == expected


def test_sub2ind_single_value_each_dim():
    dims = (4, 6, 8)
    i = 1
    j = 1
    k = 1
    expected = [0]
    result = sub2ind(dims, i, j, k)
    assert result == expected


def test_sub2ind_index_out_of_bounds_raises():
    dims = (3, 3)
    rows = [0]  # zero or less invalid in MATLAB style 1-based indices
    cols = [1]
    with pytest.raises(ValueError):
        sub2ind(dims, rows, cols)

    rows = [4]
    cols = [1]
    with pytest.raises(ValueError):
        sub2ind(dims, rows, cols)

    rows = [1]
    cols = [0]
    with pytest.raises(ValueError):
        sub2ind(dims, rows, cols)

    rows = [1]
    cols = [4]
    with pytest.raises(ValueError):
        sub2ind(dims, rows, cols)


def test_sub2ind_dims_length_mismatch_raises():
    dims = (3, 3, 3)
    # Only 2 subscripts passed for 3 dimensions
    with pytest.raises(TypeError):
        sub2ind(dims, [1], [1])

    # 4 subscripts for 3 dims
    with pytest.raises(TypeError):
        sub2ind(dims, [1], [1], [1], [1])


def test_sub2ind_indices_length_mismatch_raises():
    dims = (3, 3)
    with pytest.raises(ValueError):
        sub2ind(dims, [1, 2], [1])  # different lengths

    with pytest.raises(ValueError):
        sub2ind(dims, [1], [1, 2])  # different lengths


def test_sub2ind_non_integer_index_raises():
    dims = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(dims, [1.5], [2])

    with pytest.raises(TypeError):
        sub2ind(dims, [1], ['a'])


def test_sub2ind_with_single_int_args():
    dims = (3, 3)
    r = 2
    c = 3
    expected = [(r-1) + (c-1)*dims[0]]
    result = sub2ind(dims, r, c)
    assert result == expected


def test_sub2ind_empty_indices():
    dims = (3, 3)
    result = sub2ind(dims, [], [])
    assert result == []


def test_sub2ind_non_tuple_dims_raises():
    # dims must be a tuple
    with pytest.raises(TypeError):
        sub2ind([3, 3], [1], [1])


def test_sub2ind_with_dims_length_1_and_multiple_indices():
    dims = (5,)
    indices = [1, 3, 5]
    expected = [i-1 for i in indices]
    result = sub2ind(dims, indices)
    assert result == expected


def test_sub2ind_large_dimensions_and_indices():
    dims = (100, 100, 100)
    i = [1, 50, 100]
    j = [1, 50, 100]
    k = [1, 50, 100]
    expected = [(ii-1) + (jj-1)*dims[0] + (kk-1)*dims[0]*dims[1] for ii, jj, kk in zip(i, j, k)]
    result = sub2ind(dims, i, j, k)
    assert result == expected


def test_sub2ind_single_element_dims_and_indices():
    dims = (1,)
    idx = 1
    expected = [0]
    result = sub2ind(dims, idx)
    assert result == expected


def test_sub2ind_all_indices_are_one_should_return_zero():
    dims = (10, 10, 10)
    idx = [1]
    expected = [0]
    result = sub2ind(dims, *([idx] * len(dims)))
    assert result == expected
