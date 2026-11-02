import pytest
from pymlfunc import sub2ind

# The sub2ind function signature is assumed to be:
# sub2ind(array_shape, *subscripts)
# where array_shape is a tuple of dimension sizes
# and subscripts are one or more index arrays or ints, one per dimension.
#
# The function returns a flat (linear) index or an array of linear indices.
#
# Tests cover:
# - basic indexing correctness for 1D, 2D, 3D
# - out-of-bounds subscripts raise errors
# - broadcasting of numpy arrays for multi-dimensional indexing
# - compatibility of input types (ints, lists, tuples, numpy arrays)
# - behavior on edge indices (first and last)
# - shape and dtype of output
#
# No randomness involved, so no seeding is needed.


import numpy as np


def test_sub2ind_1d_basic():
    # 1D shape, single int subscript returns the same
    shape = (10,)
    idx = sub2ind(shape, 3)
    assert isinstance(idx, int)
    assert idx == 3


def test_sub2ind_1d_array_subscript():
    shape = (5,)
    subs = [0, 1, 2, 3, 4]
    idx = sub2ind(shape, subs)
    assert np.allclose(idx, np.arange(5))


def test_sub2ind_2d_basic():
    shape = (3, 4)
    row = 1
    col = 2
    # MATLAB-style linear indexing = col * rows + row
    # Python zero-based indexing, row-major order:
    # Usually row-major linear idx = row * ncols + col
    expected_idx = row * shape[1] + col
    idx = sub2ind(shape, row, col)
    assert idx == expected_idx


def test_sub2ind_2d_array_subscripts():
    shape = (2, 3)
    rows = [0, 1, 0]
    cols = [0, 1, 2]
    expected_idx = np.array(rows) * shape[1] + np.array(cols)
    idx = sub2ind(shape, rows, cols)
    assert np.all(idx == expected_idx)


def test_sub2ind_3d_basic():
    shape = (2, 3, 4)
    i, j, k = 1, 2, 3
    # linear index: i * (3*4) + j*4 + k
    expected_idx = i * shape[1] * shape[2] + j * shape[2] + k
    idx = sub2ind(shape, i, j, k)
    assert idx == expected_idx


def test_sub2ind_3d_array_broadcasting():
    shape = (2, 2, 2)
    i = np.array([0, 1])
    j = np.array([1, 0])
    k = 1
    # Expected: linear indices for (0,1,1) and (1,0,1)
    expected_idx = i * (shape[1] * shape[2]) + j * shape[2] + k
    idx = sub2ind(shape, i, j, k)
    assert np.all(idx == expected_idx)
    assert idx.shape == (2,)


def test_sub2ind_out_of_bounds_raises():
    shape = (3, 3)
    # Negative index (should fail)
    with pytest.raises(IndexError):
        sub2ind(shape, -1, 0)
    # Index equal to size (should fail)
    with pytest.raises(IndexError):
        sub2ind(shape, 3, 0)
    # Second dimension too large
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 3)


def test_sub2ind_all_dims_too_few_subscripts():
    # shape 3 dims, only 2 indices should raise
    shape = (2, 2, 2)
    with pytest.raises(ValueError):
        sub2ind(shape, 0, 1)


def test_sub2ind_all_dims_too_many_subscripts():
    # shape 2 dims, 3 subscripts should raise
    shape = (2, 3)
    with pytest.raises(ValueError):
        sub2ind(shape, 0, 1, 2)


def test_sub2ind_single_dim_subscript_as_tuple():
    shape = (4,)
    idx = sub2ind(shape, (2,))
    assert idx == 2


def test_sub2ind_multi_dim_subscripts_as_tuples_and_lists():
    shape = (3, 3)
    rows = (0, 1, 2)
    cols = [2, 1, 0]
    expected_idx = np.array(rows) * shape[1] + np.array(cols)
    idx = sub2ind(shape, rows, cols)
    assert np.all(idx == expected_idx)


def test_sub2ind_output_dtype_and_type():
    shape = (3, 3)
    rows = np.array([0, 1])
    cols = np.array([1, 2])
    idx = sub2ind(shape, rows, cols)
    assert isinstance(idx, np.ndarray)
    assert idx.dtype.kind in 'iu'  # integer type


def test_sub2ind_zero_indices():
    shape = (5, 5)
    idx = sub2ind(shape, 0, 0)
    assert idx == 0


def test_sub2ind_max_indices():
    shape = (4, 4, 4)
    idx = sub2ind(shape, 3, 3, 3)
    expected_idx = 3 * shape[1] * shape[2] + 3 * shape[2] + 3
    assert idx == expected_idx


def test_sub2ind_broadcasting_mismatched_shapes_raises():
    shape = (2, 2)
    a = np.array([0, 1])
    b = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        sub2ind(shape, a, b)


def test_sub2ind_single_index_array_for_1_dim_shape():
    shape = (6,)
    sub = np.array([5, 3, 1])
    idx = sub2ind(shape, sub)
    assert np.all(idx == sub)


def test_sub2ind_negative_index_explicit_out_of_bounds():
    shape = (4, 5)
    # Negative second dimension
    with pytest.raises(IndexError):
        sub2ind(shape, 2, -1)


def test_sub2ind_empty_array_subscripts():
    shape = (3, 3)
    empty = np.array([])
    idx = sub2ind(shape, empty, empty)
    assert idx.shape == empty.shape
    assert idx.size == 0


def test_sub2ind_single_element_ndarray_inputs():
    shape = (4, 4)
    r = np.array([2])
    c = np.array([1])
    idx = sub2ind(shape, r, c)
    expected = 2 * shape[1] + 1
    # Output should be an array with one element equal to expected
    assert isinstance(idx, np.ndarray)
    assert idx.shape == (1,)
    assert idx[0] == expected
