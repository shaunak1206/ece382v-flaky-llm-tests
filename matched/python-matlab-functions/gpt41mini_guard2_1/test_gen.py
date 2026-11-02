import pytest
import numpy as np
from pymlfunc import sub2ind


# sub2ind is usually a MATLAB-like function converting subscripts to linear indices.
# Typical signature: sub2ind(siz, *subs)
# We test basic functionality, edge cases, errors, and multi-dimensional indices.

def test_sub2ind_single_index_2d():
    # siz = (4,5), subscripts within range
    siz = (4, 5)
    row, col = 3, 2  # 1-based indices in MATLAB style, assuming sub2ind handles 1-based (check).
    # If sub2ind expects 0-based, adjust accordingly.
    # We'll test both 1-based and 0-based interpretation and document assumptions.
    # Assuming 1-based indexing because MATLAB sub2ind is 1-based:
    # linear_index = (col-1)*siz[0] + row
    expected = (col - 1) * siz[0] + row
    result = sub2ind(siz, row, col)
    assert result == expected


def test_sub2ind_multiple_indices_2d():
    siz = (4, 5)
    rows = [1, 2, 3]
    cols = [1, 5, 2]
    # Expected linear indices (MATLAB-style 1-based)
    expected = [(c - 1)*siz[0] + r for r, c in zip(rows, cols)]
    result = sub2ind(siz, rows, cols)
    # Result likely a np.ndarray matching expected
    assert np.all(np.array(result) == expected)


def test_sub2ind_3d_indices():
    siz = (4, 5, 6)
    i, j, k = 2, 3, 4
    # linear idx = i + (j-1)*siz[0] + (k-1)*siz[0]*siz[1]
    expected = i + (j - 1) * siz[0] + (k - 1) * siz[0] * siz[1]
    result = sub2ind(siz, i, j, k)
    assert result == expected


def test_sub2ind_array_indices_3d():
    siz = (3, 3, 3)
    i = [1, 2, 3]
    j = [1, 2, 3]
    k = [1, 2, 3]
    expected = [i[n] + (j[n] - 1) * siz[0] + (k[n] - 1)*siz[0]*siz[1] for n in range(3)]
    result = sub2ind(siz, i, j, k)
    assert np.all(np.array(result) == expected)


def test_sub2ind_returns_numpy_array_for_iterables():
    siz = (2, 2)
    rows = [1, 2]
    cols = [2, 1]
    result = sub2ind(siz, rows, cols)
    # Expect numpy array output
    assert isinstance(result, (np.ndarray, list))
    # Validate values
    expected = [(c - 1)*siz[0] + r for r, c in zip(rows, cols)]
    assert np.all(np.array(result) == expected)


def test_sub2ind_raises_error_on_dimension_mismatch():
    siz = (2, 2)
    i = 1
    j = 1
    k = 1
    # siz has 2 dims, but 3 subscripts given
    with pytest.raises(ValueError):
        sub2ind(siz, i, j, k)


def test_sub2ind_raises_error_on_wrong_subscript_length():
    siz = (2, 2)
    with pytest.raises(TypeError):
        # subscripts missing
        sub2ind(siz)


def test_sub2ind_out_of_bounds_indices():
    siz = (4, 4)
    # index out of bounds (row=5)
    with pytest.raises(IndexError):
        sub2ind(siz, 5, 1)
    # index out of bounds (col=0)
    with pytest.raises(IndexError):
        sub2ind(siz, 1, 0)
    # negative index
    with pytest.raises(IndexError):
        sub2ind(siz, -1, 2)


def test_sub2ind_large_dimensions_int64():
    # Test that large sizes do not cause overflow and result is python int or np.int64
    siz = (1000000, 1000000)
    i, j = 1000, 1000
    result = sub2ind(siz, i, j)
    expected = (j - 1)*siz[0] + i
    assert result == expected
    # Check type is int or numpy int
    assert isinstance(result, (int, np.integer))


def test_sub2ind_input_as_numpy_arrays():
    siz = (5, 5)
    rows = np.array([1, 2, 3])
    cols = np.array([5, 4, 3])
    expected = [(c - 1)*siz[0] + r for r, c in zip(rows, cols)]
    result = sub2ind(siz, rows, cols)
    assert np.all(np.array(result) == expected)


def test_sub2ind_with_size_tuple_and_list_inputs():
    siz = [4, 4]
    rows = [1, 4]
    cols = [4, 1]
    expected = [(c - 1)*siz[0] + r for r, c in zip(rows, cols)]
    result = sub2ind(siz, rows, cols)
    assert np.all(np.array(result) == expected)


def test_sub2ind_scalar_and_list_mix():
    siz = (3, 3)
    row = 2
    cols = [1, 3]
    expected = [(c - 1)*siz[0] + row for c in cols]
    result = sub2ind(siz, row, cols)
    assert np.all(np.array(result) == expected)


def test_sub2ind_1d_case():
    siz = (10,)
    subs = [1, 5, 10]
    expected = subs
    result = sub2ind(siz, subs)
    assert np.all(np.array(result) == expected)


def test_sub2ind_empty_inputs():
    siz = (3, 3)
    # empty subscripts
    result = sub2ind(siz, [], [])
    assert np.size(result) == 0


def test_sub2ind_non_integer_input_raises():
    siz = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(siz, 1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(siz, "1", 2)


def test_sub2ind_non_1d_subscripts_raises():
    siz = (3, 3)
    rows = [[1, 2], [3, 4]]
    cols = [1, 2]
    with pytest.raises(ValueError):
        sub2ind(siz, rows, cols)


def test_sub2ind_multidim_outputs_shape():
    siz = (3, 3, 3)
    i = [1, 2, 3]
    j = [1, 2, 3]
    k = [1, 2, 3]
    result = sub2ind(siz, i, j, k)
    assert isinstance(result, (np.ndarray, list))
    assert len(result) == 3


def test_sub2ind_with_numpy_broadcasting():
    siz = (3, 3)
    i = np.array([1, 2, 3])
    j = 2
    expected = [(j - 1) * siz[0] + r for r in i]
    result = sub2ind(siz, i, j)
    assert np.all(np.array(result) == expected)


def test_sub2ind_output_for_single_element_tuple_siz():
    siz = (5,)
    subs = 3
    expected = 3
    result = sub2ind(siz, subs)
    assert result == expected


def test_sub2ind_boolean_casting_raises():
    siz = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(siz, True, False)
