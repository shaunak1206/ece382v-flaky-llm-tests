import numpy as np
import pytest

from pymlfunc import sub2ind
from pymlfunc.subind import sub2ind as sub2ind_direct


def test_import_paths_are_same_function():
    assert sub2ind is sub2ind_direct


def test_docstring_example():
    mat = np.arange(1, 31).reshape((5, 6))
    ind = sub2ind(mat.shape, np.array([0, 2]), np.array([1, 3]))
    np.testing.assert_array_equal(ind, np.array([1, 15]))


def test_matches_numpy_ravel_multi_index():
    shape = (4, 7)
    rows = np.array([0, 1, 2, 3, 3, 0])
    cols = np.array([0, 6, 3, 0, 6, 5])
    expected = np.ravel_multi_index((rows, cols), shape)
    result = sub2ind(shape, rows.copy(), cols.copy())
    np.testing.assert_array_equal(result, expected)


def test_indexing_flattened_array_gives_matrix_elements():
    mat = np.arange(20).reshape((4, 5))
    rows = np.array([0, 2, 3])
    cols = np.array([4, 1, 2])
    ind = sub2ind(mat.shape, rows, cols)
    np.testing.assert_array_equal(mat.ravel()[ind], mat[rows, cols])


def test_origin_maps_to_zero():
    ind = sub2ind((3, 4), np.array([0]), np.array([0]))
    assert ind[0] == 0


def test_last_element_maps_to_size_minus_one():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([2]), np.array([3]))
    assert ind[0] == shape[0] * shape[1] - 1


def test_row_major_ordering():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([1, 1, 2]), np.array([0, 3, 0]))
    np.testing.assert_array_equal(ind, np.array([4, 7, 8]))


def test_out_of_range_row_is_minus_one():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([3]), np.array([0]))
    assert ind[0] == -1


def test_out_of_range_large_index_is_minus_one():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([10, 2]), np.array([10, 3]))
    np.testing.assert_array_equal(ind, np.array([-1, 11]))


def test_negative_result_is_minus_one():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([0]), np.array([-5]))
    assert ind[0] == -1


def test_negative_row_gives_minus_one_when_total_negative():
    shape = (3, 4)
    ind = sub2ind(shape, np.array([-1]), np.array([0]))
    assert ind[0] == -1


def test_mixed_valid_and_invalid():
    shape = (2, 3)
    rows = np.array([0, 1, 2, -1, 1])
    cols = np.array([0, 2, 0, 0, 1])
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, np.array([0, 5, -1, -1, 4]))


def test_empty_arrays_return_empty():
    ind = sub2ind((3, 4), np.array([], dtype=int), np.array([], dtype=int))
    assert ind.shape == (0,)


def test_two_dimensional_inputs_preserve_shape():
    shape = (3, 4)
    rows = np.array([[0, 1], [2, 2]])
    cols = np.array([[0, 1], [2, 3]])
    ind = sub2ind(shape, rows, cols)
    assert ind.shape == (2, 2)
    np.testing.assert_array_equal(ind, np.array([[0, 5], [10, 11]]))


def test_single_column_matrix():
    shape = (5, 1)
    rows = np.array([0, 2, 4])
    cols = np.array([0, 0, 0])
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, np.array([0, 2, 4]))


def test_single_row_matrix():
    shape = (1, 5)
    rows = np.array([0, 0, 0])
    cols = np.array([0, 2, 4])
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, np.array([0, 2, 4]))


def test_scalar_inputs_raise_type_error():
    with pytest.raises(TypeError):
        sub2ind((3, 4), 1, 2)


def test_result_dtype_is_integer():
    ind = sub2ind((3, 4), np.array([0, 1]), np.array([1, 2]))
    assert np.issubdtype(ind.dtype, np.integer)
