import numpy as np

from pymlfunc import sub2ind


def test_docstring_example_zero_based():
    mat = np.arange(1, 31).reshape((5, 6))
    ind = sub2ind(mat.shape, np.array([0, 2]), np.array([1, 3]))
    np.testing.assert_array_equal(ind, np.array([1, 15]))


def test_row_major_flat_index_for_every_cell():
    shape = (3, 4)
    rows, cols = np.indices(shape)
    ind = sub2ind(shape, rows.ravel(), cols.ravel())
    np.testing.assert_array_equal(ind, np.arange(12))


def test_matches_numpy_ravel_multi_index_on_random_in_bounds_coords():
    rng = np.random.RandomState(0)
    shape = (7, 11)
    rows = rng.randint(0, shape[0], size=50)
    cols = rng.randint(0, shape[1], size=50)
    expected = np.ravel_multi_index((rows, cols), shape)
    np.testing.assert_array_equal(sub2ind(shape, rows, cols), expected)


def test_round_trip_with_unravel_index():
    shape = (4, 6)
    flat = np.arange(24)
    rows, cols = np.unravel_index(flat, shape)
    np.testing.assert_array_equal(sub2ind(shape, rows, cols), flat)


def test_top_left_and_bottom_right_corners():
    ind = sub2ind((5, 6), np.array([0, 4]), np.array([0, 5]))
    np.testing.assert_array_equal(ind, np.array([0, 29]))


def test_negative_row_gives_minus_one():
    ind = sub2ind((5, 6), np.array([-1]), np.array([0]))
    np.testing.assert_array_equal(ind, np.array([-1]))


def test_negative_col_in_first_row_gives_minus_one():
    ind = sub2ind((5, 6), np.array([0]), np.array([-1]))
    np.testing.assert_array_equal(ind, np.array([-1]))


def test_row_equal_to_num_rows_gives_minus_one():
    # flat index 5 * 6 + 0 == 30 == number of elements
    ind = sub2ind((5, 6), np.array([5]), np.array([0]))
    np.testing.assert_array_equal(ind, np.array([-1]))


def test_upper_boundary_is_exclusive():
    # 4 * 6 + 5 == 29 is the last valid index; 5 * 6 + 0 == 30 is not
    ind = sub2ind((5, 6), np.array([4, 5]), np.array([5, 0]))
    np.testing.assert_array_equal(ind, np.array([29, -1]))


def test_mixed_valid_and_invalid_entries_are_masked_elementwise():
    shape = (4, 5)
    rows = np.array([-1, 0, 3, 4, 2])
    cols = np.array([0, 0, 4, 0, -2])
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, np.array([-1, 0, 19, -1, 8]))


def test_uses_number_of_columns_as_stride_for_non_square():
    np.testing.assert_array_equal(sub2ind((2, 3), np.array([1]), np.array([0])), [3])
    np.testing.assert_array_equal(sub2ind((3, 2), np.array([1]), np.array([0])), [2])


def test_shape_as_tuple_list_or_ndarray_all_work():
    rows = np.array([1])
    cols = np.array([2])
    for shape in [(3, 4), [3, 4], np.array([3, 4])]:
        np.testing.assert_array_equal(sub2ind(shape, rows, cols), [6])


def test_preserves_2d_input_shape():
    rows = np.array([[0, 1], [2, 3]])
    cols = np.array([[0, 1], [2, 3]])
    ind = sub2ind((4, 4), rows, cols)
    assert ind.shape == (2, 2)
    np.testing.assert_array_equal(ind, np.array([[0, 5], [10, 15]]))


def test_broadcasts_column_array_against_row_vector():
    rows = np.array([0, 1, 2])
    cols = np.array([1])
    ind = sub2ind((3, 4), rows, cols)
    np.testing.assert_array_equal(ind, np.array([1, 5, 9]))


def test_empty_index_arrays_return_empty_result():
    empty = np.array([], dtype=int)
    ind = sub2ind((3, 4), empty, empty)
    assert ind.shape == (0,)


def test_inputs_are_not_modified():
    rows = np.array([-1, 0, 9])
    cols = np.array([0, 2, 0])
    rows_before = rows.copy()
    cols_before = cols.copy()
    sub2ind((3, 4), rows, cols)
    np.testing.assert_array_equal(rows, rows_before)
    np.testing.assert_array_equal(cols, cols_before)


def test_integer_inputs_give_integer_output():
    ind = sub2ind((3, 4), np.array([0, 2]), np.array([1, 3]))
    assert np.issubdtype(ind.dtype, np.integer)


def test_every_output_is_minus_one_or_a_valid_flat_index():
    shape = (3, 4)
    total = shape[0] * shape[1]
    rng = np.random.RandomState(1)
    rows = rng.randint(-3, 6, size=200)
    cols = rng.randint(-3, 7, size=200)
    ind = sub2ind(shape, rows, cols)
    valid = (ind == -1) | ((ind >= 0) & (ind < total))
    assert valid.all()


def test_zero_sized_array_masks_everything():
    ind = sub2ind((0, 3), np.array([0, 0]), np.array([0, 2]))
    np.testing.assert_array_equal(ind, np.array([-1, -1]))


def test_single_row_and_single_column_shapes():
    np.testing.assert_array_equal(
        sub2ind((1, 5), np.array([0, 0]), np.array([4, 5])), np.array([4, -1]))
    np.testing.assert_array_equal(
        sub2ind((5, 1), np.array([3, 5]), np.array([0, 0])), np.array([3, -1]))


def test_large_shape_indices():
    shape = (1000, 1000)
    rows = np.array([999, 500])
    cols = np.array([999, 1])
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, np.array([999 * 1000 + 999, 500 * 1000 + 1]))
