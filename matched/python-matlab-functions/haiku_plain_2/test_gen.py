"""Unit tests for pymlfunc.sub2ind.

sub2ind maps 0-based (row, col) pairs to row-major flat indices
(row * ncols + col). Any index that is negative or at/after rows * cols
is replaced with -1.
"""
import numpy as np

from pymlfunc import ind2sub, sub2ind


def _expected_flat(shape, rows, cols):
    return np.ravel_multi_index((np.asarray(rows), np.asarray(cols)), shape)


def test_docstring_example_values():
    ind = sub2ind((5, 6), np.array([0, 2]), np.array([1, 3]))
    assert ind.tolist() == [1, 15]


def test_result_is_ndarray_with_input_shape():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 0, 0])
    ind = sub2ind((3, 4), rows, cols)
    assert isinstance(ind, np.ndarray)
    assert ind.shape == (3,)
    assert ind.tolist() == [0, 4, 8]


def test_matches_ravel_multi_index_for_every_cell():
    shape = (4, 7)
    rr, cc = np.meshgrid(np.arange(4), np.arange(7), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    ind = sub2ind(shape, rows, cols)
    np.testing.assert_array_equal(ind, _expected_flat(shape, rows, cols))
    assert sorted(ind.tolist()) == list(range(28))


def test_first_and_last_cell():
    ind = sub2ind((3, 4), np.array([0, 2]), np.array([0, 3]))
    assert ind.tolist() == [0, 11]


def test_row_stride_is_column_count_for_wide_shape():
    # shape (2, 7): one row step moves 7 flat positions
    ind = sub2ind((2, 7), np.array([1]), np.array([0]))
    assert ind.tolist() == [7]


def test_row_stride_is_column_count_for_tall_shape():
    # shape (7, 2): one row step moves 2 flat positions, not 7
    ind = sub2ind((7, 2), np.array([1]), np.array([0]))
    assert ind.tolist() == [2]


def test_single_row_shape():
    ind = sub2ind((1, 5), np.zeros(5, dtype=int), np.arange(5))
    assert ind.tolist() == [0, 1, 2, 3, 4]


def test_single_column_shape():
    ind = sub2ind((5, 1), np.arange(5), np.zeros(5, dtype=int))
    assert ind.tolist() == [0, 1, 2, 3, 4]


def test_one_by_one_shape():
    ind = sub2ind((1, 1), np.array([0]), np.array([0]))
    assert ind.tolist() == [0]


def test_row_at_or_past_row_count_is_minus_one():
    ind = sub2ind((3, 4), np.array([3, 5]), np.array([0, 0]))
    assert ind.tolist() == [-1, -1]


def test_last_valid_cell_kept_and_next_flat_index_is_minus_one():
    ind = sub2ind((3, 4), np.array([2, 3]), np.array([3, 0]))
    assert ind.tolist() == [11, -1]


def test_negative_row_is_minus_one():
    ind = sub2ind((3, 4), np.array([-1, -2]), np.array([0, 0]))
    assert ind.tolist() == [-1, -1]


def test_negative_column_on_first_row_is_minus_one():
    ind = sub2ind((3, 4), np.array([0]), np.array([-1]))
    assert ind.tolist() == [-1]


def test_mixed_valid_and_invalid_entries():
    rows = np.array([0, 1, -1, 5, 2])
    cols = np.array([0, 2, 0, 0, 3])
    ind = sub2ind((3, 4), rows, cols)
    assert ind.tolist() == [0, 6, -1, -1, 11]


def test_inputs_are_not_modified():
    rows = np.array([0, -1, 9])
    cols = np.array([1, 0, 0])
    rows_before = rows.copy()
    cols_before = cols.copy()
    sub2ind((3, 4), rows, cols)
    np.testing.assert_array_equal(rows, rows_before)
    np.testing.assert_array_equal(cols, cols_before)


def test_integer_inputs_give_integer_output():
    ind = sub2ind((3, 4), np.array([0, 1]), np.array([0, 1]))
    assert ind.dtype.kind == "i"


def test_float_inputs_keep_float_values_and_minus_one_sentinel():
    ind = sub2ind((3, 4), np.array([0.0, 1.0, 3.0]), np.array([1.0, 2.0, 0.0]))
    assert ind.dtype.kind == "f"
    np.testing.assert_array_equal(ind, np.array([1.0, 6.0, -1.0]))


def test_empty_inputs_return_empty_array():
    ind = sub2ind((3, 4), np.array([], dtype=int), np.array([], dtype=int))
    assert isinstance(ind, np.ndarray)
    assert ind.size == 0


def test_two_dimensional_inputs_keep_shape():
    rows = np.array([[0, 1], [2, 2]])
    cols = np.array([[3, 0], [1, 2]])
    ind = sub2ind((3, 4), rows, cols)
    assert ind.shape == (2, 2)
    assert ind.tolist() == [[3, 4], [9, 10]]


def test_broadcasting_column_and_row_vectors():
    rows = np.arange(3).reshape(3, 1)
    cols = np.arange(4).reshape(1, 4)
    ind = sub2ind((3, 4), rows, cols)
    assert ind.shape == (3, 4)
    np.testing.assert_array_equal(ind, np.arange(12).reshape(3, 4))


def test_array_shape_given_as_list():
    rows = np.array([1, 2])
    cols = np.array([2, 0])
    assert sub2ind([3, 4], rows, cols).tolist() == [6, 8]


def test_zero_row_shape_marks_everything_invalid():
    ind = sub2ind((0, 5), np.array([0, 0]), np.array([0, 4]))
    assert ind.tolist() == [-1, -1]


def test_large_shape_last_cell_and_overflow():
    rows = np.array([999, 1000])
    cols = np.array([999, 0])
    ind = sub2ind((1000, 1000), rows, cols)
    assert ind.tolist() == [999999, -1]


def test_round_trip_with_ind2sub():
    shape = (4, 5)
    rows = np.array([0, 3, 2, 1])
    cols = np.array([4, 0, 2, 3])
    ind = sub2ind(shape, rows, cols)
    rows_back, cols_back = ind2sub(shape, ind)
    np.testing.assert_array_equal(rows_back, rows)
    np.testing.assert_array_equal(cols_back, cols)
