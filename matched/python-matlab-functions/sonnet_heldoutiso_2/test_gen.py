import numpy as np
import pytest

from pymlfunc import sub2ind


def _arr(values):
    return np.array(values, dtype=int)


def test_importable_and_callable():
    assert callable(sub2ind)


def test_first_element_is_zero():
    result = sub2ind((3, 4), _arr([0]), _arr([0]))
    assert np.asarray(result).tolist() == [0]


def test_last_element_is_size_minus_one():
    result = sub2ind((3, 4), _arr([2]), _arr([3]))
    assert np.asarray(result).tolist() == [11]


def test_row_major_ordering_within_first_row():
    rows = _arr([0, 0, 0, 0])
    cols = _arr([0, 1, 2, 3])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [0, 1, 2, 3]


def test_row_major_step_between_rows():
    rows = _arr([0, 1, 2])
    cols = _arr([0, 0, 0])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [0, 4, 8]


def test_general_values():
    rows = _arr([1, 2, 0])
    cols = _arr([2, 1, 3])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [6, 9, 3]


def test_matches_numpy_ravel_multi_index():
    shape = (5, 7)
    rows = _arr([0, 1, 2, 3, 4, 4, 2])
    cols = _arr([0, 6, 3, 2, 6, 0, 5])
    expected = np.ravel_multi_index((rows, cols), shape)
    result = sub2ind(shape, rows.copy(), cols.copy())
    assert np.asarray(result).tolist() == expected.tolist()


def test_matches_numpy_flat_indexing():
    shape = (4, 6)
    data = np.arange(shape[0] * shape[1]).reshape(shape)
    rows = _arr([0, 3, 2, 1])
    cols = _arr([5, 0, 3, 4])
    idx = np.asarray(sub2ind(shape, rows.copy(), cols.copy()))
    assert data.ravel()[idx].tolist() == data[rows, cols].tolist()


def test_square_matrix():
    rows = _arr([0, 1, 2])
    cols = _arr([0, 1, 2])
    result = sub2ind((3, 3), rows, cols)
    assert np.asarray(result).tolist() == [0, 4, 8]


def test_single_column_array():
    rows = _arr([0, 1, 2, 3])
    cols = _arr([0, 0, 0, 0])
    result = sub2ind((4, 1), rows, cols)
    assert np.asarray(result).tolist() == [0, 1, 2, 3]


def test_single_row_array():
    rows = _arr([0, 0, 0])
    cols = _arr([0, 1, 2])
    result = sub2ind((1, 3), rows, cols)
    assert np.asarray(result).tolist() == [0, 1, 2]


def test_output_length_matches_input_length():
    rows = _arr([0, 1, 2, 1, 0])
    cols = _arr([1, 1, 1, 2, 2])
    result = sub2ind((3, 4), rows, cols)
    assert len(result) == 5


def test_out_of_range_row_gives_minus_one():
    rows = _arr([3])
    cols = _arr([0])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [-1]


def test_negative_index_gives_minus_one():
    rows = _arr([-1])
    cols = _arr([0])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [-1]


def test_mixed_valid_and_invalid():
    rows = _arr([0, 5, 2])
    cols = _arr([1, 0, 3])
    result = np.asarray(sub2ind((3, 4), rows, cols))
    assert result[0] == 1
    assert result[1] == -1
    assert result[2] == 11
