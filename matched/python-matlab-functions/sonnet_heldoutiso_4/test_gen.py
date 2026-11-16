import numpy as np
import pytest

from pymlfunc import sub2ind


def _as_list(result):
    return np.asarray(result).ravel().tolist()


def test_origin_maps_to_zero():
    result = sub2ind((3, 4), np.array([0]), np.array([0]))
    assert _as_list(result) == [0]


def test_last_element_maps_to_size_minus_one():
    result = sub2ind((3, 4), np.array([2]), np.array([3]))
    assert _as_list(result) == [11]


def test_diagonal_indices_row_major():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    result = sub2ind((3, 4), rows, cols)
    assert _as_list(result) == [0, 5, 10]


def test_first_row_equals_column_indices():
    cols = np.arange(5)
    rows = np.zeros(5, dtype=int)
    result = sub2ind((4, 5), rows, cols)
    assert _as_list(result) == [0, 1, 2, 3, 4]


def test_first_column_steps_by_number_of_columns():
    rows = np.arange(4)
    cols = np.zeros(4, dtype=int)
    result = sub2ind((4, 5), rows, cols)
    assert _as_list(result) == [0, 5, 10, 15]


def test_matches_numpy_ravel_multi_index_over_full_grid():
    shape = (4, 6)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    rows = rr.ravel()
    cols = cc.ravel()
    result = np.asarray(sub2ind(shape, rows, cols))
    expected = np.ravel_multi_index((rows, cols), shape)
    assert np.array_equal(result, expected)


def test_full_grid_gives_consecutive_indices():
    shape = (3, 5)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = np.asarray(sub2ind(shape, rr.ravel(), cc.ravel()))
    assert sorted(result.tolist()) == list(range(15))


def test_indices_address_flattened_array_correctly():
    data = np.arange(20).reshape(4, 5) * 3
    rows = np.array([0, 1, 3, 2])
    cols = np.array([4, 2, 0, 3])
    idx = np.asarray(sub2ind(data.shape, rows, cols))
    assert np.array_equal(data.ravel()[idx], data[rows, cols])


def test_single_row_shape():
    cols = np.arange(5)
    rows = np.zeros(5, dtype=int)
    result = sub2ind((1, 5), rows, cols)
    assert _as_list(result) == [0, 1, 2, 3, 4]


def test_single_column_shape():
    rows = np.arange(5)
    cols = np.zeros(5, dtype=int)
    result = sub2ind((5, 1), rows, cols)
    assert _as_list(result) == [0, 1, 2, 3, 4]


def test_square_shape_is_not_symmetric():
    a = _as_list(sub2ind((3, 3), np.array([0]), np.array([2])))
    b = _as_list(sub2ind((3, 3), np.array([2]), np.array([0])))
    assert a == [2]
    assert b == [6]


def test_output_length_matches_input_length():
    rows = np.array([0, 1, 2, 1, 0, 2])
    cols = np.array([1, 1, 1, 0, 0, 0])
    result = np.asarray(sub2ind((3, 3), rows, cols))
    assert result.shape == rows.shape


def test_preserves_two_dimensional_input_shape():
    rows = np.array([[0, 1], [2, 0]])
    cols = np.array([[1, 2], [3, 0]])
    result = np.asarray(sub2ind((3, 4), rows, cols))
    assert result.shape == (2, 2)
    assert result.tolist() == [[1, 6], [11, 0]]


def test_accepts_list_shape():
    result = sub2ind([3, 4], np.array([1]), np.array([2]))
    assert _as_list(result) == [6]


def test_repeated_subscripts_give_repeated_indices():
    rows = np.array([1, 1, 1])
    cols = np.array([2, 2, 2])
    result = sub2ind((3, 4), rows, cols)
    assert _as_list(result) == [6, 6, 6]


def test_empty_input_returns_empty():
    rows = np.array([], dtype=int)
    cols = np.array([], dtype=int)
    result = np.asarray(sub2ind((3, 4), rows, cols))
    assert result.size == 0


def test_out_of_range_subscript_is_not_a_valid_index():
    # Implementation may mark invalid entries as -1 or raise; either is acceptable,
    # but a valid in-range index must never be returned for an out-of-range subscript.
    rows = np.array([5])
    cols = np.array([0])
    try:
        result = np.asarray(sub2ind((3, 4), rows, cols))
    except (ValueError, IndexError):
        return
    assert result.ravel()[0] < 0 or result.ravel()[0] >= 12 or result.ravel()[0] == -1
