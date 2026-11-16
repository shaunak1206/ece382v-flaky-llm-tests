import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _arr(x):
    return np.array(x, dtype=int)


def test_origin_maps_to_zero():
    result = sub2ind((3, 4), _arr([0]), _arr([0]))
    assert np.asarray(result).tolist() == [0]


def test_first_row_indices_equal_columns():
    cols = _arr([0, 1, 2, 3])
    rows = _arr([0, 0, 0, 0])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [0, 1, 2, 3]


def test_row_major_ordering():
    # second row starts at index ncols
    result = sub2ind((3, 4), _arr([1]), _arr([0]))
    assert np.asarray(result).tolist() == [4]


def test_last_element():
    result = sub2ind((3, 4), _arr([2]), _arr([3]))
    assert np.asarray(result).tolist() == [11]


def test_multiple_points():
    rows = _arr([0, 1, 2, 2])
    cols = _arr([1, 2, 0, 3])
    result = sub2ind((3, 4), rows, cols)
    assert np.asarray(result).tolist() == [1, 6, 8, 11]


def test_matches_numpy_ravel_multi_index():
    shape = (5, 7)
    rng = np.random.RandomState(0)
    rows = rng.randint(0, shape[0], size=30)
    cols = rng.randint(0, shape[1], size=30)
    expected = np.ravel_multi_index((rows, cols), shape)
    result = sub2ind(shape, rows.copy(), cols.copy())
    assert np.array_equal(np.asarray(result), expected)


def test_matches_flat_indexing_of_numpy_array():
    shape = (4, 6)
    data = np.arange(shape[0] * shape[1]).reshape(shape) * 3
    rows = _arr([0, 1, 3, 2])
    cols = _arr([5, 0, 2, 4])
    idx = np.asarray(sub2ind(shape, rows, cols))
    assert np.array_equal(data.ravel()[idx], data[rows, cols])


def test_all_positions_cover_every_index_once():
    shape = (3, 5)
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    result = np.asarray(sub2ind(shape, rr.ravel().copy(), cc.ravel().copy()))
    assert sorted(result.tolist()) == list(range(15))


def test_single_column_array():
    shape = (5, 1)
    rows = _arr([0, 1, 2, 3, 4])
    cols = _arr([0, 0, 0, 0, 0])
    result = sub2ind(shape, rows, cols)
    assert np.asarray(result).tolist() == [0, 1, 2, 3, 4]


def test_single_row_array():
    shape = (1, 5)
    rows = _arr([0, 0, 0])
    cols = _arr([4, 0, 2])
    result = sub2ind(shape, rows, cols)
    assert np.asarray(result).tolist() == [4, 0, 2]


def test_square_matrix():
    result = sub2ind((3, 3), _arr([0, 1, 2]), _arr([0, 1, 2]))
    assert np.asarray(result).tolist() == [0, 4, 8]


def test_result_length_matches_input_length():
    rows = _arr([0, 1, 2, 1, 0])
    cols = _arr([0, 1, 2, 3, 3])
    result = sub2ind((3, 4), rows, cols)
    assert len(result) == 5


def test_out_of_range_column_is_flagged_or_raises():
    # Out-of-range subscripts should either raise or be flagged with -1,
    # and must never silently map to a valid in-range index of a wrong cell.
    shape = (3, 4)
    try:
        result = np.asarray(sub2ind(shape, _arr([5]), _arr([0])))
    except Exception:
        return
    assert result.tolist() == [-1]


def test_negative_subscript_is_flagged_or_raises():
    shape = (3, 4)
    try:
        result = np.asarray(sub2ind(shape, _arr([-1]), _arr([0])))
    except Exception:
        return
    assert result.tolist() == [-1]


def test_roundtrip_with_ind2sub():
    shape = (4, 5)
    rows = _arr([0, 1, 2, 3, 3])
    cols = _arr([0, 4, 2, 1, 4])
    ind = sub2ind(shape, rows.copy(), cols.copy())
    out_rows, out_cols = ind2sub(shape, np.asarray(ind))
    assert np.asarray(out_rows).tolist() == rows.tolist()
    assert np.asarray(out_cols).tolist() == cols.tolist()
