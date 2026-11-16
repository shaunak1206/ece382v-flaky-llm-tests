import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _grid(nrows, ncols):
    r, c = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    return r.astype(int), c.astype(int)


def _indices(nrows, ncols):
    r, c = _grid(nrows, ncols)
    return np.asarray(sub2ind((nrows, ncols), r.copy(), c.copy()))


def test_output_shape_matches_input_shape_1d():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    out = np.asarray(sub2ind((3, 4), rows, cols))
    assert out.shape == rows.shape


def test_output_shape_matches_input_shape_2d():
    r, c = _grid(3, 4)
    out = np.asarray(sub2ind((3, 4), r.copy(), c.copy()))
    assert out.shape == (3, 4)


def test_output_is_integer_typed():
    out = _indices(3, 4)
    assert np.issubdtype(out.dtype, np.integer)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 7), (5, 5), (1, 6), (6, 1)])
def test_mapping_is_bijection_onto_contiguous_range(shape):
    out = _indices(*shape).ravel()
    n = shape[0] * shape[1]
    assert len(set(out.tolist())) == n
    # either zero-based or one-based contiguous range
    assert sorted(out.tolist()) in (list(range(n)), list(range(1, n + 1)))


def test_origin_maps_to_smallest_index():
    out = _indices(3, 4)
    assert out[0, 0] == out.min()


def test_last_element_maps_to_largest_index():
    out = _indices(3, 4)
    assert out[-1, -1] == out.max()


def test_neighbour_steps_are_consistent_with_row_or_column_major():
    nrows, ncols = 3, 4
    out = _indices(nrows, ncols)
    col_step = np.unique(np.diff(out, axis=1))
    row_step = np.unique(np.diff(out, axis=0))
    assert len(col_step) == 1 and len(row_step) == 1
    # row-major: col step 1, row step ncols; column-major: row step 1, col step nrows
    assert (col_step[0], row_step[0]) in [(1, ncols), (nrows, 1)]


def test_single_row_array_indices_are_consecutive():
    out = _indices(1, 5).ravel()
    assert np.all(np.diff(out) == 1)


def test_single_column_array_indices_are_consecutive():
    out = _indices(5, 1).ravel()
    assert np.all(np.diff(out) == 1)


def test_known_value_in_allowed_conventions():
    # (row=1, col=2) in a 3x4 array:
    # row-major 0-based -> 6, column-major 0-based -> 7,
    # 1-based variants -> 7 / 8
    out = np.asarray(sub2ind((3, 4), np.array([1]), np.array([2])))
    assert out.size == 1
    assert int(out[0]) in (6, 7, 8)


def test_inputs_not_modified():
    rows = np.array([0, 1, 2, 2])
    cols = np.array([0, 3, 1, 2])
    rows_copy, cols_copy = rows.copy(), cols.copy()
    sub2ind((3, 4), rows, cols)
    assert np.array_equal(rows, rows_copy)
    assert np.array_equal(cols, cols_copy)


def test_same_subscripts_give_same_index():
    rows = np.array([1, 1, 2])
    cols = np.array([2, 2, 0])
    out = np.asarray(sub2ind((3, 4), rows, cols))
    assert out[0] == out[1]
    assert out[0] != out[2]


def test_square_array_swapped_subscripts_differ_off_diagonal():
    a = np.asarray(sub2ind((4, 4), np.array([1]), np.array([2])))
    b = np.asarray(sub2ind((4, 4), np.array([2]), np.array([1])))
    assert a[0] != b[0]


def test_diagonal_elements_distinct_and_increasing():
    d = np.arange(5)
    out = np.asarray(sub2ind((5, 5), d.copy(), d.copy()))
    assert np.all(np.diff(out) > 0)


@pytest.mark.parametrize("shape", [(3, 4), (5, 2), (4, 4)])
def test_roundtrip_with_ind2sub(shape):
    r, c = _grid(*shape)
    r, c = r.ravel(), c.ravel()
    ind = sub2ind(shape, r.copy(), c.copy())
    r2, c2 = ind2sub(shape, np.asarray(ind).copy())
    assert np.array_equal(np.asarray(r2), r)
    assert np.array_equal(np.asarray(c2), c)
