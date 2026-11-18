import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _as_tuple_of_arrays(result):
    if isinstance(result, tuple):
        return tuple(np.asarray(x) for x in result)
    return (np.asarray(result),)


def _roundtrip(shape, *subs):
    ind = sub2ind(shape, *subs)
    got = ind2sub(shape, ind)
    return ind, got


def test_sub2ind_2d_scalar_roundtrip_origin():
    shape = (3, 4)
    ind, got = _roundtrip(shape, 0, 0)
    assert np.isscalar(ind) or np.asarray(ind).shape == ()
    rows, cols = _as_tuple_of_arrays(got)
    assert rows.item() == 0
    assert cols.item() == 0


def test_sub2ind_2d_scalar_roundtrip_nonzero():
    shape = (3, 4)
    ind, got = _roundtrip(shape, 1, 2)
    rows, cols = _as_tuple_of_arrays(got)
    assert rows.item() == 1
    assert cols.item() == 2


def test_sub2ind_2d_last_element_roundtrip():
    shape = (3, 4)
    ind, got = _roundtrip(shape, 2, 3)
    rows, cols = _as_tuple_of_arrays(got)
    assert rows.item() == 2
    assert cols.item() == 3


def test_sub2ind_3d_scalar_roundtrip():
    shape = (2, 3, 4)
    ind, got = _roundtrip(shape, 1, 2, 3)
    a, b, c = _as_tuple_of_arrays(got)
    assert a.item() == 1
    assert b.item() == 2
    assert c.item() == 3


def test_sub2ind_2d_vectorized_numpy_arrays_roundtrip():
    shape = (3, 4)
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 3])
    ind = sub2ind(shape, rows, cols)
    got_rows, got_cols = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert np.array_equal(got_rows, rows)
    assert np.array_equal(got_cols, cols)


def test_sub2ind_3d_vectorized_numpy_arrays_roundtrip():
    shape = (2, 3, 4)
    a = np.array([0, 1, 1])
    b = np.array([0, 1, 2])
    c = np.array([0, 2, 3])
    ind = sub2ind(shape, a, b, c)
    got_a, got_b, got_c = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert np.array_equal(got_a, a)
    assert np.array_equal(got_b, b)
    assert np.array_equal(got_c, c)


def test_sub2ind_accepts_python_lists():
    shape = (3, 4)
    ind = sub2ind(shape, [0, 1], [2, 3])
    got_rows, got_cols = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert np.array_equal(got_rows, np.array([0, 1]))
    assert np.array_equal(got_cols, np.array([2, 3]))


def test_ind2sub_returns_same_number_of_dimensions_as_shape():
    shape = (2, 3, 4, 5)
    ind = sub2ind(shape, 1, 2, 3, 4)
    got = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert len(got) == len(shape)


def test_sub2ind_roundtrip_on_multiple_points_2d():
    shape = (4, 5)
    rows = np.array([0, 1, 2, 3])
    cols = np.array([4, 3, 2, 1])
    ind = sub2ind(shape, rows, cols)
    rr, cc = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert np.array_equal(rr, rows)
    assert np.array_equal(cc, cols)


def test_sub2ind_roundtrip_on_multiple_points_3d():
    shape = (3, 4, 5)
    x = np.array([0, 1, 2, 0])
    y = np.array([0, 1, 2, 3])
    z = np.array([4, 3, 2, 1])
    ind = sub2ind(shape, x, y, z)
    gx, gy, gz = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert np.array_equal(gx, x)
    assert np.array_equal(gy, y)
    assert np.array_equal(gz, z)


def test_sub2ind_same_input_same_output():
    shape = (3, 4)
    i1 = sub2ind(shape, 1, 2)
    i2 = sub2ind(shape, 1, 2)
    assert np.array_equal(np.asarray(i1), np.asarray(i2))


def test_ind2sub_and_sub2ind_handle_numpy_scalar_input():
    shape = (3, 4)
    ind = sub2ind(shape, np.int64(2), np.int64(1))
    got_rows, got_cols = _as_tuple_of_arrays(ind2sub(shape, ind))
    assert got_rows.item() == 2
    assert got_cols.item() == 1
