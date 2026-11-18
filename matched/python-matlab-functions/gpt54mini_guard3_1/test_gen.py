import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _call_sub2ind(shape, *subs):
    """
    Try common calling conventions without assuming the exact signature.
    """
    candidates = [
        (shape, *subs),
        (shape, subs),
    ]
    last_exc = None
    for args in candidates:
        try:
            return sub2ind(*args)
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def _call_ind2sub(shape, ind):
    """
    Try common calling conventions without assuming the exact signature.
    """
    candidates = [
        (shape, ind),
        (shape, [ind]),
    ]
    last_exc = None
    for args in candidates:
        try:
            return ind2sub(*args)
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def _as_tuple_of_arrays(result):
    """
    Normalize ind2sub outputs to a tuple of numpy arrays.
    """
    if isinstance(result, tuple):
        return tuple(np.asarray(x) for x in result)
    return (np.asarray(result),)


def test_sub2ind_scalar_roundtrip_2d():
    shape = (3, 4)
    row = 1
    col = 2
    idx = _call_sub2ind(shape, row, col)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    assert len(back) == 2
    assert back[0].item() == row
    assert back[1].item() == col


def test_sub2ind_scalar_roundtrip_3d():
    shape = (2, 3, 4)
    i, j, k = 1, 2, 3
    idx = _call_sub2ind(shape, i, j, k)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    assert len(back) == 3
    assert back[0].item() == i
    assert back[1].item() == j
    assert back[2].item() == k


def test_sub2ind_vector_roundtrip_2d():
    shape = (3, 4)
    rows = np.array([0, 1, 2], dtype=int)
    cols = np.array([0, 2, 3], dtype=int)
    idx = _call_sub2ind(shape, rows, cols)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    assert len(back) == 2
    np.testing.assert_array_equal(back[0], rows)
    np.testing.assert_array_equal(back[1], cols)


def test_sub2ind_vector_roundtrip_3d():
    shape = (2, 3, 4)
    a = np.array([0, 1, 1], dtype=int)
    b = np.array([0, 2, 1], dtype=int)
    c = np.array([0, 1, 3], dtype=int)
    idx = _call_sub2ind(shape, a, b, c)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    assert len(back) == 3
    np.testing.assert_array_equal(back[0], a)
    np.testing.assert_array_equal(back[1], b)
    np.testing.assert_array_equal(back[2], c)


def test_sub2ind_preserves_input_shape_for_1d_arrays():
    shape = (4, 5)
    rows = np.array([0, 1, 2, 3], dtype=int)
    cols = np.array([1, 1, 2, 3], dtype=int)
    idx = np.asarray(_call_sub2ind(shape, rows, cols))
    assert idx.shape == rows.shape


def test_sub2ind_supports_python_lists():
    shape = (3, 4)
    rows = [0, 1, 2]
    cols = [3, 2, 1]
    idx = _call_sub2ind(shape, rows, cols)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    np.testing.assert_array_equal(back[0], np.asarray(rows))
    np.testing.assert_array_equal(back[1], np.asarray(cols))


def test_sub2ind_repeated_coordinates_give_repeated_indices():
    shape = (5, 6)
    rows = np.array([2, 2, 2, 2], dtype=int)
    cols = np.array([4, 4, 4, 4], dtype=int)
    idx = np.asarray(_call_sub2ind(shape, rows, cols))
    assert idx.size == 4
    assert np.all(idx == idx[0])


def test_sub2ind_handles_scalar_and_array_inputs():
    shape = (4, 5)
    rows = np.array([0, 1, 2], dtype=int)
    cols = 3
    idx = _call_sub2ind(shape, rows, cols)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    np.testing.assert_array_equal(back[0], rows)
    np.testing.assert_array_equal(back[1], np.full_like(rows, cols))


def test_sub2ind_roundtrip_on_full_coordinate_grid_2d():
    shape = (2, 3)
    rows = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    cols = np.array([0, 1, 2, 0, 1, 2], dtype=int)
    idx = _call_sub2ind(shape, rows, cols)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    np.testing.assert_array_equal(back[0], rows)
    np.testing.assert_array_equal(back[1], cols)


def test_sub2ind_roundtrip_on_full_coordinate_grid_3d():
    shape = (2, 2, 2)
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    b = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    c = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    idx = _call_sub2ind(shape, a, b, c)
    back = _as_tuple_of_arrays(_call_ind2sub(shape, idx))
    np.testing.assert_array_equal(back[0], a)
    np.testing.assert_array_equal(back[1], b)
    np.testing.assert_array_equal(back[2], c)


def test_sub2ind_empty_inputs_return_empty_result():
    shape = (3, 4)
    rows = np.array([], dtype=int)
    cols = np.array([], dtype=int)
    idx = np.asarray(_call_sub2ind(shape, rows, cols))
    assert idx.shape == (0,)


def test_sub2ind_out_of_bounds_index_raises():
    shape = (3, 4)
    rows = np.array([0, 3], dtype=int)  # 3 is out of bounds for axis 0
    cols = np.array([0, 1], dtype=int)
    with pytest.raises(Exception):
        _call_sub2ind(shape, rows, cols)
