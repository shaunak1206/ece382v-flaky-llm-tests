import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _call_sub2ind(shape, *subs):
    """
    Try a few common calling conventions so the tests are resilient to the
    exact function signature used by the project.
    """
    candidates = [
        lambda: sub2ind(shape, *subs),
        lambda: sub2ind(shape, subs),
        lambda: sub2ind(*((shape,) + subs)),
    ]
    last_exc = None
    for candidate in candidates:
        try:
            return candidate()
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def _as_int(x):
    if isinstance(x, np.ndarray):
        assert x.size == 1
        return int(x.ravel()[0])
    return int(x)


def test_sub2ind_2d_first_element_is_one():
    assert _as_int(_call_sub2ind((3, 4), 1, 1)) == 1


def test_sub2ind_2d_second_row_first_column():
    assert _as_int(_call_sub2ind((3, 4), 2, 1)) == 2


def test_sub2ind_2d_first_row_second_column_is_four():
    assert _as_int(_call_sub2ind((3, 4), 1, 2)) == 4


def test_sub2ind_2d_last_element_is_product_of_shape():
    assert _as_int(_call_sub2ind((3, 4), 3, 4)) == 12


def test_sub2ind_3d_matches_matlab_column_major_order():
    # MATLAB-style column-major indexing:
    # ind = row + (col-1)*nrows + (page-1)*nrows*ncols
    assert _as_int(_call_sub2ind((3, 4, 2), 2, 3, 2)) == 20


def test_sub2ind_accepts_list_shape():
    assert _as_int(_call_sub2ind([3, 4], 3, 4)) == 12


def test_sub2ind_accepts_numpy_shape_array():
    assert _as_int(_call_sub2ind(np.array([3, 4]), 2, 2)) == 5


def test_sub2ind_vectorized_inputs_return_expected_indices():
    result = _call_sub2ind((3, 4), np.array([1, 2, 3]), np.array([1, 2, 4]))
    np.testing.assert_array_equal(np.asarray(result), np.array([1, 5, 12]))


def test_sub2ind_vectorized_inputs_with_python_lists():
    result = _call_sub2ind((3, 4), [1, 2, 3], [2, 3, 4])
    np.testing.assert_array_equal(np.asarray(result), np.array([4, 8, 12]))


def test_sub2ind_supports_three_dimensions_vectorized():
    result = _call_sub2ind(
        (3, 4, 2),
        np.array([1, 2]),
        np.array([1, 4]),
        np.array([1, 2]),
    )
    np.testing.assert_array_equal(np.asarray(result), np.array([1, 24]))


def test_sub2ind_round_trip_with_ind2sub_scalar():
    idx = _call_sub2ind((5, 6), 4, 3)
    subs = ind2sub((5, 6), int(idx))
    # Accept either a tuple/list or numpy array from ind2sub.
    assert tuple(np.asarray(subs).tolist()) == (4, 3)


def test_sub2ind_round_trip_with_ind2sub_vectorized():
    rows = np.array([1, 5, 3])
    cols = np.array([1, 2, 6])
    idx = _call_sub2ind((5, 6), rows, cols)
    subs = ind2sub((5, 6), np.asarray(idx))
    subs = np.asarray(subs)
    assert subs.shape[0] == 2
    np.testing.assert_array_equal(subs[0], rows)
    np.testing.assert_array_equal(subs[1], cols)


def test_sub2ind_rejects_zero_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 0, 1)


def test_sub2ind_rejects_negative_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), -1, 1)


def test_sub2ind_rejects_out_of_bounds_index():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 4, 1)


def test_sub2ind_rejects_mismatched_vector_lengths():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), [1, 2], [1])


def test_sub2ind_rejects_wrong_number_of_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1)


def test_sub2ind_handles_singleton_dimension():
    assert _as_int(_call_sub2ind((1, 5), 1, 5)) == 5
