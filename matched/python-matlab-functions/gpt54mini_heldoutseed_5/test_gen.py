import inspect
import numpy as np
import pytest

from pymlfunc import sub2ind, ind2sub


def _call_sub2ind(shape, *subs):
    """
    Try a few common calling conventions for sub2ind.
    Returns the result if any convention works.
    """
    candidates = [
        lambda: sub2ind(shape, *subs),
        lambda: sub2ind(shape, subs),
        lambda: sub2ind(shape, np.array(subs)),
        lambda: sub2ind(np.array(shape), *subs),
        lambda: sub2ind(np.array(shape), subs),
        lambda: sub2ind(*((shape,) + subs)),
    ]
    last_exc = None
    for cand in candidates:
        try:
            return cand()
        except Exception as exc:  # pragma: no cover - helper only
            last_exc = exc
    raise last_exc


def _call_ind2sub(shape, ind):
    """
    Try a few common calling conventions for ind2sub.
    """
    candidates = [
        lambda: ind2sub(shape, ind),
        lambda: ind2sub(shape, np.array(ind)),
        lambda: ind2sub(np.array(shape), ind),
        lambda: ind2sub(np.array(shape), np.array(ind)),
    ]
    last_exc = None
    for cand in candidates:
        try:
            return cand()
        except Exception as exc:  # pragma: no cover - helper only
            last_exc = exc
    raise last_exc


def _to_python_scalar(x):
    if isinstance(x, np.ndarray):
        assert x.size == 1
        return x.item()
    if isinstance(x, (list, tuple)):
        assert len(x) == 1
        return x[0]
    return x


def _as_flat_list(x):
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def test_sub2ind_is_exported_from_package():
    import pymlfunc

    assert hasattr(pymlfunc, "sub2ind")
    assert pymlfunc.sub2ind is sub2ind


def test_sub2ind_single_index_first_element_2d():
    result = _call_sub2ind((3, 4), 1, 1)
    assert _to_python_scalar(result) == 1


def test_sub2ind_single_index_second_row_first_col_2d():
    result = _call_sub2ind((3, 4), 2, 1)
    assert _to_python_scalar(result) == 2


def test_sub2ind_single_index_first_row_second_col_2d():
    result = _call_sub2ind((3, 4), 1, 2)
    assert _to_python_scalar(result) == 4


def test_sub2ind_single_index_last_element_2d():
    result = _call_sub2ind((3, 4), 3, 4)
    assert _to_python_scalar(result) == 12


def test_sub2ind_3d_first_element():
    result = _call_sub2ind((2, 3, 4), 1, 1, 1)
    assert _to_python_scalar(result) == 1


def test_sub2ind_3d_mixed_coordinates():
    # MATLAB / column-major linear indexing:
    # idx = i + (j-1)*dim1 + (k-1)*dim1*dim2
    result = _call_sub2ind((2, 3, 4), 2, 1, 1)
    assert _to_python_scalar(result) == 2

    result = _call_sub2ind((2, 3, 4), 1, 2, 1)
    assert _to_python_scalar(result) == 3

    result = _call_sub2ind((2, 3, 4), 1, 1, 2)
    assert _to_python_scalar(result) == 7


def test_sub2ind_vectorized_multiple_coordinates_return_shape_matches_input():
    result = _call_sub2ind((3, 4), [1, 2, 3], [1, 2, 4])
    flat = _as_flat_list(result)
    assert flat == [1, 5, 12]


def test_sub2ind_vectorized_numpy_arrays():
    rows = np.array([1, 2, 3])
    cols = np.array([2, 3, 4])
    result = _call_sub2ind((3, 4), rows, cols)
    assert _as_flat_list(result) == [4, 8, 12]


def test_sub2ind_accepts_python_tuples_for_subscripts():
    result = _call_sub2ind((3, 4), (1, 2, 3), (1, 3, 4))
    assert _as_flat_list(result) == [1, 7, 12]


def test_sub2ind_roundtrip_with_ind2sub_scalar():
    shape = (4, 5)
    subs = (3, 4)
    ind = _call_sub2ind(shape, *subs)
    back = _call_ind2sub(shape, ind)

    back_list = _as_flat_list(back)
    assert back_list == list(subs)


def test_sub2ind_roundtrip_with_ind2sub_vectorized():
    shape = (3, 4)
    rows = np.array([1, 2, 3])
    cols = np.array([1, 2, 4])
    inds = _call_sub2ind(shape, rows, cols)
    back = _call_ind2sub(shape, inds)

    # ind2sub may return a tuple of arrays/lists; normalize and compare.
    assert isinstance(back, (tuple, list))
    assert len(back) == 2
    back_rows = _as_flat_list(back[0])
    back_cols = _as_flat_list(back[1])
    assert back_rows == [1, 2, 3]
    assert back_cols == [1, 2, 4]


def test_sub2ind_with_singleton_dimension():
    # A shape with one dimension of length 1 should still work.
    result = _call_sub2ind((1, 5), 1, 5)
    assert _to_python_scalar(result) == 5


def test_sub2ind_rejects_out_of_bounds_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 4, 1)

    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 5)


def test_sub2ind_rejects_zero_or_negative_subscripts():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 0, 1)

    with pytest.raises(Exception):
        _call_sub2ind((3, 4), -1, 1)


def test_sub2ind_rejects_mismatched_subscript_lengths():
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), [1, 2], [1, 2, 3])


def test_sub2ind_rejects_wrong_number_of_dimensions():
    # If the implementation expects exactly len(shape) subscript arrays/scalars,
    # this should fail.
    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1)

    with pytest.raises(Exception):
        _call_sub2ind((3, 4), 1, 2, 3)


def test_sub2ind_function_signature_exists():
    # This is a light sanity check that the exported function is callable.
    assert callable(sub2ind)
    assert callable(ind2sub)
    assert inspect.isfunction(sub2ind) or inspect.ismethod(sub2ind) or callable(sub2ind)
