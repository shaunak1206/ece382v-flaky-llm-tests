import pytest

from pymlfunc import ind2sub, sub2ind


def _tolist(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def _call_sub2ind(shape, *subs):
    """
    Call sub2ind with a few common signatures to keep the tests robust
    against small API differences.
    """
    try:
        return sub2ind(shape, *subs)
    except TypeError:
        pass

    try:
        return sub2ind(shape, subs)
    except TypeError:
        pass

    return sub2ind(*((shape,) + subs))


def test_sub2ind_single_first_element_2d():
    assert _tolist(_call_sub2ind((3, 4), 1, 1)) == 1


def test_sub2ind_single_middle_element_2d():
    assert _tolist(_call_sub2ind((3, 4), 2, 2)) == 5


def test_sub2ind_single_last_element_2d():
    assert _tolist(_call_sub2ind((3, 4), 3, 4)) == 12


def test_sub2ind_column_major_order_2d():
    # MATLAB/Fortran order: first dimension varies fastest.
    result = _tolist(_call_sub2ind((3, 4), [1, 2, 3], [1, 1, 1]))
    assert result == [1, 2, 3]


def test_sub2ind_moves_across_columns_after_rows():
    result = _tolist(_call_sub2ind((3, 4), [1, 2, 3], [1, 2, 2]))
    assert result == [1, 5, 6]


def test_sub2ind_accepts_shape_as_list():
    assert _tolist(_call_sub2ind([3, 4], 3, 1)) == 3


def test_sub2ind_one_dimensional_shape():
    result = _tolist(_call_sub2ind((5,), [1, 2, 3, 4, 5]))
    assert result == [1, 2, 3, 4, 5]


def test_sub2ind_vectorized_3d_matches_expected_indices():
    result = _tolist(_call_sub2ind((2, 3, 4), [1, 2, 1], [1, 2, 3], [1, 1, 4]))
    assert result == [1, 8, 22]


def test_sub2ind_roundtrip_with_ind2sub_3d():
    shape = (4, 5, 6)
    rows = [1, 4, 2]
    cols = [1, 5, 3]
    pages = [1, 2, 6]

    lin = _call_sub2ind(shape, rows, cols, pages)
    back = ind2sub(shape, lin)

    # ind2sub may return a tuple of arrays/lists, so normalize each piece.
    back = tuple(_tolist(part) for part in back)
    assert back == (rows, cols, pages)


def test_sub2ind_raises_for_zero_index():
    with pytest.raises((IndexError, ValueError)):
        _call_sub2ind((3, 4), 0, 1)


def test_sub2ind_raises_for_out_of_bounds_index():
    with pytest.raises((IndexError, ValueError)):
        _call_sub2ind((3, 4), 4, 1)


def test_sub2ind_raises_for_dimension_mismatch():
    with pytest.raises((TypeError, ValueError, IndexError)):
        _call_sub2ind((3, 4), 1)


def test_sub2ind_roundtrip_single_scalar_index():
    shape = (3, 4)
    lin = _call_sub2ind(shape, 2, 3)
    back = ind2sub(shape, lin)
    back = tuple(_tolist(part) for part in back)
    assert back == ([2], [3])
