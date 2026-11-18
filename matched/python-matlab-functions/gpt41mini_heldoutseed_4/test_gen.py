import pytest
from pymlfunc import sub2ind


def test_sub2ind_basic():
    # 3x3 matrix, single subscripts, column-major order
    idx = sub2ind((3, 3), 1, 1)
    assert idx == 0
    idx = sub2ind((3, 3), 3, 2)
    assert idx == 5


def test_sub2ind_linear_indexing():
    # index corresponds to idx = row + (col-1)*num_rows in MATLAB's 1-based, zero-based here
    shape = (4, 5)
    # (1,1) -> 0
    assert sub2ind(shape, 1, 1) == 0
    # (4,1) -> 3
    assert sub2ind(shape, 4, 1) == 3
    # (1,2) -> 4
    assert sub2ind(shape, 1, 2) == 4
    # (4,5) -> 19
    assert sub2ind(shape, 4, 5) == 19


def test_sub2ind_3d():
    # 3D array shape 2x3x4
    shape = (2, 3, 4)
    # (1,1,1) => 0
    assert sub2ind(shape, 1, 1, 1) == 0
    # (2,3,4) => last element
    expected = (2 - 1) + (3 - 1) * 2 + (4 - 1) * 2 * 3
    assert sub2ind(shape, 2, 3, 4) == expected
    # (1,2,3)
    expected = (1 - 1) + (2 - 1) * 2 + (3 - 1) * 2 * 3
    assert sub2ind(shape, 1, 2, 3) == expected


def test_sub2ind_out_of_bounds_low():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 1)  # zero index not allowed


def test_sub2ind_out_of_bounds_high():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, 4, 1)  # row > shape[0]


def test_sub2ind_wrong_number_of_subs():
    shape = (3, 3)
    # Too few subscripts
    with pytest.raises(ValueError):
        sub2ind(shape, 1)
    # Too many subscripts
    with pytest.raises(ValueError):
        sub2ind(shape, 1, 1, 1)


def test_sub2ind_non_integer_subscripts():
    shape = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(shape, 1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(shape, 1, "2")


def test_sub2ind_empty_shape():
    with pytest.raises(ValueError):
        sub2ind((), 1)


def test_sub2ind_shape_zero_dimension():
    with pytest.raises(ValueError):
        sub2ind((0, 3), 1, 1)


def test_sub2ind_large_indices():
    shape = (1000, 1000)
    idx = sub2ind(shape, 1000, 1000)
    expected = (1000 - 1) + (1000 - 1) * 1000
    assert idx == expected


def test_sub2ind_negative_indices():
    shape = (3, 3)
    with pytest.raises(IndexError):
        sub2ind(shape, -1, 1)
    with pytest.raises(IndexError):
        sub2ind(shape, 1, -1)


def test_sub2ind_tuple_shape():
    shape = (3, 3)
    idx = sub2ind(shape, 2, 3)
    expected = (2 - 1) + (3 - 1) * 3
    assert idx == expected


def test_sub2ind_subscripts_as_lists():
    shape = (3, 3)
    idx = sub2ind(shape, [1], [3])
    expected = (1 - 1) + (3 - 1) * 3
    assert idx == expected


def test_sub2ind_subscripts_as_tuples():
    shape = (3, 3)
    idx = sub2ind(shape, (2,), (2,))
    expected = (2 - 1) + (2 - 1) * 3
    assert idx == expected


def test_sub2ind_shape_as_list():
    shape = [3, 3, 3]
    idx = sub2ind(shape, 3, 2, 1)
    expected = (3 - 1) + (2 - 1) * 3 + (1 - 1) * 3 * 3
    assert idx == expected


def test_sub2ind_1d_array():
    shape = (5,)
    idx = sub2ind(shape, 3)
    assert idx == 2
    idx = sub2ind(shape, 1)
    assert idx == 0
    with pytest.raises(IndexError):
        sub2ind(shape, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 6)


def test_sub2ind_max_dims():
    # Test with 5 dimensions
    shape = (2, 2, 2, 2, 2)
    subs = (2, 1, 2, 1, 2)
    expected = (
        (2 - 1)
        + (1 - 1) * 2
        + (2 - 1) * 2 * 2
        + (1 - 1) * 2 * 2 * 2
        + (2 - 1) * 2 * 2 * 2 * 2
    )
    idx = sub2ind(shape, *subs)
    assert idx == expected
