import pytest
import random
import numpy as np

from pymlfunc import sub2ind, ind2sub


@pytest.mark.parametrize(
    "shape, subs, expected",
    [
        # 2D examples
        ((3, 4), (0, 0), 0),
        ((3, 4), (1, 0), 1),
        ((3, 4), (0, 1), 3),
        ((3, 4), (2, 3), 11),
        # 3D example
        ((2, 3, 4), (1, 2, 3), 1 + 2*2 + 3*6),  # 1 + 4 + 18 = 23
    ],
)
def test_sub2ind_basic(shape, subs, expected):
    # Seed numpy's generator here in case sub2ind uses numpy.random (assuming)
    np.random.seed(0)
    random.seed(0)
    idx = sub2ind(shape, *subs)
    assert idx == expected


def test_sub2ind_ind2sub_roundtrip():
    np.random.seed(0)
    random.seed(0)
    shapes = [
        (5,),
        (4, 3),
        (2, 3, 4),
        (2, 2, 2, 2),
    ]
    for shape in shapes:
        size = np.prod(shape)
        for idx in [0, size - 1, size // 2]:
            subs = ind2sub(shape, idx)
            idx2 = sub2ind(shape, *subs)
            assert idx == idx2


def test_sub2ind_invalid_index_raises():
    np.random.seed(0)
    random.seed(0)
    shape = (3, 3)
    # Subscript less than zero
    with pytest.raises(IndexError):
        sub2ind(shape, -1, 0)
    # Subscript equal to dimension size
    with pytest.raises(IndexError):
        sub2ind(shape, 3, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 3)
    # Too few subscripts
    with pytest.raises(TypeError):
        sub2ind(shape, 1)
    # Too many subscripts
    with pytest.raises(TypeError):
        sub2ind(shape, 0, 1, 2)


def test_sub2ind_out_of_bounds_negative():
    np.random.seed(0)
    random.seed(0)
    shape = (4, 4)
    with pytest.raises(IndexError):
        sub2ind(shape, -1, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, -1)


def test_sub2ind_out_of_bounds_too_large():
    np.random.seed(0)
    random.seed(0)
    shape = (4, 5)
    with pytest.raises(IndexError):
        sub2ind(shape, 4, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 5)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_sub2ind_all_zeros(shape):
    np.random.seed(0)
    random.seed(0)
    subs = tuple(0 for _ in shape)
    idx = sub2ind(shape, *subs)
    assert idx == 0


def test_sub2ind_large_shape():
    np.random.seed(0)
    random.seed(0)
    shape = (1000, 2000, 3)
    subs = (999, 1999, 2)
    idx = sub2ind(shape, *subs)
    # Check idx is within range
    assert 0 <= idx < np.prod(shape)
    # Inverse check
    subs2 = ind2sub(shape, idx)
    assert subs == subs2


def test_sub2ind_with_zeros_in_shape():
    np.random.seed(0)
    random.seed(0)
    shape = (0,)
    with pytest.raises(IndexError):
        sub2ind(shape, 0)
    shape = (3, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 0)


def test_sub2ind_dtype_and_return_type():
    np.random.seed(0)
    random.seed(0)
    shape = (10, 10)
    idx = sub2ind(shape, 5, 7)
    assert isinstance(idx, int)
    # Confirm idx with known formula
    assert idx == 5 + 7 * shape[0]


def test_sub2ind_random_large_inputs():
    # Even if sub2ind uses random, seed here
    np.random.seed(12345)
    random.seed(54321)

    shape = (100, 50, 20)
    # Random valid subscripts
    subs = (
        np.random.randint(0, shape[0]),
        np.random.randint(0, shape[1]),
        np.random.randint(0, shape[2]),
    )
    idx = sub2ind(shape, *subs)
    # Check idx is within range
    assert 0 <= idx < np.prod(shape)
    # Check roundtrip
    assert ind2sub(shape, idx) == subs


def test_sub2ind_float_subscript_raises():
    np.random.seed(0)
    random.seed(0)
    shape = (3, 3)
    with pytest.raises(TypeError):
        sub2ind(shape, 0.5, 1)
    with pytest.raises(TypeError):
        sub2ind(shape, 1, 2.5)


def test_sub2ind_tuple_subscripts():
    np.random.seed(0)
    random.seed(0)
    shape = (3, 4)
    subs = (2, 3)
    idx1 = sub2ind(shape, *subs)
    idx2 = sub2ind(shape, 2, 3)
    assert idx1 == idx2


def test_sub2ind_with_single_dimension():
    np.random.seed(0)
    random.seed(0)
    shape = (10,)
    for i in range(shape[0]):
        idx = sub2ind(shape, i)
        assert idx == i


def test_sub2ind_zero_dim_raises():
    np.random.seed(0)
    random.seed(0)
    with pytest.raises(TypeError):
        sub2ind((),)


def test_sub2ind_non_integer_shape_raises():
    np.random.seed(0)
    random.seed(0)
    with pytest.raises(TypeError):
        sub2ind((3.5, 2), 1, 1)

    with pytest.raises(TypeError):
        sub2ind(("3", 2), 1, 1)


def test_ind2sub_sub2ind_consistency_multiple_shapes():
    np.random.seed(0)
    random.seed(0)
    shapes = [
        (2, 3),
        (2, 2, 2),
        (4, 5, 6),
    ]
    for shape in shapes:
        max_idx = np.prod(shape) - 1
        for idx in [0, max_idx // 3, max_idx // 2, max_idx]:
            subs = ind2sub(shape, idx)
            idx2 = sub2ind(shape, *subs)
            assert idx == idx2


def test_sub2ind_negative_dimension_raises():
    np.random.seed(0)
    random.seed(0)
    with pytest.raises(IndexError):
        sub2ind((3, -2), 1, 0)


def test_sub2ind_non_tuple_shape_raises():
    np.random.seed(0)
    random.seed(0)
    with pytest.raises(TypeError):
        sub2ind("123", 1, 2)

    with pytest.raises(TypeError):
        sub2ind(123, 1)


def test_sub2ind_index_error_message():
    np.random.seed(0)
    random.seed(0)
    shape = (3, 3)
    with pytest.raises(IndexError) as excinfo:
        sub2ind(shape, 3, 0)
    assert "index" in str(excinfo.value).lower()


def test_sub2ind_large_indices_raises():
    np.random.seed(0)
    random.seed(0)
    shape = (1, 1)
    with pytest.raises(IndexError):
        sub2ind(shape, 1, 0)
    with pytest.raises(IndexError):
        sub2ind(shape, 0, 1)
