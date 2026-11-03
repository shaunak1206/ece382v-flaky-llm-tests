import pytest
import numpy as np
import random

from pymlfunc import sub2ind, ind2sub


def test_sub2ind_single_dimension():
    random.seed(0)
    np.random.seed(0)
    sizes = (10,)
    subs = (3,)
    idx = sub2ind(sizes, *subs)
    assert isinstance(idx, int)
    assert 0 <= idx < 10
    # Because sizes is 10, indices 0..9 valid
    assert idx == 3


def test_sub2ind_two_dimensions_basic():
    random.seed(1)
    np.random.seed(1)
    sizes = (5, 4)
    subs = (2, 3)
    idx = sub2ind(sizes, *subs)
    # index = row + col*size[0] = 2 + 3*5 = 2 + 15 = 17
    assert idx == 17
    assert isinstance(idx, int)
    assert 0 <= idx < 20


def test_sub2ind_three_dimensions_basic():
    random.seed(2)
    np.random.seed(2)
    sizes = (3, 4, 5)
    subs = (1, 2, 3)
    idx = sub2ind(sizes, *subs)
    # Manual calculation:
    # idx = x + y*size_x + z*size_x*size_y
    # = 1 + 2*3 + 3*3*4 = 1 + 6 + 36 = 43
    assert idx == 43
    assert 0 <= idx < 3*4*5


def test_sub2ind_indices_out_of_bounds():
    random.seed(3)
    np.random.seed(3)
    sizes = (3, 3)
    subs = (3, 0)  # index 3 out of bounds for dim 0 (0-based)
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)
    subs = (0, 3)
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)
    subs = (-1, 0)
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)


def test_sub2ind_dimension_mismatch_more_subs():
    random.seed(4)
    np.random.seed(4)
    sizes = (4, 5)
    subs = (1, 2, 3)
    with pytest.raises(ValueError):
        sub2ind(sizes, *subs)


def test_sub2ind_dimension_mismatch_less_subs():
    random.seed(5)
    np.random.seed(5)
    sizes = (4, 5, 6)
    subs = (3, 2)
    with pytest.raises(ValueError):
        sub2ind(sizes, *subs)


def test_sub2ind_zero_dimensions():
    random.seed(6)
    np.random.seed(6)
    # sizes empty tuple, zero dimensions
    sizes = ()
    with pytest.raises(ValueError):
        sub2ind(sizes)


def test_sub2ind_negative_indices():
    random.seed(7)
    np.random.seed(7)
    sizes = (5, 5)
    subs = (-1, 2)
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)
    subs = (3, -2)
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)


def test_sub2ind_large_dimensions():
    random.seed(8)
    np.random.seed(8)
    sizes = (100, 100, 100)
    subs = (50, 50, 50)
    idx = sub2ind(sizes, *subs)
    expected = 50 + 50*100 + 50*100*100
    assert idx == expected
    assert 0 <= idx < 1000000


def test_sub2ind_with_numpy_ints():
    random.seed(9)
    np.random.seed(9)
    sizes = (10, 10)
    subs = (np.int64(3), np.int32(4))
    idx = sub2ind(sizes, *subs)
    expected = 3 + 4*10
    assert idx == expected


def test_sub2ind_with_numpy_arrays_and_broadcast(monkeypatch):
    random.seed(10)
    np.random.seed(10)
    sizes = (3, 3)
    # Passing numpy arrays instead of ints should raise (usually expect ints)
    with pytest.raises(TypeError):
        sub2ind(sizes, np.array([1]), np.array([0]))


def test_sub2ind_consistency_with_ind2sub():
    random.seed(11)
    np.random.seed(11)
    sizes = (7, 6, 5)
    subs = (3, 4, 2)
    idx = sub2ind(sizes, *subs)
    subs_back = ind2sub(sizes, idx)
    assert subs_back == subs


def test_sub2ind_consistency_with_ind2sub_randomized():
    random.seed(12)
    np.random.seed(12)
    sizes = (10, 10, 10)
    for a in range(0, 10, 3):
        for b in range(0,10, 4):
            for c in range(0,10, 5):
                subs = (a, b, c)
                idx = sub2ind(sizes, *subs)
                subs_back = ind2sub(sizes, idx)
                assert subs_back == subs


def test_sub2ind_edge_last_element():
    random.seed(13)
    np.random.seed(13)
    sizes = (5, 5)
    subs = (4, 4)  # last valid indices (0-based)
    idx = sub2ind(sizes, *subs)
    # last index = size[0]*size[1] - 1 = 25 - 1 = 24
    assert idx == 24


def test_sub2ind_edge_first_element():
    random.seed(14)
    np.random.seed(14)
    sizes = (8, 9)
    subs = (0, 0)
    idx = sub2ind(sizes, *subs)
    assert idx == 0


def test_sub2ind_with_one_as_size():
    random.seed(15)
    np.random.seed(15)
    sizes = (1, 10)
    subs = (0, 5)
    idx = sub2ind(sizes, *subs)
    # idx = 0 + 5*1 = 5
    assert idx == 5


def test_sub2ind_with_sizes_containing_one_elsewhere():
    random.seed(16)
    np.random.seed(16)
    sizes = (7, 1, 9)
    subs = (3, 0, 4)
    idx = sub2ind(sizes, *subs)
    # idx = 3 + 0*7 + 4*7*1 = 3 + 0 + 28 = 31
    assert idx == 31


def test_sub2ind_large_index_out_of_bounds():
    random.seed(17)
    np.random.seed(17)
    sizes = (2, 2, 2)
    subs = (1, 1, 2)  # last index 2 is out of bounds for dim=2
    with pytest.raises(IndexError):
        sub2ind(sizes, *subs)


def test_sub2ind_non_integer_indices():
    random.seed(18)
    np.random.seed(18)
    sizes = (4, 4)
    subs = (1.5, 2)
    with pytest.raises(TypeError):
        sub2ind(sizes, *subs)


def test_sub2ind_non_integer_sizes():
    random.seed(19)
    np.random.seed(19)
    sizes = (4.0, 5.0)
    subs = (2, 3)
    with pytest.raises(TypeError):
        sub2ind(sizes, *subs)


def test_sub2ind_empty_sizes():
    random.seed(20)
    np.random.seed(20)
    sizes = ()
    subs = ()
    with pytest.raises(ValueError):
        sub2ind(sizes, *subs)


def test_sub2ind_unicode_indices(monkeypatch):
    random.seed(21)
    np.random.seed(21)
    sizes = (3, 3)
    subs = ('1', 2)
    with pytest.raises(TypeError):
        sub2ind(sizes, *subs)


def test_sub2ind_boolean_indices():
    random.seed(22)
    np.random.seed(22)
    sizes = (3, 3)
    subs = (True, False)  # bool is subclass of int, True=1, False=0
    idx = sub2ind(sizes, *subs)
    # idx = 1 + 0*3 = 1
    assert idx == 1
