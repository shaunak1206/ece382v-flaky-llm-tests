import pytest

from pymlfunc import sub2ind, ind2sub


def test_sub2ind_2d_first_element_is_one_based():
    assert sub2ind((3, 4), 1, 1) == 1


def test_sub2ind_2d_column_major_order_across_first_column():
    assert sub2ind((3, 4), 1, 2) == 4
    assert sub2ind((3, 4), 2, 2) == 5
    assert sub2ind((3, 4), 3, 2) == 6


def test_sub2ind_2d_last_element():
    assert sub2ind((3, 4), 3, 4) == 12


def test_sub2ind_2d_middle_element():
    assert sub2ind((5, 7), 4, 6) == 29


def test_sub2ind_3d_first_element_is_one_based():
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1


def test_sub2ind_3d_last_element():
    assert sub2ind((2, 3, 4), 2, 3, 4) == 24


def test_sub2ind_3d_column_major_formula():
    # For shape (2, 3, 4), linear index is:
    # 1 + (i - 1) + (j - 1) * 2 + (k - 1) * 2 * 3
    assert sub2ind((2, 3, 4), 2, 2, 3) == 16
    assert sub2ind((2, 3, 4), 1, 3, 2) == 11


def test_sub2ind_same_input_same_output_is_deterministic():
    idx1 = sub2ind((4, 5), 3, 4)
    idx2 = sub2ind((4, 5), 3, 4)
    assert idx1 == idx2 == 15


def test_sub2ind_rejects_zero_index():
    with pytest.raises(Exception):
        sub2ind((3, 4), 0, 1)


def test_sub2ind_rejects_negative_index():
    with pytest.raises(Exception):
        sub2ind((3, 4), -1, 1)


def test_sub2ind_rejects_out_of_bounds_row():
    with pytest.raises(Exception):
        sub2ind((3, 4), 4, 1)


def test_sub2ind_rejects_out_of_bounds_column():
    with pytest.raises(Exception):
        sub2ind((3, 4), 1, 5)


def test_sub2ind_rejects_mismatched_number_of_dimensions():
    with pytest.raises(Exception):
        sub2ind((3, 4), 1)

    with pytest.raises(Exception):
        sub2ind((3, 4), 1, 2, 3)


def test_ind2sub_roundtrip_2d_scalar_index():
    shape = (3, 4)
    subs = ind2sub(shape, 7)
    assert tuple(subs) == (1, 3)
    assert sub2ind(shape, *subs) == 7


def test_ind2sub_roundtrip_3d_scalar_index():
    shape = (2, 3, 4)
    subs = ind2sub(shape, 16)
    assert tuple(subs) == (2, 2, 3)
    assert sub2ind(shape, *subs) == 16


def test_ind2sub_roundtrip_first_and_last_elements():
    shape = (5, 6)

    first = ind2sub(shape, 1)
    last = ind2sub(shape, 30)

    assert tuple(first) == (1, 1)
    assert tuple(last) == (5, 6)
    assert sub2ind(shape, *first) == 1
    assert sub2ind(shape, *last) == 30
