import pytest

from pymlfunc import sub2ind


def test_returns_one_for_first_element_2d():
    assert sub2ind((3, 4), 1, 1) == 1


def test_column_major_order_within_first_column():
    # MATLAB-style column-major ordering: rows advance fastest.
    assert sub2ind((3, 4), 2, 1) == 2
    assert sub2ind((3, 4), 3, 1) == 3


def test_column_major_order_advances_to_next_column():
    assert sub2ind((3, 4), 1, 2) == 4
    assert sub2ind((3, 4), 1, 3) == 7


def test_last_element_2d_equals_total_numel():
    assert sub2ind((3, 4), 3, 4) == 12


def test_single_row_matrix():
    assert sub2ind((1, 5), 1, 1) == 1
    assert sub2ind((1, 5), 1, 5) == 5


def test_single_column_matrix():
    assert sub2ind((5, 1), 4, 1) == 4


def test_square_matrix_corners():
    assert sub2ind((4, 4), 1, 1) == 1
    assert sub2ind((4, 4), 4, 1) == 4
    assert sub2ind((4, 4), 1, 4) == 13
    assert sub2ind((4, 4), 4, 4) == 16


def test_3d_first_and_second_page_offsets():
    # dims (2, 3, 4): index = i + (j - 1) * 2 + (k - 1) * 6
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1
    assert sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert sub2ind((2, 3, 4), 1, 1, 2) == 7


def test_3d_last_element_equals_total_numel():
    assert sub2ind((2, 3, 4), 2, 3, 4) == 24


def test_3d_middle_element():
    # 2 + (2 - 1) * 2 + (3 - 1) * 6 = 2 + 2 + 12 = 16
    assert sub2ind((2, 3, 4), 2, 2, 3) == 16


def test_trailing_singleton_dimension_is_ignored():
    # A (3, 4, 1) array has the same linear indexing as a (3, 4) array.
    assert sub2ind((3, 4, 1), 2, 3) == sub2ind((3, 4), 2, 3) == 8


def test_linear_indices_are_unique_and_cover_range():
    dims = (3, 4)
    indices = [
        sub2ind(dims, r, c)
        for c in range(1, dims[1] + 1)
        for r in range(1, dims[0] + 1)
    ]
    assert sorted(indices) == list(range(1, dims[0] * dims[1] + 1))


def test_row_subscript_zero_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 0, 1)


def test_row_subscript_beyond_dimension_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 4, 1)


def test_column_subscript_beyond_dimension_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 1, 5)


def test_negative_subscript_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), -1, 1)
