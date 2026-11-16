"""Unit tests for sub2ind in pymlfunc.

Assumed convention (MATLAB-compatible): subscripts are 1-based and the
linear index is column-major, so for dims (R, C) the element (r, c) maps to
(c - 1) * R + r.
"""
import pytest

from pymlfunc import sub2ind


# --- Basic 2-D behaviour -----------------------------------------------------

def test_first_element_maps_to_one():
    assert sub2ind((3, 4), 1, 1) == 1


def test_column_major_order_down_first_column():
    assert sub2ind((3, 4), 2, 1) == 2
    assert sub2ind((3, 4), 3, 1) == 3


def test_moves_to_next_column_after_rows_exhausted():
    assert sub2ind((3, 4), 1, 2) == 4


def test_last_element_maps_to_total_count():
    assert sub2ind((3, 4), 3, 4) == 12


def test_square_matrix_diagonal():
    assert sub2ind((3, 3), 1, 1) == 1
    assert sub2ind((3, 3), 2, 2) == 5
    assert sub2ind((3, 3), 3, 3) == 9


def test_single_row_matrix_is_sequential():
    for col in range(1, 6):
        assert sub2ind((1, 5), 1, col) == col


def test_single_column_matrix_is_sequential():
    for row in range(1, 6):
        assert sub2ind((5, 1), row, 1) == row


def test_single_element_matrix():
    assert sub2ind((1, 1), 1, 1) == 1


def test_non_square_tall_matrix():
    # 5 rows, 2 columns: element (4, 2) is the 9th in column-major order.
    assert sub2ind((5, 2), 4, 2) == 9


def test_non_square_wide_matrix():
    # 2 rows, 5 columns: element (2, 5) is the last of 10.
    assert sub2ind((2, 5), 2, 5) == 10


def test_result_is_int():
    result = sub2ind((4, 4), 2, 3)
    assert isinstance(result, int)
    assert result == 10


def test_accepts_list_dims():
    assert sub2ind([3, 4], 2, 2) == 5


# --- Basic N-D behaviour -----------------------------------------------------

def test_3d_origin_maps_to_one():
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1


def test_3d_first_page_offsets():
    # dims (2, 3, 4): page k starts at (k - 1) * 6 + 1
    assert sub2ind((2, 3, 4), 1, 1, 2) == 7
    assert sub2ind((2, 3, 4), 1, 1, 3) == 13


def test_3d_last_element_maps_to_total_count():
    assert sub2ind((2, 3, 4), 2, 3, 4) == 24


def test_3d_mixed_subscripts():
    assert sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert sub2ind((2, 3, 4), 1, 2, 1) == 3


# --- Error handling ----------------------------------------------------------

def test_row_zero_is_invalid():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 0, 1)


def test_column_zero_is_invalid():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 1, 0)


def test_row_out_of_bounds_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 4, 1)


def test_column_out_of_bounds_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 1, 5)


def test_negative_subscript_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), -1, 2)


def test_subscript_count_mismatch_raises():
    with pytest.raises((ValueError, IndexError, TypeError)):
        sub2ind((2, 3, 4), 1, 1)
