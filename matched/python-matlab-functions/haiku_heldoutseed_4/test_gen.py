"""Unit tests for sub2ind in pymlfunc.

These tests follow MATLAB's sub2ind semantics: subscripts are 1-based,
linear indices are 1-based, and the layout is column-major.
"""
import itertools

import pytest

from pymlfunc import sub2ind


def _reference_sub2ind(dims, *subs):
    """Independent reference implementation of MATLAB's column-major sub2ind."""
    dims = list(dims)
    # Pad trailing singleton dimensions so each subscript has a matching size.
    while len(dims) < len(subs):
        dims.append(1)
    index = 0
    stride = 1
    for i, sub in enumerate(subs):
        index += (sub - 1) * stride
        stride *= dims[i]
    return index + 1


def test_module_exports_sub2ind():
    assert callable(sub2ind)


def test_first_element_maps_to_one_2d():
    assert sub2ind((3, 4), 1, 1) == 1


def test_last_element_maps_to_numel_2d():
    assert sub2ind((3, 4), 3, 4) == 12


def test_column_major_order_2d():
    # Row 2, column 1 should come right after row 1, column 1.
    assert sub2ind((2, 2), 2, 1) == 2
    assert sub2ind((2, 2), 1, 2) == 3


def test_column_major_enumeration_2d():
    rows, cols = 3, 4
    expected = 1
    for c in range(1, cols + 1):
        for r in range(1, rows + 1):
            assert sub2ind((rows, cols), r, c) == expected
            expected += 1


def test_single_row_matrix():
    assert sub2ind((1, 5), 1, 3) == 3


def test_single_column_matrix():
    assert sub2ind((5, 1), 4, 1) == 4


def test_square_matrix_corner_values():
    assert sub2ind((4, 4), 4, 1) == 4
    assert sub2ind((4, 4), 1, 4) == 13
    assert sub2ind((4, 4), 4, 4) == 16


def test_3d_basic_values():
    assert sub2ind((2, 3, 4), 1, 1, 1) == 1
    assert sub2ind((2, 3, 4), 2, 1, 1) == 2
    assert sub2ind((2, 3, 4), 1, 2, 1) == 3
    assert sub2ind((2, 3, 4), 1, 1, 2) == 7
    assert sub2ind((2, 3, 4), 2, 3, 4) == 24


def test_3d_matches_reference_exhaustively():
    dims = (2, 3, 4)
    for subs in itertools.product(*(range(1, d + 1) for d in dims)):
        assert sub2ind(dims, *subs) == _reference_sub2ind(dims, *subs)


def test_4d_last_element_is_numel():
    dims = (2, 2, 2, 2)
    assert sub2ind(dims, 2, 2, 2, 2) == 16


def test_trailing_singleton_dimension_is_ignored():
    # sub2ind((3, 4, 1), r, c, 1) should equal sub2ind((3, 4), r, c).
    for r, c in itertools.product(range(1, 4), range(1, 5)):
        assert sub2ind((3, 4, 1), r, c, 1) == sub2ind((3, 4), r, c)


def test_dims_as_list_and_tuple_agree():
    assert sub2ind([3, 4], 2, 3) == sub2ind((3, 4), 2, 3)


def test_vectorized_subscripts_return_linear_indices():
    rows = [1, 2, 3]
    cols = [1, 1, 2]
    result = sub2ind((3, 4), rows, cols)
    assert list(result) == [1, 2, 6]


@pytest.mark.parametrize(
    "dims, subs",
    [
        ((3, 4), (0, 1)),
        ((3, 4), (1, 0)),
        ((3, 4), (4, 1)),
        ((3, 4), (1, 5)),
        ((2, 3, 4), (1, 1, 5)),
    ],
)
def test_out_of_range_subscripts_raise(dims, subs):
    with pytest.raises((ValueError, IndexError)):
        sub2ind(dims, *subs)


def test_large_dimensions_no_overflow_in_value():
    assert sub2ind((1000, 1000), 1000, 1000) == 1_000_000


def test_result_matches_reference_for_random_like_sample():
    dims = (5, 7, 3)
    samples = [(1, 1, 1), (5, 7, 3), (2, 4, 2), (3, 6, 1), (4, 2, 3)]
    for subs in samples:
        assert sub2ind(dims, *subs) == _reference_sub2ind(dims, *subs)
