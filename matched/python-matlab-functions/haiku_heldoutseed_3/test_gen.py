"""Unit tests for pymlfunc.sub2ind.

These tests follow MATLAB semantics: subscripts and linear indices are
1-based, and linear indices are column-major (the first subscript varies
fastest). Example: sub2ind([3 4], 2, 3) == 8 in MATLAB.
"""
import itertools

import pytest

from pymlfunc import ind2sub, sub2ind


def test_sub2ind_is_callable():
    assert callable(sub2ind)


def test_first_element_maps_to_one():
    assert sub2ind((3, 4), 1, 1) == 1


def test_last_element_maps_to_numel():
    assert sub2ind((3, 4), 3, 4) == 12


def test_matlab_documented_example():
    # MATLAB: sub2ind([3 4], 2, 3) -> 8
    assert sub2ind((3, 4), 2, 3) == 8


def test_column_major_order_row_step_is_one():
    assert sub2ind((3, 4), 2, 1) - sub2ind((3, 4), 1, 1) == 1
    assert sub2ind((3, 4), 3, 1) - sub2ind((3, 4), 2, 1) == 1


def test_column_major_order_column_step_is_number_of_rows():
    rows, cols = 3, 4
    for r in range(1, rows + 1):
        for c in range(1, cols):
            step = sub2ind((rows, cols), r, c + 1) - sub2ind((rows, cols), r, c)
            assert step == rows


def test_all_subscripts_map_to_unique_indices_covering_range():
    dims = (3, 4)
    results = [
        sub2ind(dims, r, c)
        for r, c in itertools.product(range(1, dims[0] + 1), range(1, dims[1] + 1))
    ]
    assert sorted(results) == list(range(1, dims[0] * dims[1] + 1))


def test_round_trip_with_ind2sub_2d():
    dims = (3, 5)
    for r, c in itertools.product(range(1, dims[0] + 1), range(1, dims[1] + 1)):
        idx = sub2ind(dims, r, c)
        assert tuple(ind2sub(dims, idx)) == (r, c)


def test_single_row_matrix():
    assert sub2ind((1, 5), 1, 3) == 3


def test_single_column_matrix():
    assert sub2ind((5, 1), 4, 1) == 4


def test_single_element_matrix():
    assert sub2ind((1, 1), 1, 1) == 1


def test_three_dimensional_indexing():
    # index = r + (c - 1) * 2 + (p - 1) * 6 for dims (2, 3, 4)
    dims = (2, 3, 4)
    assert sub2ind(dims, 1, 1, 1) == 1
    assert sub2ind(dims, 2, 1, 1) == 2
    assert sub2ind(dims, 1, 2, 1) == 3
    assert sub2ind(dims, 1, 1, 2) == 7
    assert sub2ind(dims, 2, 3, 4) == 24


def test_round_trip_with_ind2sub_3d():
    dims = (2, 3, 4)
    for subs in itertools.product(
        range(1, dims[0] + 1), range(1, dims[1] + 1), range(1, dims[2] + 1)
    ):
        idx = sub2ind(dims, *subs)
        assert tuple(ind2sub(dims, idx)) == subs


def test_large_matrix_last_element():
    assert sub2ind((100, 200), 100, 200) == 20000


@pytest.mark.parametrize("dims, subs", [
    ((3, 4), (4, 1)),   # row too large
    ((3, 4), (1, 5)),   # column too large
    ((3, 4), (10, 10)), # both too large
])
def test_out_of_range_subscripts_raise(dims, subs):
    with pytest.raises((ValueError, IndexError)):
        sub2ind(dims, *subs)


@pytest.mark.parametrize("subs", [(0, 1), (1, 0), (-1, 1)])
def test_non_positive_subscripts_raise(subs):
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), *subs)


def test_out_of_range_in_third_dimension_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((2, 3, 4), 1, 1, 5)
