import pytest

from pymlfunc import sub2ind


def test_sub2ind_is_callable():
    assert callable(sub2ind)


def test_top_left_maps_to_one():
    assert sub2ind((3, 4), 1, 1) == 1


def test_bottom_right_maps_to_numel():
    assert sub2ind((3, 4), 3, 4) == 12


def test_column_major_order_second_column_start():
    # Column-major: the first column holds indices 1..3, so (1, 2) is 4.
    assert sub2ind((3, 4), 1, 2) == 4


def test_column_major_order_within_column():
    assert sub2ind((3, 4), 2, 2) == 5
    assert sub2ind((3, 4), 3, 2) == 6


def test_consecutive_rows_in_column_are_consecutive_indices():
    results = [sub2ind((5, 2), r, 1) for r in range(1, 6)]
    assert results == [1, 2, 3, 4, 5]


def test_consecutive_columns_jump_by_number_of_rows():
    results = [sub2ind((5, 3), 1, c) for c in range(1, 4)]
    assert results == [1, 6, 11]


def test_non_square_dims_last_element():
    assert sub2ind((2, 7), 2, 7) == 14


def test_single_row_matrix():
    for k in range(1, 6):
        assert sub2ind((1, 5), 1, k) == k


def test_single_column_matrix():
    for k in range(1, 6):
        assert sub2ind((5, 1), k, 1) == k


def test_square_matrix_all_indices_unique_and_complete():
    dims = (4, 4)
    indices = [
        sub2ind(dims, r, c)
        for r in range(1, dims[0] + 1)
        for c in range(1, dims[1] + 1)
    ]
    assert sorted(indices) == list(range(1, 17))


def test_three_dimensional_first_and_last():
    dims = (2, 3, 4)
    assert sub2ind(dims, 1, 1, 1) == 1
    assert sub2ind(dims, 2, 3, 4) == 24


def test_three_dimensional_middle_element():
    # index = i + (j - 1) * 2 + (k - 1) * 6
    assert sub2ind((2, 3, 4), 2, 2, 2) == 2 + 2 + 6


def test_returns_integer_type():
    result = sub2ind((3, 4), 2, 3)
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "dims, subs",
    [
        ((3, 4), (4, 1)),   # row too large
        ((3, 4), (1, 5)),   # column too large
        ((3, 4), (0, 1)),   # row zero
        ((3, 4), (1, 0)),   # column zero
        ((3, 4), (-1, 2)),  # negative row
    ],
)
def test_out_of_range_subscripts_raise(dims, subs):
    with pytest.raises((ValueError, IndexError)):
        sub2ind(dims, *subs)
