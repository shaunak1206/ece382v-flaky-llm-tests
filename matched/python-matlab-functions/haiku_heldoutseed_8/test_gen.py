import itertools

import pytest

from pymlfunc import sub2ind


def test_first_element_maps_to_one():
    assert sub2ind((3, 4), 1, 1) == 1


def test_last_element_2d():
    assert sub2ind((3, 4), 3, 4) == 12


def test_column_major_order_2d():
    # MATLAB uses column-major order: moving down a column increments by 1.
    assert sub2ind((3, 4), 2, 1) == 2
    assert sub2ind((3, 4), 3, 1) == 3
    # Moving to the next column jumps by the number of rows.
    assert sub2ind((3, 4), 1, 2) == 4


def test_matches_column_major_enumeration_2d():
    dims = (3, 4)
    expected = 1
    for col in range(1, dims[1] + 1):
        for row in range(1, dims[0] + 1):
            assert sub2ind(dims, row, col) == expected
            expected += 1


def test_matches_column_major_enumeration_3d():
    dims = (2, 3, 4)
    expected = 1
    for page in range(1, dims[2] + 1):
        for col in range(1, dims[1] + 1):
            for row in range(1, dims[0] + 1):
                assert sub2ind(dims, row, col, page) == expected
                expected += 1


def test_3d_known_values():
    dims = (2, 3, 4)
    assert sub2ind(dims, 1, 1, 1) == 1
    assert sub2ind(dims, 2, 1, 1) == 2
    assert sub2ind(dims, 1, 2, 1) == 3
    assert sub2ind(dims, 1, 1, 2) == 7
    assert sub2ind(dims, 2, 3, 4) == 24


def test_list_dims_same_as_tuple_dims():
    assert sub2ind([3, 4], 2, 3) == sub2ind((3, 4), 2, 3)


def test_single_row_vector():
    # A 1xN array: linear index equals the column index.
    for col in range(1, 6):
        assert sub2ind((1, 5), 1, col) == col


def test_single_column_vector():
    # An Nx1 array: linear index equals the row index.
    for row in range(1, 6):
        assert sub2ind((5, 1), row, 1) == row


def test_square_matrix_diagonal():
    n = 5
    for i in range(1, n + 1):
        assert sub2ind((n, n), i, i) == (i - 1) * n + i


def test_result_is_int():
    result = sub2ind((3, 4), 2, 3)
    assert isinstance(result, int)


def test_linear_indices_are_unique_and_cover_range():
    dims = (4, 3, 2)
    indices = [
        sub2ind(dims, r, c, p)
        for r, c, p in itertools.product(
            range(1, dims[0] + 1),
            range(1, dims[1] + 1),
            range(1, dims[2] + 1),
        )
    ]
    total = dims[0] * dims[1] * dims[2]
    assert sorted(indices) == list(range(1, total + 1))


@pytest.mark.parametrize(
    "dims, subs",
    [
        ((3, 4), (0, 1)),
        ((3, 4), (1, 0)),
        ((3, 4), (4, 1)),
        ((3, 4), (1, 5)),
        ((2, 3, 4), (1, 1, 0)),
        ((2, 3, 4), (3, 1, 1)),
        ((2, 3, 4), (1, 4, 1)),
        ((2, 3, 4), (1, 1, 5)),
    ],
)
def test_out_of_range_subscripts_raise(dims, subs):
    with pytest.raises((ValueError, IndexError)):
        sub2ind(dims, *subs)


def test_negative_subscript_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), -1, 1)


def test_subscript_count_mismatch_raises_or_is_rejected():
    # Supplying more subscripts than dimensions is invalid for a 2-D shape.
    with pytest.raises((ValueError, IndexError, TypeError)):
        sub2ind((3, 4), 1, 1, 1)


def test_non_integer_subscript_raises():
    with pytest.raises((ValueError, IndexError, TypeError)):
        sub2ind((3, 4), 1.5, 1)
