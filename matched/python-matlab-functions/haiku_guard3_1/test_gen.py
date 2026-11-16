"""Unit tests for sub2ind in pymlfunc/subind.py, exported via pymlfunc/__init__.py.

Assumed contract (MATLAB-compatible, the purpose of the project):
    sub2ind(dims, rows, cols) -> linear index
    - rows and cols are 1-based subscripts
    - linear indices are 1-based and column-major (MATLAB order)
    - scalar inputs return a scalar; sequence inputs return a sequence of the same length
    - out-of-range subscripts raise ValueError or IndexError

sub2ind uses no randomness, no hashing, no clock, and no global state, so no seeding,
monkeypatching, or tolerance comparisons are needed. All values are exact integers.
"""

import pymlfunc
from pymlfunc import sub2ind
import pymlfunc.subind as subind_module
import pytest


def _to_list(result):
    """Normalize a scalar or sequence result to a plain Python list."""
    if isinstance(result, (list, tuple)):
        return [int(x) for x in result]
    try:
        return [int(x) for x in result]
    except TypeError:
        return [int(result)]


def test_export_is_same_object_as_module_function():
    assert pymlfunc.sub2ind is subind_module.sub2ind


def test_top_left_corner_is_one():
    assert _to_list(sub2ind((3, 4), 1, 1)) == [1]


def test_bottom_right_corner_is_last_index():
    assert _to_list(sub2ind((3, 4), 3, 4)) == [12]


def test_first_row_second_column_is_row_count_plus_one():
    # Column-major: column 2 starts after the 3 elements of column 1.
    assert _to_list(sub2ind((3, 4), 1, 2)) == [4]


def test_last_row_first_column():
    assert _to_list(sub2ind((3, 4), 3, 1)) == [3]


def test_column_major_full_enumeration_2x3():
    dims = (2, 3)
    got = []
    for c in range(1, 4):
        for r in range(1, 3):
            got.extend(_to_list(sub2ind(dims, r, c)))
    assert got == [1, 2, 3, 4, 5, 6]


def test_row_varies_fastest_within_a_column():
    values = [_to_list(sub2ind((4, 5), r, 2))[0] for r in range(1, 5)]
    assert values == [5, 6, 7, 8]
    assert all(b - a == 1 for a, b in zip(values, values[1:]))


def test_vector_rows_and_cols_match_elementwise_scalar_calls():
    rows = [1, 2, 3, 2]
    cols = [1, 2, 3, 4]
    expected = [_to_list(sub2ind((3, 4), r, c))[0] for r, c in zip(rows, cols)]
    assert _to_list(sub2ind((3, 4), rows, cols)) == expected


def test_vector_inputs_known_values():
    assert _to_list(sub2ind((3, 4), [1, 3], [1, 4])) == [1, 12]


def test_tuple_inputs_give_same_result_as_list_inputs():
    assert _to_list(sub2ind((3, 4), (1, 2, 3), (1, 2, 4))) == _to_list(
        sub2ind((3, 4), [1, 2, 3], [1, 2, 4])
    )


def test_dims_as_list_matches_dims_as_tuple():
    assert _to_list(sub2ind([3, 4], 2, 3)) == _to_list(sub2ind((3, 4), 2, 3))


def test_single_row_matrix():
    # dims (1, 5): linear index equals column index.
    assert _to_list(sub2ind((1, 5), 1, [1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]


def test_single_column_matrix():
    # dims (5, 1): linear index equals row index.
    assert _to_list(sub2ind((5, 1), [1, 2, 3, 4, 5], 1)) == [1, 2, 3, 4, 5]


def test_one_by_one_matrix():
    assert _to_list(sub2ind((1, 1), 1, 1)) == [1]


def test_empty_subscripts_return_empty_result():
    assert _to_list(sub2ind((3, 4), [], [])) == []


def test_large_dims_corner_values():
    assert _to_list(sub2ind((1000, 1000), 1000, 1000)) == [1000000]
    assert _to_list(sub2ind((1000, 1000), 1, 1000)) == [999001]


def test_mapping_is_bijective_over_full_grid():
    dims = (4, 3)
    indices = [
        _to_list(sub2ind(dims, r, c))[0]
        for r in range(1, dims[0] + 1)
        for c in range(1, dims[1] + 1)
    ]
    assert sorted(indices) == list(range(1, dims[0] * dims[1] + 1))


@pytest.mark.parametrize("row", [0, -1])
def test_row_below_one_raises(row):
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), row, 1)


def test_row_above_dimension_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 4, 1)


@pytest.mark.parametrize("col", [0, -2])
def test_column_below_one_raises(col):
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 1, col)


def test_column_above_dimension_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), 1, 5)


def test_vector_with_one_out_of_range_element_raises():
    with pytest.raises((ValueError, IndexError)):
        sub2ind((3, 4), [1, 2, 9], [1, 1, 1])
