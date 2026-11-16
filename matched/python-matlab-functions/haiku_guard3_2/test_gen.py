"""Unit tests for sub2ind in pymlfunc/subind.py (re-exported by pymlfunc/__init__.py).

The tests assume MATLAB-style sub2ind semantics: subscripts are mapped to a
linear index in column-major (first-dimension-fastest) order. The subscript
origin is detected at runtime (1-based like MATLAB, or 0-based) so the tests
check the same mapping under either convention. No randomness, clocks, or
global state are used, so every test is deterministic.
"""
import itertools

import pytest

from pymlfunc import sub2ind


def _origin():
    """Return 1 if sub2ind is 1-based, 0 if it is 0-based.

    For dims (3, 4), subscript (1, 1) maps to linear index 1 in a 1-based
    column-major scheme and to index 4 in a 0-based scheme.
    """
    value = sub2ind((3, 4), 1, 1)
    if value == 1:
        return 1
    if value == 4:
        return 0
    pytest.fail("sub2ind((3, 4), 1, 1) returned {!r}; expected 1 or 4".format(value))


def _expected(dims, subs, origin):
    """Reference column-major linear index for subscripts in the given origin."""
    index = 0
    stride = 1
    for dim, sub in zip(dims, subs):
        index += (sub - origin) * stride
        stride *= dim
    return index + origin


def _all_subscripts(dims, origin):
    """All valid subscript tuples for dims, in the given origin."""
    return itertools.product(*[range(origin, origin + d) for d in dims])


def test_origin_is_zero_or_one():
    origin = _origin()
    assert origin in (0, 1)


def test_first_subscript_maps_to_origin():
    origin = _origin()
    assert sub2ind((3, 4), origin, origin) == origin


def test_single_element_dims():
    origin = _origin()
    assert sub2ind((1, 1), origin, origin) == origin


def test_row_vector_dims_are_linear():
    origin = _origin()
    for col in range(origin, origin + 5):
        assert sub2ind((1, 5), origin, col) == origin + (col - origin)


def test_column_vector_dims_are_linear():
    origin = _origin()
    for row in range(origin, origin + 5):
        assert sub2ind((5, 1), row, origin) == origin + (row - origin)


def test_last_element_maps_to_last_index_2d():
    origin = _origin()
    last = sub2ind((3, 4), origin + 2, origin + 3)
    assert last == origin + 11


def test_column_major_order_2d():
    origin = _origin()
    dims = (3, 4)
    for row, col in _all_subscripts(dims, origin):
        assert sub2ind(dims, row, col) == _expected(dims, (row, col), origin)


def test_moving_down_a_column_increments_by_one():
    origin = _origin()
    rows = 5
    for col in range(origin, origin + 3):
        for row in range(origin, origin + rows - 1):
            assert sub2ind((rows, 3), row + 1, col) - sub2ind((rows, 3), row, col) == 1


def test_moving_to_next_column_increments_by_row_count():
    origin = _origin()
    rows, cols = 5, 3
    for row in range(origin, origin + rows):
        for col in range(origin, origin + cols - 1):
            diff = sub2ind((rows, cols), row, col + 1) - sub2ind((rows, cols), row, col)
            assert diff == rows


def test_2d_mapping_is_bijection_onto_contiguous_range():
    origin = _origin()
    dims = (4, 6)
    values = sorted(sub2ind(dims, r, c) for r, c in _all_subscripts(dims, origin))
    assert values == list(range(origin, origin + 4 * 6))


def test_3d_column_major_order():
    origin = _origin()
    dims = (2, 3, 4)
    for subs in _all_subscripts(dims, origin):
        assert sub2ind(dims, *subs) == _expected(dims, subs, origin)


def test_3d_mapping_is_bijection():
    origin = _origin()
    dims = (2, 3, 2)
    values = [sub2ind(dims, *subs) for subs in _all_subscripts(dims, origin)]
    assert len(set(values)) == 2 * 3 * 2
    assert min(values) == origin
    assert max(values) == origin + 2 * 3 * 2 - 1


def test_4d_mapping_is_bijection():
    origin = _origin()
    dims = (2, 2, 2, 3)
    values = [sub2ind(dims, *subs) for subs in _all_subscripts(dims, origin)]
    assert sorted(values) == list(range(origin, origin + 2 * 2 * 2 * 3))


def test_large_dims_corner_matches_formula():
    origin = _origin()
    dims = (100, 200)
    row, col = origin + 99, origin + 199
    assert sub2ind(dims, row, col) == origin + 100 * 200 - 1


def test_result_is_integer_valued():
    origin = _origin()
    result = sub2ind((3, 4), origin + 1, origin + 2)
    assert int(result) == result


def test_repeated_calls_are_consistent():
    origin = _origin()
    first = sub2ind((7, 9), origin + 4, origin + 5)
    second = sub2ind((7, 9), origin + 4, origin + 5)
    assert first == second


def test_out_of_range_row_raises():
    origin = _origin()
    with pytest.raises((IndexError, ValueError)):
        sub2ind((3, 4), origin + 3, origin)


def test_out_of_range_column_raises():
    origin = _origin()
    with pytest.raises((IndexError, ValueError)):
        sub2ind((3, 4), origin, origin + 4)
