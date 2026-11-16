"""Unit tests for pymlfunc.sub2ind.

The tests use a pure-Python reference implementation. Because the expected
index base (0- or 1-based) and memory order (column- or row-major) are not
fixed by the API shown here, a module-level fixture first probes sub2ind to
detect its convention, then checks every other result against that convention.
Each test also asserts that the detected convention is consistent across probes.
"""

import itertools

import pytest

from pymlfunc import sub2ind, ind2sub


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _strides(dims, order):
    """Return per-dimension strides for 'col' (MATLAB-style) or 'row' order."""
    n = len(dims)
    strides = [1] * n
    if order == "col":
        for k in range(1, n):
            strides[k] = strides[k - 1] * dims[k - 1]
    elif order == "row":
        for k in range(n - 2, -1, -1):
            strides[k] = strides[k + 1] * dims[k + 1]
    else:
        raise ValueError(order)
    return strides


def _expected(dims, subs, base, order):
    """Reference linear index for subscripts `subs` in an array of shape `dims`."""
    strides = _strides(dims, order)
    offset = sum((s - base) * st for s, st in zip(subs, strides))
    return base + offset


_PROBES = [
    ((3, 4), (1, 2)),
    ((3, 4), (2, 1)),
    ((3, 4), (2, 3)),
    ((3, 4), (1, 1)),
]


@pytest.fixture(scope="module")
def convention():
    """Detect (base, order) of sub2ind by matching probe results."""
    for base, order in itertools.product((0, 1), ("col", "row")):
        if all(
            sub2ind(dims, *subs) == _expected(dims, subs, base, order)
            for dims, subs in _PROBES
        ):
            return base, order
    pytest.fail("sub2ind does not match any supported index convention")


def _grid(dims, base):
    """All valid subscript tuples for `dims` under the given base."""
    ranges = [range(base, base + d) for d in dims]
    return list(itertools.product(*ranges))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_integer_for_scalar_subscripts(convention):
    base, _ = convention
    result = sub2ind((3, 4), base, base)
    assert isinstance(result, int)


def test_first_element_maps_to_base(convention):
    base, _ = convention
    assert sub2ind((3, 4), base, base) == base


def test_last_element_maps_to_numel(convention):
    base, _ = convention
    dims = (3, 4)
    last = (base + dims[0] - 1, base + dims[1] - 1)
    assert sub2ind(dims, *last) == base + dims[0] * dims[1] - 1


@pytest.mark.parametrize("dims", [(3, 4), (1, 5), (5, 1), (2, 2), (4, 7)])
def test_matches_reference_over_full_grid(convention, dims):
    base, order = convention
    for subs in _grid(dims, base):
        assert sub2ind(dims, *subs) == _expected(dims, subs, base, order)


@pytest.mark.parametrize("dims", [(3, 4), (2, 5), (6, 3)])
def test_mapping_is_bijective_over_grid(convention, dims):
    base, _ = convention
    indices = [sub2ind(dims, *subs) for subs in _grid(dims, base)]
    assert len(set(indices)) == len(indices) == dims[0] * dims[1]


@pytest.mark.parametrize("dims", [(3, 4), (2, 5), (4, 4)])
def test_indices_cover_contiguous_range(convention, dims):
    base, _ = convention
    indices = sorted(sub2ind(dims, *subs) for subs in _grid(dims, base))
    assert indices == list(range(base, base + dims[0] * dims[1]))


def test_single_row_array(convention):
    base, order = convention
    dims = (1, 6)
    for col in range(base, base + dims[1]):
        assert sub2ind(dims, base, col) == _expected(dims, (base, col), base, order)


def test_single_column_array(convention):
    base, order = convention
    dims = (6, 1)
    for row in range(base, base + dims[0]):
        assert sub2ind(dims, row, base) == _expected(dims, (row, base), base, order)


def test_row_neighbours_differ_by_one(convention):
    base, _ = convention
    dims = (5, 5)
    # Moving down one row changes the linear index by exactly 1 in column-major order.
    if convention[1] == "col":
        assert sub2ind(dims, base + 1, base) - sub2ind(dims, base, base) == 1
    else:
        assert sub2ind(dims, base + 1, base) - sub2ind(dims, base, base) == dims[1]


def test_column_neighbours_differ_by_stride(convention):
    base, order = convention
    dims = (5, 5)
    step = dims[0] if order == "col" else 1
    delta = sub2ind(dims, base, base + 1) - sub2ind(dims, base, base)
    assert delta == step


def test_three_dimensional_matches_reference(convention):
    base, order = convention
    dims = (2, 3, 4)
    for subs in _grid(dims, base):
        assert sub2ind(dims, *subs) == _expected(dims, subs, base, order)


def test_round_trip_with_ind2sub(convention):
    base, _ = convention
    dims = (4, 6)
    for subs in _grid(dims, base):
        linear = sub2ind(dims, *subs)
        assert tuple(ind2sub(dims, linear)) == subs


@pytest.mark.parametrize("which", ["row_high", "row_low", "col_high", "col_low"])
def test_out_of_range_subscripts_raise(convention, which):
    base, _ = convention
    dims = (3, 4)
    row, col = base, base
    if which == "row_high":
        row = base + dims[0]
    elif which == "row_low":
        row = base - 1
    elif which == "col_high":
        col = base + dims[1]
    else:
        col = base - 1
    with pytest.raises((ValueError, IndexError)):
        sub2ind(dims, row, col)
