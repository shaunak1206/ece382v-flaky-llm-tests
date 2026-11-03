import numpy as np
import numpy.random
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


GRID = [10, 10]
CELL = [1, 1]


def _pin(monkeypatch, fx=0.5, fy=0.5, pick=0, log=None):
    """Make make_blot deterministic.

    make_blot imports numpy.random.random and numpy.random.choice at call
    time, so patching those attributes on the numpy.random module is enough.
    The start cell is (floor(fx * width), floor(fy * height)). Every weighted
    draw returns the candidate at position `pick`, and the weight vector it
    was given is appended to `log` when a list is supplied.
    """
    fractions = iter([fx, fy])
    monkeypatch.setattr(numpy.random, "random", lambda: next(fractions))

    def fake_choice(indices, p=None):
        if log is not None:
            log.append(np.asarray(p, dtype=float))
        return int(indices[pick])

    monkeypatch.setattr(numpy.random, "choice", fake_choice)


def _cells(blot):
    out = []
    blot.for_each_cell(lambda x, y: out.append((x, y)))
    return out


def _mask(blocked=()):
    m = np.ones(GRID, dtype=bool)
    for x, y in blocked:
        m[x, y] = False
    return m


def test_new_blot_is_empty():
    assert DyeBlot([4, 4], [1, 1]).num_cells() == 0


def test_for_each_cell_on_empty_blot_calls_nothing():
    calls = []
    DyeBlot([4, 4], [1, 1]).for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_two_ints(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 3)
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert len(seen) == 3
    for x, y in seen:
        assert isinstance(x, int)
        assert isinstance(y, int)


def test_for_each_cell_visits_each_cell_once(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 6, squeeze=True)
    seen = _cells(blot)
    assert len(seen) == blot.num_cells()
    assert len(set(seen)) == len(seen)


def test_for_each_cell_returns_none(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 2)
    assert blot.for_each_cell(lambda x, y: None) is None


def test_for_each_cell_propagates_exceptions(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 2)

    def boom(x, y):
        raise ValueError("stop")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


def test_start_on_masked_cell_returns_none(monkeypatch):
    _pin(monkeypatch)
    assert make_blot(GRID, CELL, 3, permitted=_mask(blocked=[(5, 5)])) is None


def test_start_on_masked_cell_returns_none_when_squeezing(monkeypatch):
    _pin(monkeypatch)
    mask = _mask(blocked=[(5, 5)])
    assert make_blot(GRID, CELL, 3, permitted=mask, squeeze=True) is None


def test_single_required_cell_is_the_start(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 1)
    assert _cells(blot) == [(5, 5)]


def test_zero_required_cells_still_yields_start_cell(monkeypatch):
    # The start cell is added before the size check, so a request for zero
    # cells still yields a one-cell blot.
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 0)
    assert _cells(blot) == [(5, 5)]


def test_growth_order_with_fixed_choice(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 3)
    assert _cells(blot) == [(5, 5), (4, 5), (5, 4)]


def test_grows_to_requested_size_in_open_grid(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 25, squeeze=True)
    assert blot.num_cells() == 25


def test_cells_are_unique_and_in_bounds(monkeypatch):
    _pin(monkeypatch)
    blot = make_blot(GRID, CELL, 25, squeeze=True)
    cells = _cells(blot)
    assert len(set(cells)) == len(cells)
    assert all(0 <= x < GRID[0] and 0 <= y < GRID[1] for x, y in cells)


def test_no_squeeze_returns_none_at_grid_edge(monkeypatch):
    # From (0, 0) the first weighted draw is (-1, 0), which is off the grid.
    _pin(monkeypatch, fx=0.0, fy=0.0)
    assert make_blot(GRID, CELL, 5) is None


def test_squeeze_discards_out_of_bounds_neighbours(monkeypatch):
    _pin(monkeypatch, fx=0.0, fy=0.0)
    blot = make_blot(GRID, CELL, 3, squeeze=True)
    cells = _cells(blot)
    assert blot.num_cells() == 3
    assert cells[0] == (0, 0)
    assert all(0 <= x < GRID[0] and 0 <= y < GRID[1] for x, y in cells)


def test_no_squeeze_rejects_masked_neighbour(monkeypatch):
    # The first weighted draw from (5, 5) is (4, 5), which is masked.
    _pin(monkeypatch)
    mask = _mask(blocked=[(4, 5)])
    assert make_blot(GRID, CELL, 2, permitted=mask, squeeze=False) is None


def test_squeeze_grows_around_blocked_column(monkeypatch):
    _pin(monkeypatch)
    mask = _mask(blocked=[(4, y) for y in range(GRID[1])])
    blot = make_blot(GRID, CELL, 10, permitted=mask, squeeze=True)
    cells = _cells(blot)
    assert blot.num_cells() == 10
    assert all(x >= 5 for x, _ in cells)


def test_loop_stops_when_region_exhausted(monkeypatch):
    # A 3 x 1 grid holds only 3 cells, so the blot stops short of the
    # requested 10 once every neighbour has been consumed.
    _pin(monkeypatch)
    blot = make_blot([3, 1], CELL, 10, squeeze=True)
    assert sorted(_cells(blot)) == [(0, 0), (1, 0), (2, 0)]


def test_edge_weights_follow_cell_dimensions(monkeypatch):
    # cell_dims = [2, 3]: left and right neighbours share a 3-unit edge,
    # top and bottom neighbours share a 2-unit edge.
    log = []
    _pin(monkeypatch, log=log)
    make_blot(GRID, [2, 3], 2)
    assert len(log) == 1
    assert np.allclose(log[0], [0.3, 0.2, 0.3, 0.2])


def test_shared_neighbour_weight_accumulates(monkeypatch):
    # After (5, 5), (4, 5) and (5, 4) are in the blot, the neighbour (4, 4)
    # borders both (4, 5) and (5, 4), so its weight is 2 while the other
    # frontier cells have weight 1.
    log = []
    _pin(monkeypatch, log=log)
    make_blot(GRID, CELL, 4)
    assert len(log) == 3
    assert np.allclose(sorted(log[2]), sorted([1 / 8.0] * 6 + [2 / 8.0]))


def test_explicit_all_true_mask_matches_default(monkeypatch):
    _pin(monkeypatch)
    default_cells = _cells(make_blot(GRID, CELL, 12, squeeze=True))
    _pin(monkeypatch)
    masked_cells = _cells(
        make_blot(GRID, CELL, 12, permitted=_mask(), squeeze=True)
    )
    assert default_cells == masked_cells


def test_integer_mask_matches_boolean_mask(monkeypatch):
    blocked = [(4, y) for y in range(GRID[1])]
    int_mask = np.ones(GRID, dtype=int)
    for x, y in blocked:
        int_mask[x, y] = 0

    _pin(monkeypatch)
    from_bool = _cells(
        make_blot(GRID, CELL, 10, permitted=_mask(blocked=blocked), squeeze=True)
    )
    _pin(monkeypatch)
    from_int = _cells(make_blot(GRID, CELL, 10, permitted=int_mask, squeeze=True))
    assert from_bool == from_int
