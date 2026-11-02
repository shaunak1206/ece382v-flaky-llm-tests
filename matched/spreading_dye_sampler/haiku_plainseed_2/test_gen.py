"""Unit tests for spreading_dye_sampler.dye_blot (DyeBlot and make_blot)."""

import collections

import numpy as np
import numpy.random
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _cells_of(blot):
    """Collect the cells of a blot through for_each_cell, in visiting order."""
    out = []
    blot.for_each_cell(lambda x, y: out.append((x, y)))
    return out


def _force_start(monkeypatch, grid_dims, start):
    """Make make_blot seed its blot at `start`.

    make_blot calls numpy.random.random() once for x and once for y, and
    takes floor(u * width) and floor(v * height). Feeding the middle of the
    corresponding cell makes the floor land exactly on `start`.
    """
    width, height = grid_dims
    values = iter([(start[0] + 0.5) / width, (start[1] + 0.5) / height])
    monkeypatch.setattr(numpy.random, "random", lambda: next(values))


def _always_first(monkeypatch, calls=None):
    """Replace numpy.random.choice with one that always returns index 0.

    If `calls` is a list, each call's probability vector is appended to it.
    """

    def fake_choice(indices, p=None):
        if calls is not None:
            calls.append(np.asarray(p, dtype=float))
        return 0

    monkeypatch.setattr(numpy.random, "choice", fake_choice)


def _is_four_connected(cells):
    """True if the set of cells is one region under 4-neighbour adjacency."""
    cells = set(cells)
    if not cells:
        return True
    start = next(iter(cells))
    seen = {start}
    queue = collections.deque([start])
    while queue:
        x, y = queue.popleft()
        for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if n in cells and n not in seen:
                seen.add(n)
                queue.append(n)
    return seen == cells


# ---------------------------------------------------------------------------
# DyeBlot
# ---------------------------------------------------------------------------


def test_new_dye_blot_has_no_cells():
    blot = DyeBlot([4, 4], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([4, 4], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_then_y_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(2, 2), (1, 2), (2, 1)]
    seen = []
    blot.for_each_cell(lambda *args: seen.append(args))
    assert seen == [(2, 2), (1, 2), (2, 1)]


def test_for_each_cell_visits_each_cell_of_a_built_blot_once():
    blot = make_blot([8, 8], [1, 1], 12, squeeze=True)
    visited = _cells_of(blot)
    assert len(visited) == blot.num_cells() == 12
    assert len(set(visited)) == 12


def test_for_each_cell_propagates_callback_errors():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0)]

    def boom(x, y):
        raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        blot.for_each_cell(boom)


def test_for_each_cell_leaves_blot_unchanged():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 0)]
    blot.for_each_cell(lambda x, y: None)
    assert blot.num_cells() == 2
    assert _cells_of(blot) == [(0, 0), (1, 0)]


# ---------------------------------------------------------------------------
# make_blot: seeding and the trivial cases
# ---------------------------------------------------------------------------


def test_make_blot_returns_dye_blot_instance(monkeypatch):
    _force_start(monkeypatch, [4, 4], (1, 1))
    blot = make_blot([4, 4], [2, 2], 1)
    assert isinstance(blot, DyeBlot)


@pytest.mark.parametrize("squeeze", [False, True])
def test_returns_none_when_start_cell_is_masked(monkeypatch, squeeze):
    grid = [4, 4]
    _force_start(monkeypatch, grid, (2, 1))
    permitted = np.ones((4, 4), dtype=bool)
    permitted[2, 1] = False
    assert make_blot(grid, [1, 1], 3, permitted=permitted, squeeze=squeeze) is None


def test_start_cell_is_floor_of_random_fraction_times_grid_dims(monkeypatch):
    # grid is 4 wide and 8 high: floor(0.5 * 4) = 2, floor(0.75 * 8) = 6.
    values = iter([0.5, 0.75])
    monkeypatch.setattr(numpy.random, "random", lambda: next(values))
    blot = make_blot([4, 8], [1, 1], 1)
    assert _cells_of(blot) == [(2, 6)]


def test_single_cell_request_never_draws_a_neighbour(monkeypatch):
    _force_start(monkeypatch, [5, 5], (2, 2))

    def no_choice(*args, **kwargs):
        raise AssertionError("choice should not be called for a 1-cell blot")

    monkeypatch.setattr(numpy.random, "choice", no_choice)
    blot = make_blot([5, 5], [1, 1], 1)
    assert _cells_of(blot) == [(2, 2)]


def test_one_by_one_grid_needs_squeeze_to_return_a_blot():
    # Every neighbour of the only cell is off the grid, so without squeeze
    # the first draw fails and the blot is abandoned.
    assert make_blot([1, 1], [1, 1], 2) is None
    blot = make_blot([1, 1], [1, 1], 2, squeeze=True)
    assert _cells_of(blot) == [(0, 0)]


def test_squeeze_false_returns_none_when_first_draw_leaves_grid(monkeypatch):
    grid = [2, 2]
    _force_start(monkeypatch, grid, (0, 0))
    _always_first(monkeypatch)  # first neighbour in order is (-1, 0)
    assert make_blot(grid, [1, 1], 2) is None


# ---------------------------------------------------------------------------
# make_blot: growth and structure
# ---------------------------------------------------------------------------


def test_squeeze_true_reaches_requested_size_on_open_grid():
    blot = make_blot([10, 10], [1, 1], 25, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 25


def test_squeeze_true_saturates_at_grid_size():
    blot = make_blot([2, 2], [1, 1], 10, squeeze=True)
    assert set(_cells_of(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_cells_are_unique_and_inside_grid():
    grid = [6, 4]
    blot = make_blot(grid, [1, 1], 15, squeeze=True)
    cells = _cells_of(blot)
    assert len(set(cells)) == len(cells) == 15
    for x, y in cells:
        assert 0 <= x < grid[0]
        assert 0 <= y < grid[1]


def test_cells_form_a_four_connected_region():
    for _ in range(20):
        blot = make_blot([10, 10], [1, 1], 30, squeeze=True)
        assert _is_four_connected(_cells_of(blot))


def test_squeeze_false_returns_none_when_start_is_isolated(monkeypatch):
    grid = [3, 3]
    _force_start(monkeypatch, grid, (1, 1))
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True
    assert make_blot(grid, [1, 1], 2, permitted=permitted) is None


def test_squeeze_true_returns_single_cell_when_start_is_isolated(monkeypatch):
    grid = [3, 3]
    _force_start(monkeypatch, grid, (1, 1))
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True
    blot = make_blot(grid, [1, 1], 2, permitted=permitted, squeeze=True)
    assert _cells_of(blot) == [(1, 1)]


def test_squeeze_true_fills_only_the_permitted_component(monkeypatch):
    grid = [3, 3]
    _force_start(monkeypatch, grid, (1, 1))
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, :] = True  # the column x == 1
    blot = make_blot(grid, [1, 1], 10, permitted=permitted, squeeze=True)
    assert set(_cells_of(blot)) == {(1, 0), (1, 1), (1, 2)}


def test_permitted_mask_is_indexed_as_x_then_y(monkeypatch):
    # Mask shape is (width, height) = (4, 2). Only (3, 1) and (2, 1) are
    # permitted, so the blot must be exactly those two cells.
    grid = [4, 2]
    _force_start(monkeypatch, grid, (3, 1))
    permitted = np.zeros((4, 2), dtype=bool)
    permitted[3, 1] = True
    permitted[2, 1] = True
    blot = make_blot(grid, [1, 1], 5, permitted=permitted, squeeze=True)
    assert set(_cells_of(blot)) == {(3, 1), (2, 1)}


# ---------------------------------------------------------------------------
# make_blot: neighbour selection weights
# ---------------------------------------------------------------------------


def test_neighbour_draw_probabilities_follow_shared_edge_lengths(monkeypatch):
    # cell_dims = [3, 5]: width 3, height 5. Left/right neighbours share a
    # vertical edge of length height (5); up/down neighbours share a
    # horizontal edge of length width (3). Total weight is 16.
    grid = [5, 5]
    _force_start(monkeypatch, grid, (2, 2))
    calls = []
    _always_first(monkeypatch, calls)
    make_blot(grid, [3, 5], 2, squeeze=True)
    assert len(calls) == 1
    expected = sorted([5 / 16, 5 / 16, 3 / 16, 3 / 16])
    assert np.allclose(sorted(calls[0]), expected)


def test_neighbour_weight_accumulates_over_adjacent_blot_cells(monkeypatch):
    # With index 0 always chosen, the growth is (2,2) -> (1,2) -> (2,1) -> (3,2).
    # Before the third draw, (1,1) touches both (1,2) and (2,1), so its
    # weight is 3 + 5 = 8 and the total is 32.
    grid = [5, 5]
    _force_start(monkeypatch, grid, (2, 2))
    calls = []
    _always_first(monkeypatch, calls)
    blot = make_blot(grid, [3, 5], 4, squeeze=True)
    assert len(calls) == 3
    expected = sorted(np.array([5, 3, 5, 8, 3, 3, 5]) / 32.0)
    assert np.allclose(sorted(calls[2]), expected)
    assert set(_cells_of(blot)) == {(2, 2), (1, 2), (2, 1), (3, 2)}
