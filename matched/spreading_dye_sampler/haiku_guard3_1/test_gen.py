import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _is_edge_connected(cells):
    """Return True if the set of (x, y) cells is 4-connected."""
    cell_set = set(cells)
    if not cell_set:
        return True
    start = next(iter(cell_set))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == cell_set


def _in_bounds(cell, grid_dims):
    x, y = cell
    return 0 <= x < grid_dims[0] and 0 <= y < grid_dims[1]


# ---------------------------------------------------------------------------
# DyeBlot basics
# ---------------------------------------------------------------------------

def test_dyeblot_stores_dims_and_starts_empty():
    np.random.seed(0)
    blot = DyeBlot([10, 20], [3, 4])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [3, 4]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_counts_cells_list_length():
    np.random.seed(0)
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1)]
    assert blot.num_cells() == 3


def test_for_each_cell_on_empty_blot_never_calls_function():
    np.random.seed(0)
    blot = DyeBlot([5, 5], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_then_y_in_stored_order():
    np.random.seed(0)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 1), (2, 3), (3, 0)]
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(0, 1), (2, 3), (3, 0)]


def test_for_each_cell_calls_once_per_cell():
    np.random.seed(0)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 0), (1, 1), (2, 2), (3, 3)]
    count = []
    blot.for_each_cell(lambda x, y: count.append(1))
    assert len(count) == blot.num_cells() == 4


def test_for_each_cell_propagates_exceptions():
    np.random.seed(0)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 0)]

    def boom(x, y):
        raise ValueError("stop")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


# ---------------------------------------------------------------------------
# make_blot: basic construction
# ---------------------------------------------------------------------------

def test_make_blot_size_one_returns_single_in_bounds_cell():
    np.random.seed(11)
    blot = make_blot([5, 5], [2, 2], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    cell = blot._cells[0]
    assert _in_bounds(cell, [5, 5])


def test_make_blot_returns_requested_number_of_cells_on_open_grid():
    np.random.seed(1234)
    blot = make_blot([6, 6], [1, 1], 10)
    assert blot is not None
    assert blot.num_cells() == 10


def test_make_blot_stores_grid_and_cell_dims():
    np.random.seed(5)
    blot = make_blot([8, 9], [2, 3], 4)
    assert blot is not None
    assert blot._grid_dims == [8, 9]
    assert blot._cell_dims == [2, 3]


def test_make_blot_cells_are_unique_in_bounds_and_connected_over_many_seeds():
    for seed in range(25):
        np.random.seed(seed)
        blot = make_blot([7, 5], [1, 1], 12)
        assert blot is not None
        cells = blot._cells
        assert len(cells) == len(set(cells)) == 12
        assert all(_in_bounds(c, [7, 5]) for c in cells)
        assert _is_edge_connected(cells)


def test_make_blot_cell_coordinates_are_python_ints():
    np.random.seed(3)
    blot = make_blot([6, 6], [1, 1], 8)
    assert blot is not None
    for x, y in blot._cells:
        assert isinstance(x, int)
        assert isinstance(y, int)


def test_make_blot_full_grid_request_covers_whole_grid():
    np.random.seed(42)
    blot = make_blot([3, 2], [1, 1], 6)
    assert blot is not None
    assert blot.num_cells() == 6
    assert set(blot._cells) == {(x, y) for x in range(3) for y in range(2)}


def test_make_blot_same_seed_gives_same_blot():
    np.random.seed(99)
    first = make_blot([10, 10], [1, 1], 15)
    np.random.seed(99)
    second = make_blot([10, 10], [1, 1], 15)
    assert first is not None and second is not None
    assert first._cells == second._cells


def test_make_blot_default_permitted_matches_explicit_all_true_mask():
    np.random.seed(77)
    default = make_blot([6, 4], [1, 1], 9)
    np.random.seed(77)
    explicit = make_blot([6, 4], [1, 1], 9,
                         permitted=np.ones([6, 4], dtype=bool))
    assert default is not None and explicit is not None
    assert default._cells == explicit._cells


def test_make_blot_does_not_modify_permitted_mask():
    np.random.seed(8)
    mask = np.ones([5, 5], dtype=bool)
    mask[2, :] = False
    before = mask.copy()
    make_blot([5, 5], [1, 1], 4, permitted=mask, squeeze=True)
    assert np.array_equal(mask, before)


# ---------------------------------------------------------------------------
# make_blot: exceeding the grid
# ---------------------------------------------------------------------------

def test_make_blot_oversized_request_with_squeeze_fills_whole_grid():
    np.random.seed(21)
    blot = make_blot([3, 2], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert set(blot._cells) == {(x, y) for x in range(3) for y in range(2)}
    assert blot.num_cells() == 6


def test_make_blot_oversized_request_without_squeeze_returns_none():
    np.random.seed(21)
    # The grid boundary always produces an out-of-bounds neighbour, which
    # is rejected when squeeze is False, so construction must fail.
    assert make_blot([3, 2], [1, 1], 100, squeeze=False) is None


# ---------------------------------------------------------------------------
# make_blot: permitted masks (with fixed start via monkeypatch)
# ---------------------------------------------------------------------------

def test_make_blot_blocked_start_returns_none():
    np.random.seed(5)
    mask = np.zeros([4, 4], dtype=bool)
    assert make_blot([4, 4], [1, 1], 3, permitted=mask) is None


def test_make_blot_squeeze_true_keeps_only_permitted_cells_over_many_seeds():
    mask = np.ones([5, 5], dtype=bool)
    mask[2, :] = False  # block the middle column
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 10, permitted=mask, squeeze=True)
        if blot is None:
            # Only possible if the random start landed on a blocked cell.
            continue
        assert all(mask[c] for c in blot._cells)
        assert _is_edge_connected(blot._cells)


def test_make_blot_squeeze_false_only_permitted_cells_or_none_over_many_seeds():
    mask = np.ones([5, 5], dtype=bool)
    mask[2, :] = False
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 4, permitted=mask, squeeze=False)
        if blot is not None:
            assert all(mask[c] for c in blot._cells)
            assert blot.num_cells() == 4


def test_make_blot_squeeze_true_with_single_permitted_cell_yields_it(monkeypatch):
    np.random.seed(0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # start at (0, 0)
    mask = np.zeros([3, 3], dtype=bool)
    mask[0, 0] = True
    blot = make_blot([3, 3], [1, 1], 5, permitted=mask, squeeze=True)
    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_make_blot_squeeze_false_returns_none_when_all_neighbours_blocked(monkeypatch):
    np.random.seed(0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # start at (0, 0)
    # Grid is 3 wide, 1 high; the only neighbour of (0, 0) inside the
    # grid, (1, 0), is blocked, and the others are out of bounds.
    mask = np.array([[True], [False], [True]])
    assert make_blot([3, 1], [1, 1], 3, permitted=mask, squeeze=False) is None


def test_make_blot_squeeze_true_skips_blocked_neighbour_deterministically(monkeypatch):
    np.random.seed(0)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)  # start at (0, 0)
    mask = np.array([[True], [False], [True]])
    blot = make_blot([3, 1], [1, 1], 3, permitted=mask, squeeze=True)
    assert blot is not None
    assert blot._cells == [(0, 0)]


# ---------------------------------------------------------------------------
# for_each_cell on a constructed blot
# ---------------------------------------------------------------------------

def test_for_each_cell_on_made_blot_visits_every_cell_exactly_once():
    np.random.seed(314)
    blot = make_blot([8, 8], [1, 1], 20)
    assert blot is not None
    visited = []
    blot.for_each_cell(lambda x, y: visited.append((x, y)))
    assert len(visited) == blot.num_cells() == 20
    assert set(visited) == set(blot._cells)
    assert len(set(visited)) == 20
