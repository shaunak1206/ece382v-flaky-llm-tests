import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _cells_of(blot):
    """Collect the cells of a blot via the public for_each_cell API."""
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _is_connected(cells):
    """Check 4-connectivity of a collection of (x, y) cells."""
    remaining = set(cells)
    if not remaining:
        return True
    stack = [next(iter(remaining))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen or cell not in remaining:
            continue
        seen.add(cell)
        x, y = cell
        stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
    return seen == remaining


# ---------------------------------------------------------------------------
# DyeBlot.num_cells / DyeBlot.for_each_cell
# ---------------------------------------------------------------------------

def test_new_dyeblot_has_zero_cells():
    blot = DyeBlot([10, 10], [2, 2])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([10, 10], [2, 2])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_visits_cells_in_stored_order():
    blot = DyeBlot([10, 10], [2, 2])
    blot._cells = [(1, 2), (3, 4), (0, 0)]
    visited = []
    blot.for_each_cell(lambda x, y: visited.append((x, y)))
    assert visited == [(1, 2), (3, 4), (0, 0)]


def test_for_each_cell_passes_exactly_two_positional_args():
    blot = DyeBlot([10, 10], [2, 2])
    blot._cells = [(5, 6)]
    received = []

    def record(*args):
        received.append(args)

    blot.for_each_cell(record)
    assert received == [(5, 6)]


def test_for_each_cell_count_matches_num_cells():
    blot = DyeBlot([10, 10], [2, 2])
    blot._cells = [(0, 0), (1, 0), (1, 1)]
    visited = []
    blot.for_each_cell(lambda x, y: visited.append(1))
    assert len(visited) == blot.num_cells() == 3


def test_for_each_cell_propagates_exceptions():
    blot = DyeBlot([10, 10], [2, 2])
    blot._cells = [(0, 0)]

    def boom(x, y):
        raise ValueError("stop")

    with pytest.raises(ValueError, match="stop"):
        blot.for_each_cell(boom)


# ---------------------------------------------------------------------------
# make_blot
# ---------------------------------------------------------------------------

def test_make_blot_returns_dyeblot_instance():
    np.random.seed(0)
    blot = make_blot([10, 10], [1, 1], 5)
    assert isinstance(blot, DyeBlot)


def test_make_blot_single_cell_request_gives_one_cell():
    np.random.seed(0)
    blot = make_blot([10, 10], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_open_grid_reaches_exact_requested_size():
    np.random.seed(1)
    blot = make_blot([10, 10], [1, 1], 20, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 20


def test_make_blot_cells_are_unique():
    np.random.seed(2)
    blot = make_blot([10, 10], [1, 1], 30, squeeze=True)
    cells = _cells_of(blot)
    assert len(cells) == len(set(cells))


def test_make_blot_cells_lie_within_grid():
    np.random.seed(3)
    width, height = 8, 5
    blot = make_blot([width, height], [1, 1], 25, squeeze=True)
    for x, y in _cells_of(blot):
        assert 0 <= x < width
        assert 0 <= y < height


def test_make_blot_is_four_connected():
    np.random.seed(4)
    blot = make_blot([12, 12], [1, 1], 40, squeeze=True)
    assert _is_connected(_cells_of(blot))


def test_make_blot_same_seed_gives_same_blot():
    np.random.seed(42)
    first = _cells_of(make_blot([10, 10], [1, 1], 15, squeeze=True))
    np.random.seed(42)
    second = _cells_of(make_blot([10, 10], [1, 1], 15, squeeze=True))
    assert first == second


def test_make_blot_request_larger_than_grid_fills_whole_grid():
    np.random.seed(5)
    blot = make_blot([3, 3], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert set(_cells_of(blot)) == {(x, y) for x in range(3) for y in range(3)}


def test_make_blot_1x1_grid_request_one_returns_single_cell():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 1)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_1x1_grid_request_two_without_squeeze_returns_none():
    # Every possible neighbour of the only cell lies outside the grid,
    # so without squeeze the construction must fail.
    for seed in range(5):
        np.random.seed(seed)
        assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_1x1_grid_request_two_with_squeeze_returns_single_cell():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 2, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_strip_grid_covers_all_cells():
    np.random.seed(6)
    blot = make_blot([3, 1], [1, 1], 3, squeeze=True)
    assert blot is not None
    assert set(_cells_of(blot)) == {(0, 0), (1, 0), (2, 0)}


def test_make_blot_all_blocked_returns_none():
    permitted = np.zeros((3, 3), dtype=bool)
    for seed in range(5):
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True) is None


def test_make_blot_mask_respected_with_squeeze():
    permitted = np.zeros((6, 6), dtype=bool)
    permitted[:3, :] = True  # only cells with x < 3 are permitted
    results = []
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 8, permitted=permitted, squeeze=True)
        if blot is not None:
            results.append(blot)
    assert results, "expected at least one successful construction"
    for blot in results:
        cells = _cells_of(blot)
        assert blot.num_cells() == 8
        assert all(x < 3 for x, _ in cells)


def test_make_blot_single_permitted_cell_with_squeeze():
    permitted = np.zeros((5, 1), dtype=bool)
    permitted[2, 0] = True
    successes = []
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([5, 1], [1, 1], 2, permitted=permitted, squeeze=True)
        if blot is not None:
            successes.append(_cells_of(blot))
    assert successes, "expected the start to land on the permitted cell at least once"
    for cells in successes:
        assert cells == [(2, 0)]


def test_make_blot_single_permitted_cell_without_squeeze_never_grows():
    permitted = np.zeros((5, 1), dtype=bool)
    permitted[2, 0] = True
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([5, 1], [1, 1], 2, permitted=permitted, squeeze=False)
        # Any growth attempt hits a non-permitted or out-of-grid cell.
        assert blot is None
