import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def collect_cells(blot):
    """Gather the cells of a blot through the public for_each_cell API."""
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def is_connected(cells):
    """Return True if the cells form one 4-connected region."""
    if not cells:
        return True
    cell_set = set(cells)
    start = cells[0]
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            n = (nx, ny)
            if n in cell_set and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cell_set)


def within_bounds(cell, grid_dims):
    x, y = cell
    return 0 <= x < grid_dims[0] and 0 <= y < grid_dims[1]


# ---------------------------------------------------------------------------
# DyeBlot basics
# ---------------------------------------------------------------------------

def test_new_dyeblot_has_zero_cells():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_new_dyeblot_stores_dimensions():
    blot = DyeBlot([12, 34], [3, 4])
    assert blot._grid_dims == [12, 34]
    assert blot._cell_dims == [3, 4]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = DyeBlot([10, 10], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_positionally():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(2, 5)]
    received = []

    def record(x, y):
        received.append((x, y))

    blot.for_each_cell(record)
    assert received == [(2, 5)]


def test_for_each_cell_visits_cells_in_insertion_order():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 2), (3, 4), (0, 0)]
    assert collect_cells(blot) == [(1, 2), (3, 4), (0, 0)]


def test_for_each_cell_visits_each_cell_once_and_count_matches():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1)]
    visits = []
    blot.for_each_cell(lambda x, y: visits.append((x, y)))
    assert len(visits) == blot.num_cells() == 3


# ---------------------------------------------------------------------------
# make_blot: basic construction
# ---------------------------------------------------------------------------

def test_make_blot_returns_dyeblot_instance():
    np.random.seed(0)
    blot = make_blot([10, 10], [1, 1], 5, squeeze=True)
    assert isinstance(blot, DyeBlot)


def test_make_blot_returns_requested_number_of_cells():
    np.random.seed(1)
    blot = make_blot([10, 10], [1, 1], 7, squeeze=True)
    assert blot.num_cells() == 7
    assert len(collect_cells(blot)) == 7


def test_make_blot_with_one_requested_cell_returns_only_start():
    np.random.seed(2)
    blot = make_blot([10, 10], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_cells_are_unique():
    np.random.seed(3)
    blot = make_blot([20, 20], [1, 1], 15, squeeze=True)
    cells = collect_cells(blot)
    assert len(cells) == len(set(cells))


def test_make_blot_cells_lie_within_grid():
    grid = [15, 12]
    np.random.seed(4)
    blot = make_blot(grid, [2, 2], 20, squeeze=True)
    for cell in collect_cells(blot):
        assert within_bounds(cell, grid)


def test_make_blot_cells_form_connected_region():
    np.random.seed(5)
    blot = make_blot([20, 20], [1, 1], 25, squeeze=True)
    assert is_connected(collect_cells(blot))


def test_make_blot_cell_coordinates_are_python_ints():
    np.random.seed(6)
    blot = make_blot([10, 10], [1, 1], 6, squeeze=True)
    for x, y in collect_cells(blot):
        assert isinstance(x, int)
        assert isinstance(y, int)


def test_make_blot_is_reproducible_with_same_seed():
    np.random.seed(7)
    first = collect_cells(make_blot([10, 10], [1, 1], 8, squeeze=True))
    np.random.seed(7)
    second = collect_cells(make_blot([10, 10], [1, 1], 8, squeeze=True))
    assert first == second


def test_make_blot_does_not_modify_permitted_mask():
    mask = np.ones((8, 8), dtype=bool)
    mask[3, 3] = False
    original = mask.copy()
    np.random.seed(8)
    make_blot([8, 8], [1, 1], 10, permitted=mask, squeeze=True)
    assert np.array_equal(mask, original)


# ---------------------------------------------------------------------------
# make_blot: small grids and boundary handling
# ---------------------------------------------------------------------------

def test_squeeze_true_fills_small_grid_completely():
    np.random.seed(9)
    blot = make_blot([2, 2], [1, 1], 10, squeeze=True)
    cells = collect_cells(blot)
    assert sorted(cells) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_single_cell_grid_squeeze_false_returns_none_when_growing():
    # The only neighbours of the lone cell lie outside the grid, which are
    # never permitted, so a non-squeezed growth must fail.
    np.random.seed(10)
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_single_cell_grid_squeeze_true_returns_single_cell():
    np.random.seed(11)
    blot = make_blot([1, 1], [1, 1], 5, squeeze=True)
    assert collect_cells(blot) == [(0, 0)]


def test_one_dimensional_grid_squeeze_true_covers_whole_strip():
    np.random.seed(12)
    blot = make_blot([5, 1], [1, 1], 5, squeeze=True)
    assert sorted(collect_cells(blot)) == [(x, 0) for x in range(5)]


# ---------------------------------------------------------------------------
# make_blot: permitted masks
# ---------------------------------------------------------------------------

def test_start_cell_not_permitted_returns_none_squeeze_true():
    mask = np.zeros((6, 6), dtype=bool)
    np.random.seed(13)
    assert make_blot([6, 6], [1, 1], 3, permitted=mask, squeeze=True) is None


def test_start_cell_not_permitted_returns_none_squeeze_false():
    mask = np.zeros((6, 6), dtype=bool)
    np.random.seed(14)
    assert make_blot([6, 6], [1, 1], 3, permitted=mask, squeeze=False) is None


def test_squeeze_true_never_includes_masked_cells():
    mask = np.ones((20, 20), dtype=bool)
    mask[::3, :] = False
    for seed in range(5):
        np.random.seed(seed)
        blot = make_blot([20, 20], [1, 1], 30, permitted=mask, squeeze=True)
        if blot is None:
            continue
        for x, y in collect_cells(blot):
            assert mask[x, y]


def test_squeeze_true_stays_within_isolated_region():
    # Column x=2 is masked, splitting the strip into {0,1} and {3,4}.
    mask = np.array([True, True, False, True, True]).reshape(5, 1)
    allowed_regions = (
        {(0, 0), (1, 0)},
        {(3, 0), (4, 0)},
    )
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([5, 1], [1, 1], 5, permitted=mask, squeeze=True)
        if blot is None:
            continue
        assert set(collect_cells(blot)) in allowed_regions


def test_squeeze_false_returns_none_when_masked_cell_is_reached():
    mask = np.array([True, True, False, True, True]).reshape(5, 1)
    for seed in range(10):
        np.random.seed(seed)
        assert make_blot([5, 1], [1, 1], 5, permitted=mask, squeeze=False) is None


def test_single_permitted_cell_with_squeeze_false_returns_none():
    # Whichever start cell is drawn, growth to size 2 must fail, because
    # every neighbour of the only permitted cell is masked.
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    for seed in range(10):
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 2, permitted=mask, squeeze=False) is None


def test_explicit_all_true_mask_behaves_like_default():
    mask = np.ones((4, 4), dtype=bool)
    np.random.seed(20)
    blot = make_blot([4, 4], [1, 1], 4, permitted=mask, squeeze=True)
    assert blot.num_cells() == 4
    assert is_connected(collect_cells(blot))
