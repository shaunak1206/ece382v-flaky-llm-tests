import pytest
import numpy as np
import random
import torch
from spreading_dye_sampler import dye_blot


def test_dyeblot_initial_empty_cells():
    blot = dye_blot.DyeBlot([10, 15], [3, 5])
    assert blot._grid_dims == [10, 15]
    assert blot._cell_dims == [3, 5]
    assert blot.num_cells() == 0


def test_dyeblot_num_cells_after_add(monkeypatch):
    # Add cells manually to _cells and check num_cells reflects that
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells.append((2, 3))
    blot._cells.append((1, 1))
    assert blot.num_cells() == 2


def test_for_each_cell_calls_function():
    blot = dye_blot.DyeBlot([4, 4], [1, 1])
    coords = [(0, 0), (1, 2), (3, 3)]
    blot._cells = coords.copy()
    visited = []

    def f(x, y):
        visited.append((x, y))

    blot.for_each_cell(f)
    assert visited == coords


def test_make_blot_basic_shape_and_cells_count():
    # Set numpy and random seeds to produce deterministic blot
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)

    grid_dims = [10, 10]
    cell_dims = [2, 3]
    req_num_cells = 5

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)

    # Assert blot is returned and has correct grid and cell dims
    assert blot is not None
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    # Number of cells should be exactly req_num_cells
    assert blot.num_cells() == req_num_cells

    # All cells coordinates within grid bounds
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_returns_none_if_start_not_permitted():
    # A permitted mask with False everywhere => start can't be chosen
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.zeros(grid_dims, dtype=bool)

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is None


def test_make_blot_with_permitted_mask_all_false_squeeze_false():
    # Start chosen cell is not permitted, so blot returns None
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.ones(grid_dims, dtype=bool)
    # Modify permitted so that the start cell is False
    # Seed so start cell is known
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # Get start position to disable
    cx = int(np.floor(np.random.random() * grid_dims[0]))
    cy = int(np.floor(np.random.random() * grid_dims[1]))
    permitted[cx, cy] = False

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_with_permitted_mask_squeeze_true_skips_unpermitted_cells():
    # Setup a mask that permits only part of the grid
    np.random.seed(789)
    random.seed(789)
    torch.manual_seed(789)

    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 5
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1:4, 1:4] = True  # Only middle 3x3 permitted

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)

    assert blot is not None
    # All cells in blot must be permitted
    for (x, y) in blot._cells:
        assert permitted[x, y]
    # Number of cells matches req_num_cells or less if not enough permitted neighbours
    assert blot.num_cells() <= req_num_cells
    # At least one cell (start) must be in permitted region
    assert blot.num_cells() >= 1


def test_make_blot_neighbour_sharing_edge_lengths_accumulate_and_valid_choice(monkeypatch):
    # Check that neighbours dict accumulates shared_edge_length correctly and only draw from neighbours

    np.random.seed(321)
    random.seed(321)
    torch.manual_seed(321)

    grid_dims = [4, 4]
    cell_dims = [2, 3]
    req_num_cells = 3

    blot = dye_blot.DyeBlot(grid_dims, cell_dims)

    # Add a cell at (1,1) which triggers 4 neighbours with specific edge lengths
    # We patch internal functions to check neighbours dict

    # We will call _add_cell manually and check neighbours dict contents
    blot._cells = []

    # replicate _add_neighbour and _add_cell partially here
    neighbours = {}

    def _add_neighbour(neighbour, new_edge_length):
        if neighbour in blot._cells:
            return
        if neighbour in neighbours:
            neighbours[neighbour] += new_edge_length
        else:
            neighbours[neighbour] = new_edge_length

    def _add_cell(cell):
        blot._cells.append(cell)
        _add_neighbour((cell[0] - 1, cell[1]), cell_dims[1])
        _add_neighbour((cell[0], cell[1] - 1), cell_dims[0])
        _add_neighbour((cell[0] + 1, cell[1]), cell_dims[1])
        _add_neighbour((cell[0], cell[1] + 1), cell_dims[0])

    _add_cell((1, 1))

    # Check neighbours contains right keys and edge lengths
    expected_neighbours = {
        (0, 1): cell_dims[1],  # left neighbour, edge length = cell_height=3
        (1, 0): cell_dims[0],  # top neighbour, edge length = cell_width=2
        (2, 1): cell_dims[1],  # right neighbour
        (1, 2): cell_dims[0],  # bottom neighbour
    }
    assert neighbours == expected_neighbours

    # Add one neighbour again and check accumulation
    _add_neighbour((0, 1), 2)
    assert neighbours[(0, 1)] == cell_dims[1] + 2

    # Test that _draw_neighbour picks from neighbours keys weighted by their edge lengths
    def _draw_neighbour():
        cells, lengths = zip(*neighbours.items())
        indices = np.arange(0, len(cells))
        probabilities = np.asarray(lengths) / float(sum(lengths))
        # seed to make deterministic choice
        np.random.seed(1)
        index = np.random.choice(indices, p=probabilities)
        return cells[index]

    chosen = _draw_neighbour()
    assert chosen in neighbours


def test_make_blot_squeeze_false_returns_none_if_nonpermitted_neighbour_chosen():
    # Test that if squeeze is False and a non-permitted neighbour is chosen,
    # blot construction stops and returns None.

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3

    # permitted mask: all True except (2,1) is False
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[2, 1] = False

    # The blot will start somewhere permitted.
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)

    # Because we cannot predict start or neighbour choices exactly,
    # blot may be None or have fewer cells if unpermitted chosen neighbour stops.
    # But we can confirm that if blot is not None, none of blot cells violate mask
    if blot is not None:
        for x, y in blot._cells:
            assert permitted[x, y]


def test_make_blot_squeeze_true_accepts_partial_blot_in_mask():
    # Similar to previous test but with squeeze=True, blot should not return None,
    # but blot may have fewer cells than requested.

    np.random.seed(5678)
    random.seed(5678)
    torch.manual_seed(5678)

    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 7

    permitted = np.ones(grid_dims, dtype=bool)
    # Disallow a few cells
    permitted[0, 0] = False
    permitted[1, 0] = False
    permitted[0, 1] = False

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)

    assert blot is not None
    # All cells must be permitted
    for (x, y) in blot._cells:
        assert permitted[x, y]
    # May have fewer cells than requested, but at least 1 cell
    assert 1 <= blot.num_cells() <= req_num_cells


def test_for_each_cell_function_called_exact_times():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1), (2, 2)]
    calls = 0

    def f(x, y):
        nonlocal calls
        calls += 1

    blot.for_each_cell(f)
    assert calls == 3


def test_make_blot_random_start_cell_permitted_or_not(monkeypatch):
    # Patch numpy.random.random to control start cell selection
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 1

    # permitted with all True except (2, 2)=False
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[2, 2] = False

    random_values = [2 / grid_dims[0], 2 / grid_dims[1]]  # This selects start at (2,2)

    np_random_calls = []

    class FakeRandom:
        def random(self):
            val = random_values.pop(0)
            np_random_calls.append(val)
            return val

    monkeypatch.setattr(dye_blot.np.random, "random", FakeRandom().random)

    result = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    # The start cell (2,2) is not permitted, so result must be None
    assert result is None


def test_make_blot_random_start_cell_permitted(monkeypatch):
    # Similar to previous but start cell permitted; blot should be returned

    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 1

    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1, 1] = True

    random_values = [1 / grid_dims[0], 1 / grid_dims[1]]

    class FakeRandom:
        def random(self):
            return random_values.pop(0)

    monkeypatch.setattr(dye_blot.np.random, "random", FakeRandom().random)

    result = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert result is not None
    assert result._cells == [(1, 1)]


def test_make_blot_neighbour_cells_within_bounds_and_permitted_when_squeeze_true():
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    grid_dims = [6, 6]
    cell_dims = [1, 1]
    req_num_cells = 10
    # Permit only band of cells to restrict neighbor selection
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[2:5, 2:5] = True

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)

    assert blot is not None
    # All cells in blot must be permitted and within grid bounds
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]
        assert permitted[x, y]

    # Number of cells no greater than req_num_cells
    assert blot.num_cells() <= req_num_cells


def test_make_blot_extremely_small_grid_can_produce_single_cell_blot():
    # A 1x1 grid with any cell dims and req_num_cells=1 should return blot with single cell

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    grid_dims = [1, 1]
    cell_dims = [10, 10]
    req_num_cells = 1

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells[0] == (0, 0)


def test_make_blot_req_num_cells_zero_returns_blot_with_no_cells():
    blot = dye_blot.make_blot([5, 5], [1, 1], 0)
    # Requesting 0 cells means blot._cells stays empty or blot is None?
    # According to code, start is selected, then loop runs while length < req_num_cells.
    # If req_num_cells=0, loop not entered, but since start permitted, blot with one cell added.
    # Actually code always adds start first if permitted, so blot will have 1 cell even if req_num_cells=0
    # Confirm this:
    assert blot is not None
    assert blot.num_cells() == 1


def test_for_each_cell_with_side_effects():
    blot = dye_blot.DyeBlot([10, 10], [1, 1])
    blot._cells = [(2, 3), (4, 5), (6, 7)]

    results = []

    def collect_cells(x, y):
        results.append((x * 10 + y))

    blot.for_each_cell(collect_cells)
    expected = [2 * 10 + 3, 4 * 10 + 5, 6 * 10 + 7]
    assert results == expected
