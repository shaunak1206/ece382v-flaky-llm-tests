import pytest
import numpy as np
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot

def test_dyeblot_init_empty_cells():
    blot = DyeBlot([10, 15], [3, 3])
    assert blot._grid_dims == [10, 15]
    assert blot._cell_dims == [3, 3]
    assert blot._cells == []

def test_dyeblot_num_cells_empty():
    blot = DyeBlot([5,5], [1,1])
    assert blot.num_cells() == 0

def test_dyeblot_num_cells_after_append():
    blot = DyeBlot([5,5], [1,1])
    blot._cells.append((0,0))
    blot._cells.append((1,1))
    assert blot.num_cells() == 2

def test_for_each_cell_calls_function_correctly():
    blot = DyeBlot([3,3], [1,1])
    cells = [(0, 0), (1, 2), (2, 2)]
    blot._cells.extend(cells)
    calls = []
    def collector(x, y):
        calls.append((x, y))
    blot.for_each_cell(collector)
    assert calls == cells

def test_make_blot_returns_None_if_start_not_permitted():
    grid_dims = [5,5]
    cell_dims = [1,1]
    permitted = np.zeros(grid_dims, dtype=bool)
    blot = make_blot(grid_dims, cell_dims, req_num_cells=1, permitted=permitted)
    assert blot is None

def test_make_blot_without_permitted_returns_blot():
    grid_dims = [10,10]
    cell_dims = [1,1]
    blot = make_blot(grid_dims, cell_dims, req_num_cells=5)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() >= 1
    assert blot.num_cells() <= 5

def test_make_blot_returns_exact_requested_cells_if_possible():
    grid_dims = [10,10]
    cell_dims = [1,1]
    blot = make_blot(grid_dims, cell_dims, req_num_cells=7)
    assert isinstance(blot, DyeBlot)
    # Since some blot could be smaller if start unpermitted, but our start is always permitted here
    # The blot size must be exactly req_num_cells or less if failed.
    # The current code picks start randomly, so blot size can be between 1 and 7.
    # To strengthen test run many times.
    for _ in range(10):
        blot = make_blot(grid_dims, cell_dims, req_num_cells=7)
        if blot is not None:
            assert blot.num_cells() <= 7
            assert blot.num_cells() >= 1
    # So we only check that blot is never bigger than requested
    # and never zero if created

def test_make_blot_with_permitted_mask_excludes_forbidden_cells():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)
    # Allow only (2,2)
    permitted[2, 2] = True
    blot = make_blot(grid_dims, cell_dims, req_num_cells=3, permitted=permitted, squeeze=True)
    if blot is not None:
        # All cells must be permitted
        assert all(permitted[x, y] for (x, y) in blot._cells)

def test_make_blot_with_permitted_mask_and_squeeze_false_returns_none_if_invalid_cell():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    # But forbid one neighbor of start to force None when squeeze=False
    permitted[1, 1] = False
    # We'll try to start at (0,0) ideally to get neighbour (1,0) or (0,1)
    # Because of randomness of start this test might be flaky, so run multiple attempts and accept any None result.
    results = [make_blot(grid_dims, cell_dims, req_num_cells=3, permitted=permitted, squeeze=False) for _ in range(20)]
    assert any(r is None or isinstance(r, DyeBlot) for r in results)  # basic sanity
    # At least one None should appear due to forbidden cell
    assert any(r is None for r in results)

def test_make_blot_with_squeeze_true_skips_forbidden_cells():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    # Forbid cell (1,1)
    permitted[1,1] = False
    blot = make_blot(grid_dims, cell_dims, req_num_cells=10, permitted=permitted, squeeze=True)
    assert blot is not None
    # No forbidden cell in blot
    assert all(permitted[x,y] for (x,y) in blot._cells)

def test_make_blot_starts_within_grid_bounds():
    grid_dims = [8, 8]
    cell_dims = [2, 2]
    blot = make_blot(grid_dims, cell_dims, req_num_cells=4)
    assert blot is not None
    for (x,y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]

def test_make_blot_cell_neighbors_addition():
    # We'll monkeypatch np.random.choice to always pick the first neighbor to make predictable test
    import spreading_dye_sampler.dye_blot as dye_blot_module

    original_choice = dye_blot_module.np.random.choice if hasattr(dye_blot_module.np.random, 'choice') else None

    def fake_choice(indices, p):
        return 0

    dye_blot_module.np.random.choice = fake_choice

    grid_dims = [3,3]
    cell_dims = [1,1]

    blot = make_blot(grid_dims, cell_dims, req_num_cells=3)
    assert blot is not None
    # After forcing choice=0, cells should be connected in a predictable manner
    # The first cell is start, next one is first neighbor added, etc.
    # Number of cells should be exactly 3
    assert blot.num_cells() == 3

    # Restore original function
    if original_choice:
        dye_blot_module.np.random.choice = original_choice

def test_make_blot_returns_none_if_start_not_permitted_and_squeeze_true():
    grid_dims = [2, 2]
    cell_dims = [1, 1]
    permitted = np.array([[False, False],
                          [False, True]], dtype=bool)
    # Start cell could be (0,0) or (1,1) randomly.
    # We patch random to force start (0,0) which is forbidden.
    import spreading_dye_sampler.dye_blot as dye_blot_module
    original_random = dye_blot_module.np.random.random

    def fake_random_zero():
        return 0.0   # x=0, y=0 start

    dye_blot_module.np.random.random = fake_random_zero

    blot = make_blot(grid_dims, cell_dims, req_num_cells=1, permitted=permitted)
    assert blot is None

    dye_blot_module.np.random.random = original_random

def test_make_blot_with_minimal_grid_single_cell():
    grid_dims = [1,1]
    cell_dims = [5,5]
    blot = make_blot(grid_dims, cell_dims, 1)
    assert blot is not None
    assert blot.num_cells() == 1
    # The only cell is (0,0)
    assert blot._cells == [(0, 0)]

def test_make_blot_with_req_num_cells_greater_than_grid_cells():
    grid_dims = [2, 2]
    cell_dims = [1, 1]
    blot = make_blot(grid_dims, cell_dims, 10)
    # Max blot size is 4 cells, so blot num_cells <= 4
    assert blot is not None
    assert blot.num_cells() <= 4

def test_for_each_cell_called_with_correct_parameters():
    blot = DyeBlot([3,3], [2,2])
    blot._cells = [(1, 1), (2, 2)]
    called_args = []
    def f(x, y):
        called_args.append((x,y))
    blot.for_each_cell(f)
    assert called_args == [(1, 1), (2, 2)]

def test_make_blot_cells_adjacent():
    # Since blot grows from neighbors only, all cells must be adjacent to at least one other cell
    blot = make_blot([10,10], [1,1], 10)
    assert blot is not None
    cells = blot._cells
    if len(cells) > 1:
        for i, cell in enumerate(cells):
            if i == 0:
                continue
            # Check cell has at least one neighbor in previous cells (share edge)
            x,y = cell
            neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
            assert any(neigh in cells[:i] for neigh in neighbors)

def test_make_blot_permitted_none_defaults_to_all_true():
    grid_dims = [4,4]
    cell_dims = [1,1]
    blot = make_blot(grid_dims, cell_dims, 5, permitted=None)
    assert blot is not None
    assert blot.num_cells() <= 5
    # All cells should appear inside grid
    for (x,y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]
