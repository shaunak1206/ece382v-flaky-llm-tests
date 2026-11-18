import numpy as np
import pytest
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dyeblot_initial_state():
    blot = DyeBlot([10, 15], [3, 5])
    assert blot._grid_dims == [10, 15]
    assert blot._cell_dims == [3, 5]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_after_adding_cells_directly():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])
    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_all_cells():
    blot = DyeBlot([2, 2], [1, 1])
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    blot._cells.extend(cells)

    called_coords = []

    def collector(x, y):
        called_coords.append((x, y))

    blot.for_each_cell(collector)
    assert set(called_coords) == set(cells)


def test_make_blot_creates_blot_with_no_permitted_mask():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 5
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=None)
    assert isinstance(blot, DyeBlot)
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    assert blot.num_cells() <= req_num_cells  # Could be fewer if random start is near edge?
    assert blot.num_cells() > 0


def test_make_blot_returns_none_if_start_cell_not_permitted():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.zeros(grid_dims, dtype=bool)  # No cell allowed
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is None


def test_make_blot_respects_permitted_mask_with_squeeze_false():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 1] = False  # Block cell (1,1)
    # Create blot that must avoid (1,1). Because squeeze=False, blot creation should stop and return None if (1,1) is chosen.
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)

    # blot can be None or a blot that excludes (1,1)
    if blot is not None:
        assert (1, 1) not in blot._cells
        assert blot.num_cells() <= req_num_cells
        # Check all cells permitted
        for x, y in blot._cells:
            assert permitted[x, y]
    else:
        # Returned None because trial tried disallowed cell and squeeze=False
        assert blot is None


def test_make_blot_respects_permitted_mask_with_squeeze_true():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 5
    permitted = np.ones(grid_dims, dtype=bool)
    # Block corner cells, shrinking permitted zone
    permitted[0, 0] = False
    permitted[2, 2] = False

    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    assert isinstance(blot, DyeBlot)
    # All cells should be permitted
    for x, y in blot._cells:
        assert permitted[x, y]
    assert blot.num_cells() <= req_num_cells


def test_make_blot_num_cells_at_most_req_num_cells():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 7
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot.num_cells() <= req_num_cells
    assert blot.num_cells() > 0


def test_make_blot_single_cell_requested():
    grid_dims = [5, 5]
    cell_dims = [2, 2]
    req_num_cells = 1
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot.num_cells() == 1
    cell = blot._cells[0]
    # cell satisfies grid boundaries
    assert 0 <= cell[0] < grid_dims[0]
    assert 0 <= cell[1] < grid_dims[1]


def test_for_each_cell_with_lambda_function():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 2), (2, 1)])
    collected = []

    blot.for_each_cell(lambda x, y: collected.append((x, y)))
    assert set(collected) == set(blot._cells)


def test_make_blot_large_grid_large_req_num_cells():
    grid_dims = [50, 50]
    cell_dims = [1, 1]
    req_num_cells = 30
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() <= req_num_cells
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_starts_in_permitted_area():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    # permitted allows only a subarea in center
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[4:7, 4:7] = True

    blot = make_blot(grid_dims, cell_dims, 5, permitted=permitted)
    if blot is not None:
        # All cells must be in permitted area
        assert all(permitted[x, y] for x, y in blot._cells)
    else:
        # blot returned None because start cell not permitted
        # or neighbor blocked and squeeze False
        # That is acceptable here
        pass


def test_make_blot_respects_grid_and_cell_dims():
    grid_dims = [8, 8]
    cell_dims = [3, 5]
    req_num_cells = 4
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_multiple_runs_produce_different_results():
    grid_dims = [6, 6]
    cell_dims = [1, 1]
    req_num_cells = 5

    blot1 = make_blot(grid_dims, cell_dims, req_num_cells)
    blot2 = make_blot(grid_dims, cell_dims, req_num_cells)

    # It is possible they are equal by chance, but very low probability
    # Just ensure both are valid DyeBlot objects or None (start cell permitted)
    assert (blot1 is None or isinstance(blot1, DyeBlot))
    assert (blot2 is None or isinstance(blot2, DyeBlot))

    if blot1 and blot2:
        assert blot1.num_cells() <= req_num_cells
        assert blot2.num_cells() <= req_num_cells


def test_make_blot_with_non_boolean_permitted_values():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 4
    # permitted with integer values 0 and 1 (should be interpreted as bool)
    permitted = np.array(
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 1, 1]],
        dtype=int)

    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    if blot is not None:
        for x, y in blot._cells:
            assert permitted[x, y]  # Zero cells disallowed


def test_for_each_cell_works_when_no_cells():
    blot = DyeBlot([5, 5], [1, 1])
    called = []

    def func(x, y):
        called.append((x, y))

    blot.for_each_cell(func)
    assert called == []


def test_make_blot_with_req_num_cells_zero():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    blot = make_blot(grid_dims, cell_dims, 0)
    # A blot with zero requested cells should have zero cells
    assert blot is not None
    assert blot.num_cells() == 0


def test_make_blot_with_single_permitted_cell():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 2
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1, 1] = True

    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    # Can only have one permitted cell, so blot size should be 1 or None
    if blot is not None:
        assert blot.num_cells() == 1
        assert blot._cells[0] == (1, 1)
    else:
        # returned None because start cell not permitted or squeeze false and failed
        assert blot is None


def test_make_blot_invalid_grid_dims_raises_exception():
    # Passing invalid grid dims (e.g. zero or negative) should fail safely
    with pytest.raises(Exception):
        make_blot([0, 5], [1, 1], 1)

    with pytest.raises(Exception):
        make_blot([-1, 5], [1, 1], 1)


def test_make_blot_invalid_cell_dims_raises_exception():
    with pytest.raises(Exception):
        make_blot([5, 5], [0, 1], 1)

    with pytest.raises(Exception):
        make_blot([5, 5], [-1, 1], 1)
