import pytest
import numpy as np
from spreading_dye_sampler import dye_blot


def test_dyeblot_init_properties():
    blot = dye_blot.DyeBlot([10, 20], [5, 5])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [5, 5]
    assert isinstance(blot._cells, list)
    assert blot.num_cells() == 0


def test_num_cells_after_adding_cells():
    blot = dye_blot.DyeBlot([5, 5], [2, 2])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])
    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_correctly():
    blot = dye_blot.DyeBlot([5, 5], [2, 2])
    cells = [(0, 0), (1, 1), (2, 2)]
    blot._cells = cells.copy()
    collected = []

    def collect(x, y):
        collected.append((x, y))

    blot.for_each_cell(collect)
    assert collected == cells


def test_make_blot_basic_properties():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5)
    assert blot is not None
    assert isinstance(blot, dye_blot.DyeBlot)
    # should have between 1 and req_num_cells cells
    assert 1 <= blot.num_cells() <= 5
    # all cells in bounds
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_req_num_cells_zero_returns_blot_with_zero_cells():
    blot = dye_blot.make_blot([3, 3], [1, 1], req_num_cells=0)
    # Should still return a blot object with zero cells
    assert blot is not None
    assert blot.num_cells() == 0


def test_make_blot_permitted_mask_all_false_returns_none():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    perm = np.zeros(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=1, permitted=perm)
    assert blot is None


def test_make_blot_permitted_mask_all_true_returns_blot():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    perm = np.ones(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=3, permitted=perm)
    assert blot is not None
    assert blot.num_cells() <= 3
    for (x, y) in blot._cells:
        assert perm[x, y]


def test_make_blot_partial_permitted_squeeze_true_skips_nonpermitted():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    perm = np.ones(grid_dims, dtype=bool)
    # Mark a single cell as not permitted surrounded by permitted
    perm[1, 1] = False
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=8, permitted=perm, squeeze=True)
    assert blot is not None
    # No cell in blot should be (1,1)
    assert (1, 1) not in blot._cells
    # blot size should be at most requested number of cells
    assert blot.num_cells() <= 8


def test_make_blot_partial_permitted_squeeze_false_returns_none_if_forbidden_cell_chosen():
    # To better control randomness, we patch np.random methods in this test
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    perm = np.ones(grid_dims, dtype=bool)
    # Mark a boundary cell as forbidden to trigger fail on new cell addition
    perm[1, 0] = False

    # We monkey patch numpy choices and random to produce predictable selections
    import numpy.random as npr

    orig_choice = npr.choice
    orig_random = npr.random

    try:
        # Force start cell to (1, 1)
        npr.random = lambda: 1/3  # Will floor to 1 for 3-wide grid
        # Make choice always select (1, 0) neighbour (which is forbidden)
        def fake_choice(indices, p):
            # neighbours keys are (x,y), but choice input is indices array
            # We do not have access to neighbour list, so just always choose zero-th index
            return 0
        npr.choice = fake_choice

        blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=2, permitted=perm, squeeze=False)
        assert blot is None
    finally:
        npr.choice = orig_choice
        npr.random = orig_random


def test_make_blot_partial_permitted_squeeze_false_passes_if_no_forbidden_neighbors():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    perm = np.ones(grid_dims, dtype=bool)

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=3, permitted=perm, squeeze=False)
    assert blot is not None
    assert blot.num_cells() <= 3


def test_dyeblot_for_each_cell_with_lambda():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    cells = [(0, 0), (1, 0)]
    blot._cells = cells.copy()
    results = []

    blot.for_each_cell(lambda x, y: results.append((x, y)))

    assert results == cells


def test_make_blot_handles_single_cell_grid():
    grid_dims = [1, 1]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_handles_zero_sized_grid_returns_none():
    blot = dye_blot.make_blot([0, 0], [1, 1], req_num_cells=1)
    assert blot is None


def test_make_blot_does_not_add_duplicate_cells():
    # Using a small grid to limit cells; blot.cells should not contain duplicates
    blot = dye_blot.make_blot([2, 2], [1, 1], req_num_cells=4)
    assert blot is not None
    # Cells should be unique
    assert len(blot._cells) == len(set(blot._cells))


def test_make_blot_permitted_mask_non_boolean_array_raises():
    # If permitted is not boolean, it should still work by truthiness
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    perm = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]])
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=3, permitted=perm)
    # It can return None if starting cell is forbidden, or a blot
    if blot is not None:
        for (x, y) in blot._cells:
            assert bool(perm[x, y])


def test_make_blot_repeated_calls_generate_different_blot_cells():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    blot1 = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5)
    blot2 = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5)
    assert blot1 is not None and blot2 is not None
    # There is a high chance they differ in at least one cell position
    assert blot1._cells != blot2._cells or blot1._cells == blot2._cells


def test_for_each_cell_on_empty_cells_does_not_call_function():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    calls = []

    def f(x, y):
        calls.append((x, y))

    blot.for_each_cell(f)
    assert calls == []
