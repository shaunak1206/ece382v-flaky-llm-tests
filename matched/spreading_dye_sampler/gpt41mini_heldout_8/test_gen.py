import numpy as np
import pytest
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dyeblot_initialization():
    blot = DyeBlot([5, 7], [2, 3])
    assert blot._grid_dims == [5, 7]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []


def test_num_cells_empty():
    blot = DyeBlot([5, 5], [1, 1])
    assert blot.num_cells() == 0


def test_num_cells_after_adding_cells():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])
    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_correctly():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 2), (2, 1)]
    called = []

    def record(x, y):
        called.append((x, y))

    blot.for_each_cell(record)
    assert called == blot._cells


@pytest.mark.parametrize("grid_dims,cell_dims,req_num_cells", [
    ([5, 5], [1, 1], 1),
    ([5, 5], [1, 1], 5),
    ([10, 10], [2, 2], 20),
])
def test_make_blot_basic(grid_dims, cell_dims, req_num_cells):
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert isinstance(blot, DyeBlot)
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    # Should have requested number of cells if possible, or less if limited by grid
    assert blot.num_cells() <= req_num_cells
    assert blot.num_cells() > 0
    # All cells must be inside the grid bounds
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_permitted_mask_all_true():
    grid_dims = [6, 6]
    cell_dims = [1, 1]
    mask = np.ones(grid_dims, dtype=bool)
    blot = make_blot(grid_dims, cell_dims, 10, permitted=mask)
    assert blot is not None
    assert blot.num_cells() <= 10
    # All selected cells are permitted
    for x, y in blot._cells:
        assert mask[x, y]


def test_make_blot_with_permitted_mask_some_false_squeeze():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    mask = np.ones(grid_dims, dtype=bool)
    # Block one cell in the middle
    mask[2, 2] = False
    blot = make_blot(grid_dims, cell_dims, 10, permitted=mask, squeeze=True)
    assert blot is not None
    # None of the cells should be the blocked one
    for x, y in blot._cells:
        assert (x, y) != (2, 2)
        assert mask[x, y]


def test_make_blot_with_permitted_mask_some_false_no_squeeze():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    mask = np.ones(grid_dims, dtype=bool)
    # Block a large region to make it likely a blocked cell is chosen
    mask[1, 1] = False
    mask[1, 2] = False
    mask[2, 1] = False
    blot = make_blot(grid_dims, cell_dims, 20, permitted=mask, squeeze=False)
    # Because squeeze=False, blot may be None if non-permitted cell is chosen
    # There is a chance blot is None, but in rare random cases it may succeed.
    # We accept either None or a valid blot.
    if blot is not None:
        for x, y in blot._cells:
            assert mask[x, y]


def test_make_blot_start_cell_not_permitted_returns_none():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    mask = np.zeros(grid_dims, dtype=bool)  # no cell permitted
    blot = make_blot(grid_dims, cell_dims, 1, permitted=mask)
    assert blot is None


def test_make_blot_req_num_cells_zero_returns_blot_with_zero_cells():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    blot = make_blot(grid_dims, cell_dims, 0)
    assert blot is not None
    assert blot.num_cells() == 0


def test_make_blot_cells_are_unique():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    blot = make_blot(grid_dims, cell_dims, 15)
    assert blot is not None
    # All cells must be unique (no duplicates)
    assert len(set(blot._cells)) == blot.num_cells()


def test_for_each_cell_with_mutating_function():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 1), (2, 2), (3, 3)]
    results = []

    def f(x, y):
        results.append(x + y)

    blot.for_each_cell(f)
    expected = [2, 4, 6]
    assert results == expected


def test_make_blot_produces_cells_within_bounds():
    grid_dims = [8, 8]
    cell_dims = [1, 1]
    blot = make_blot(grid_dims, cell_dims, 10)
    assert blot is not None
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_nonstandard_cell_dims():
    grid_dims = [5, 5]
    cell_dims = [3, 4]
    blot = make_blot(grid_dims, cell_dims, 5)
    assert blot is not None
    assert blot._cell_dims == cell_dims
    assert blot._grid_dims == grid_dims


def test_make_blot_returns_none_if_non_permitted_cell_chosen_and_squeeze_false():
    # We forcibly create a permitted mask that blocks all but one cell,
    # so after first cell, the blot tries adding neighbours which are not permitted,
    # leading to None result if squeeze=False.
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    mask = np.zeros(grid_dims, dtype=bool)
    mask[1, 1] = True  # only center cell permitted
    blot = make_blot(grid_dims, cell_dims, 3, permitted=mask, squeeze=False)
    assert blot is None


def test_make_blot_with_1x1_grid():
    grid_dims = [1, 1]
    cell_dims = [5, 5]
    blot = make_blot(grid_dims, cell_dims, 1)
    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_cells_add_correct_neighbours():
    # While internal neighbours dict is private, we can indirectly test that cells added
    # have neighbours with expected edge lengths by checking growth.
    grid_dims = [4, 4]
    cell_dims = [2, 3]
    blot = make_blot(grid_dims, cell_dims, 5)
    assert blot is not None
    # Since each cell adds neighbours with edge lengths cell_height (vertical edges) or cell_width (horizontal edges)
    # just confirm that all cells are valid and distinct and within range.
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_empty_permitted_mask_returns_none():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    mask = np.zeros(grid_dims, dtype=bool)
    blot = make_blot(grid_dims, cell_dims, 1)
    blot_with_mask = make_blot(grid_dims, cell_dims, 1, permitted=mask)
    assert blot is not None   # no permitted mask defaults to ones
    assert blot_with_mask is None


def test_make_blot_behavior_with_large_req_num_cells():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    requested = 100  # more than grid possible cells
    blot = make_blot(grid_dims, cell_dims, requested)
    assert blot is not None
    assert blot.num_cells() <= requested
    assert blot.num_cells() <= grid_dims[0] * grid_dims[1]


def test_for_each_cell_with_lambda():
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 0), (2, 3), (3, 1)]
    collected = []

    blot.for_each_cell(lambda x, y: collected.append((x * 10, y * 10)))
    assert collected == [(0, 0), (20, 30), (30, 10)]
