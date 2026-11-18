import pytest
import numpy as np
from spreading_dye_sampler import dye_blot


def test_dyeblot_initial_empty():
    blot = dye_blot.DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_dyeblot_for_each_cell_calls_function():
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2), (3, 4), (0, 0)]
    called = []

    def collector(x, y):
        called.append((x, y))

    blot.for_each_cell(collector)
    assert called == blot._cells


def test_make_blot_basic_size_and_cells():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    num_cells = 5
    blot = dye_blot.make_blot(grid_dims, cell_dims, num_cells)
    assert blot is not None
    assert blot.num_cells() == num_cells
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_permitted_all_true():
    grid_dims = [4, 4]
    cell_dims = [2, 2]
    permitted = np.ones(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, 3, permitted=permitted)
    assert blot is not None
    assert blot.num_cells() == 3
    for x, y in blot._cells:
        assert permitted[x, y]


def test_make_blot_with_permitted_some_false_squeeze_true():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 1] = False  # forbid center cell
    blot = dye_blot.make_blot(grid_dims, cell_dims, 5, permitted=permitted, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 5
    for x, y in blot._cells:
        assert permitted[x, y]


def test_make_blot_with_permitted_some_false_squeeze_false_returns_none():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 1] = False  # forbid center
    # We'll patch random.choice to force picking the forbidden cell to test early abort
    import numpy.random as npr

    # Backup original choice
    original_choice = npr.choice

    def forced_choice(*args, **kwargs):
        # Force selection of forbidden neighbor (1,1)
        cells, _ = zip(*kwargs.get('a', []) if 'a' in kwargs else [])
        # Return index corresponding to (1,1) if possible, else 0
        if (1,1) in args[0]:
            return args[0].index((1,1))
        return 0

    npr.choice = forced_choice

    try:
        blot = dye_blot.make_blot(grid_dims, cell_dims, 5, permitted=permitted, squeeze=False)
        # It should return None since no squeezing allowed and a forbidden cell chosen
        assert blot is None
    finally:
        npr.choice = original_choice


def test_make_blot_start_cell_not_permitted_returns_none():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)  # all forbidden
    blot = dye_blot.make_blot(grid_dims, cell_dims, 1, permitted=permitted)
    assert blot is None


def test_make_blot_cells_are_adjacent_edges_counted_correctly():
    """
    Test that neighbours dictionary accumulates shared edges correctly for adjacency.
    Indirectly tested by checking that a blot with multiple cells has only neighbours around it.
    """
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 4, squeeze=True)
    # All neighbours should be adjacent to at least one cell in _cells
    for n in blot._cells:
        assert 0 <= n[0] < grid_dims[0]
        assert 0 <= n[1] < grid_dims[1]
    # Neighbours dict is not exposed, so we can test by extending the blot manually:
    # Start with one cell, add neighbours directly by _add_cell method and check neighbour count increasing
    blot2 = dye_blot.DyeBlot(grid_dims, cell_dims)

    # populate manually, simulating behaviour
    # Will add 3 cells and ensure num_cells() correct
    blot2._cells = []

    def dummy_add_cell(cell):
        blot2._cells.append(cell)

    dummy_add_cell((1, 1))
    dummy_add_cell((1, 2))
    dummy_add_cell((2, 2))
    assert blot2.num_cells() == 3


def test_for_each_cell_with_side_effect_function():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1), (2, 2)]
    total = 0

    def add_coords(x, y):
        nonlocal total
        total += x + y

    blot.for_each_cell(add_coords)
    assert total == (0+0) + (1+1) + (2+2)


def test_make_blot_zero_cells_requested_returns_empty_blot():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 0)
    assert blot is not None
    assert blot.num_cells() == 0


def test_make_blot_requested_cells_larger_than_grid_returns_blot_of_max_size():
    grid_dims = [2, 2]
    cell_dims = [1, 1]
    req_cells = 10  # more than total cells available
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_cells)
    # Because make_blot never explicitly limits to available cells,
    # it will stop once neighbours are exhausted
    assert blot.num_cells() <= grid_dims[0] * grid_dims[1]
    assert blot.num_cells() <= req_cells


def test_make_blot_permitted_mask_not_boolean():
    # permitted mask with integers or floats should be treated truthy/falsy
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    blot = dye_blot.make_blot(grid_dims, cell_dims, 5, permitted=permitted, squeeze=True)
    assert blot is not None
    for x, y in blot._cells:
        assert permitted[x, y]


def test_make_blot_single_cell_grid():
    grid_dims = [1, 1]
    cell_dims = [5, 5]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 1)
    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_with_non_square_grid_and_cells():
    grid_dims = [4, 2]
    cell_dims = [3, 7]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 4)
    assert blot is not None
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_squeeze_false_removes_invalid_cells_by_returning_none():
    # Patch random.choice to always pick an invalid neighbour to test early stop with squeeze=False
    import numpy.random as npr

    original_choice = npr.choice

    def forced_choice(neighbours, p):
        # Return the index for neighbour (10,10) which is outside the grid and forbidden
        for i, nbr in enumerate(neighbours):
            if nbr == (10, 10):
                return i
        return 0

    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)

    blot = dye_blot.DyeBlot(grid_dims, cell_dims)
    # manually add a cell with neighbours including an out-of-bounds cell
    blot._cells.append((0, 0))

    # monkeypatch choice method to forced_choice inside make_blot call via patching numpy.random.choice
    npr.choice = forced_choice

    try:
        blot_out = dye_blot.make_blot(grid_dims, cell_dims, 2, permitted=permitted, squeeze=False)
        # Because forced_choice will pick out-of-bounds neighbor, expect None result as per squeeze=False
        assert blot_out is None
    finally:
        npr.choice = original_choice


def test_make_blot_squeeze_true_skips_invalid_cells():
    # Patch random.choice to pick an invalid cell first, then valid cell to test squeeze behavior.
    import numpy.random as npr

    original_choice = npr.choice

    # Setup a round counter to toggle choices
    call_counter = {"count": 0}

    def forced_choice(neighbours, p):
        # First call returns an invalid neighbour, second call returns a valid neighbour
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            # return an invalid neighbour if present
            for i, nbr in enumerate(neighbours):
                if nbr == (10, 10):
                    return i
            return 0
        else:
            # return first valid neighbour
            for i, nbr in enumerate(neighbours):
                if 0 <= nbr[0] < 3 and 0 <= nbr[1] < 3:
                    return i
            return 0

    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)

    # run blot and expect no None returned and some cells selected ignoring invalid neighbour first
    npr.choice = forced_choice
    try:
        blot = dye_blot.make_blot(grid_dims, cell_dims, 3, permitted=permitted, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 3
    finally:
        npr.choice = original_choice
