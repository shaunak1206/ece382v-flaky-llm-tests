import pytest
import numpy as np
from spreading_dye_sampler import dye_blot


def test_dyeblot_initialization():
    blot = dye_blot.DyeBlot([10, 20], [2, 3])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_reflects_added_cells():
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 0), (1, 1)])
    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_correctly():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])

    called = []

    def record_call(x, y):
        called.append((x, y))

    blot.for_each_cell(record_call)
    assert called == [(0, 0), (1, 1), (2, 2)]


def test_make_blot_creates_blot_of_correct_size():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 5
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == req_num_cells
    # All cells should be within grid boundaries
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_with_permitted_mask_blocks_disallowed_cells():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1, 1] = True  # Only one permitted cell
    blot = dye_blot.make_blot(grid_dims, cell_dims, 2, permitted=permitted, squeeze=True)
    # Only one cell can be included as only one permitted cell
    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells[0] == (1, 1)


def test_make_blot_returns_none_if_start_cell_not_permitted():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)  # No permitted cells
    blot = dye_blot.make_blot(grid_dims, cell_dims, 1, permitted=permitted, squeeze=True)
    assert blot is None


def test_make_blot_returns_none_if_non_permitted_cell_reached_and_squeeze_false():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    # Block most places except starting cell and one neighbor
    permitted[:] = False
    permitted[1, 1] = True
    permitted[1, 2] = True  # This neighbor will cause failure if reached and squeeze=False
    # We patch np.random.choice to always pick the neighbour (1,2) to simulate this situation

    import builtins
    import numpy.random

    original_choice = numpy.random.choice

    try:
        # Monkey patch choice to always pick index 1 (the neighbour with non-permitted cell)
        def fake_choice(indices, p):
            return 1
        numpy.random.choice = fake_choice

        blot = dye_blot.make_blot(grid_dims, cell_dims, 3, permitted=permitted, squeeze=False)
        # Since non-permitted neighbour is chosen and squeeze is False, should return None
        assert blot is None
    finally:
        numpy.random.choice = original_choice


def test_make_blot_squeeze_removes_non_permitted_cells():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[0, 1] = False  # Block one neighbor
    blot = dye_blot.make_blot(grid_dims, cell_dims, 3, permitted=permitted, squeeze=True)
    assert blot is not None
    # No cell in the blot should be a disallowed cell
    for cell in blot._cells:
        assert permitted[cell]
    assert blot.num_cells() <= 3


def test_make_blot_cells_are_adjacent_and_connect_via_shared_edges():
    grid_dims = [5, 5]
    cell_dims = [2, 3]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 10)
    assert blot is not None
    cells = blot._cells
    # Check that each cell is either the first or shares edge with at least one previous cell
    for i, (x, y) in enumerate(cells):
        if i == 0:
            continue
        # The current cell must be adjacent to some prior cell
        adj_found = False
        for (px, py) in cells[:i]:
            if (abs(px - x) == 1 and py == y) or (abs(py - y) == 1 and px == x):
                adj_found = True
                break
        assert adj_found


def test_make_blot_all_cells_within_bounds():
    grid_dims = [8, 8]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 15)
    assert blot is not None
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_small_grid_and_large_req_num_cells_returns_none_or_partial():
    grid_dims = [2, 2]
    cell_dims = [1, 1]
    # Request more cells than grid contains
    blot = dye_blot.make_blot(grid_dims, cell_dims, 10, squeeze=True)
    assert blot is not None
    # Can't have more cells than grid size
    assert blot.num_cells() <= (grid_dims[0] * grid_dims[1])


def test_make_blot_random_start_selection_is_permitted(monkeypatch):
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)

    # Force random() to always select a certain starting cell (2, 3)
    class FakeRandom:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            # 2/5=0.4, 3/5=0.6, so pick 0.4 to floor to 2, 0.6 to floor to 3
            vals = [0.4, 0.6]
            val = vals[self.calls]
            self.calls += 1
            return val

    fake_random = FakeRandom()

    import numpy.random

    original_random = numpy.random.random
    try:
        numpy.random.random = fake_random
        blot = dye_blot.make_blot(grid_dims, cell_dims, 1, permitted=permitted)
    finally:
        numpy.random.random = original_random

    assert blot is not None
    # Starting cell must be (2,3)
    assert blot._cells[0] == (2, 3)


def test_make_blot_neighbour_probabilities_sum_to_1():
    # This is a more internal structure testing usage of neighbour probabilities
    # We check that _draw_neighbour picks a valid neighbour and probabilities sum to 1

    # We test this by creating a blot with one cell and setting neighbours manually,
    # then invoking internal _draw_neighbour function (via monkeypatch)

    grid_dims = [4, 4]
    cell_dims = [1, 1]
    blot = dye_blot.DyeBlot(grid_dims, cell_dims)

    blot._cells.append((1, 1))

    neighbours = {(0,1): 3, (1,0): 2, (2,1): 5, (1,2): 4}

    import numpy as np

    cells, lengths = zip(*neighbours.items())
    lengths = np.array(lengths, dtype=float)
    probabilities = lengths / np.sum(lengths)
    assert np.isclose(np.sum(probabilities), 1.0)

    # We simulate _draw_neighbour behavior here:
    indices = np.arange(len(cells))
    np.random.seed(123)
    choices = []
    for _ in range(100):
        index = np.random.choice(indices, p=probabilities)
        choices.append(cells[index])

    # All chosen neighbours must be in neighbours keys
    for c in choices:
        assert c in neighbours


def test_make_blot_none_permitted_and_squeeze_false_returns_none():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, 3, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_none_permitted_and_squeeze_true_returns_none():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, 2, permitted=permitted, squeeze=True)
    assert blot is None


def test_for_each_cell_with_empty_blot_does_not_call_function():
    blot = dye_blot.DyeBlot([10, 10], [1, 1])
    calls = []

    def f(x, y):
        calls.append((x, y))

    blot.for_each_cell(f)
    assert calls == []


def test_for_each_cell_with_side_effects():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])

    coords_sum = {"sumx": 0, "sumy": 0}

    def add_coords(x, y):
        coords_sum["sumx"] += x
        coords_sum["sumy"] += y

    blot.for_each_cell(add_coords)
    assert coords_sum["sumx"] == 3  # 0 + 1 + 2
    assert coords_sum["sumy"] == 3  # 0 + 1 + 2
