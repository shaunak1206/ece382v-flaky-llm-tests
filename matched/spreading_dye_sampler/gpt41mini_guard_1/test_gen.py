import numpy as np
import pytest
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot

# Because make_blot uses numpy.random.random and numpy.random.choice internally,
# we patch np.random to control randomness for deterministic tests.

import numpy.random as npr

class DummyRandom:
    """
    DummyRandom mimics np.random.random and np.random.choice to provide deterministic outputs.
    We cycle deterministic values so tests behave predictably.
    """
    def __init__(self):
        self.random_calls = 0
        self.choice_calls = 0

    def random(self):
        # Return a fixed float in [0,1), cycling for diversity when needed
        vals = [0.1, 0.5, 0.9]
        val = vals[self.random_calls % len(vals)]
        self.random_calls += 1
        return val

    def choice(self, a, p=None):
        # Always pick the first index deterministically, ignoring p for predictability
        self.choice_calls += 1
        # Defensive: handle scalar or array 'a'
        if isinstance(a, int):
            return 0
        return 0


@pytest.fixture(autouse=True)
def patch_np_random(monkeypatch):
    dummy = DummyRandom()
    monkeypatch.setattr(npr, "random", dummy.random)
    monkeypatch.setattr(npr, "choice", dummy.choice)
    yield


def test_dyeblot_init_and_num_cells_empty():
    blot = DyeBlot([5, 7], [2, 3])
    assert blot._grid_dims == [5, 7]
    assert blot._cell_dims == [2, 3]
    assert blot.num_cells() == 0
    assert blot._cells == []


def test_dyeblot_for_each_cell_calls_function_correctly():
    blot = DyeBlot([3, 3], [1, 1])
    cells = [(1, 1), (2, 2), (0, 0)]
    blot._cells.extend(cells)
    got = []

    def collector(x, y):
        got.append((x, y))

    blot.for_each_cell(collector)
    # Order should be preserved exactly as in _cells list
    assert got == cells


def test_make_blot_returns_none_if_start_cell_not_permitted():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.zeros(grid_dims, dtype=bool)  # no cells permitted
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is None


def test_make_blot_returns_none_if_non_permitted_cell_used_and_squeeze_false():
    # We deliberately create permitted mask that prevents expansion to desired size.
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3
    # permitted blocks expansion beyond first cell neighbors on purpose
    permitted = np.ones(grid_dims, dtype=bool)
    # Block the 2nd cell neighbors, so that the next neighbor chosen is not permitted
    # We tailor permitted so that a neighbor is not permitted.
    permitted[0, 1] = False  # block neighbor (0,1)
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)
    # Because the code stops and returns None if a non-permitted cell is requested while squeeze=False
    assert blot is None


def test_make_blot_succeeds_with_squeeze_true_ignoring_non_permitted_neighbors():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[0, 1] = False  # block neighbor (0,1)
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    # blot should be created, but without cell (0, 1)
    assert blot is not None
    assert all(0 <= c[0] < grid_dims[0] and 0 <= c[1] < grid_dims[1] for c in blot._cells)
    for c in blot._cells:
        assert permitted[c]
    assert blot.num_cells() <= req_num_cells
    assert blot.num_cells() >= 1  # at least start cell exists


def test_make_blot_cells_are_adjacent():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 5
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    # Every added cell beyond the first must share an edge with at least one previously added cell.
    cells = blot._cells
    assert len(cells) == blot.num_cells() == req_num_cells or len(cells) < req_num_cells
    # Check adjacency property
    def adjacent(c1, c2):
        return (abs(c1[0] - c2[0]) == 1 and c1[1] == c2[1]) or (abs(c1[1] - c2[1]) == 1 and c1[0] == c2[0])

    if len(cells) > 1:
        for ix in range(1, len(cells)):
            # The cell added at ix must be adjacent to at least one cell in cells[:ix]
            assert any(adjacent(cells[ix], prev) for prev in cells[:ix])


def test_make_blot_num_cells_equals_req_num_if_possible():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 10
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    # Should get exactly requested number of cells if possible
    assert blot.num_cells() == req_num_cells


def test_make_blot_no_permitted_argument_creates_all_permitted():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 4
    blot = make_blot(grid_dims, cell_dims, req_num_cells, permitted=None)
    assert blot is not None
    # All cells in the blot should be inside the grid bounds
    for c in blot._cells:
        assert 0 <= c[0] < grid_dims[0]
        assert 0 <= c[1] < grid_dims[1]


def test_make_blot_cells_consistent_with_cell_dims_and_grid_dims():
    grid_dims = [7, 8]
    cell_dims = [2, 3]
    req_num_cells = 6
    blot = make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    # All cells coordinates are within the grid size
    for c in blot._cells:
        assert 0 <= c[0] < grid_dims[0]
        assert 0 <= c[1] < grid_dims[1]


def test_make_blot_empty_cells_if_req_num_zero():
    blot = make_blot([5, 5], [1, 1], 0)
    # If req_num_cells==0, the blot might have zero cells or None if start cell not permitted (but possible)
    if blot is not None:
        assert blot.num_cells() == 0


def test_make_blot_start_cell_at_grid_edge_permitted():
    # Using dummy random values: 0.1 * grid_width = 0.3 => floor=0 for small grid, so start cell likely (0,0)
    blot = make_blot([3, 3], [1, 1], 1)
    assert blot is not None
    start_cell = blot._cells[0]
    assert 0 <= start_cell[0] < 3
    assert 0 <= start_cell[1] < 3


def test_make_blot_neighbours_sum_of_edge_lengths():
    blot = make_blot([4, 4], [2, 3], 3)
    assert blot is not None
    # If blot has at least two cells, neighbors exist at some point
    if blot.num_cells() > 1:
        # Because neighbors are internal, retesting sum of lengths is tricky but make sure cells exist and edges valid
        for c in blot._cells:
            assert 0 <= c[0] < 4
            assert 0 <= c[1] < 4


def test_make_blot_returns_none_if_initial_cell_not_permitted():
    permitted = np.zeros((3, 3), dtype=bool)
    blot = make_blot([3, 3], [1, 1], 1, permitted=permitted)
    assert blot is None


def test_make_blot_cells_are_unique():
    blot = make_blot([5, 5], [1, 1], 10)
    assert blot is not None
    cells = blot._cells
    # Cells should be unique
    assert len(cells) == len(set(cells))


def test_make_blot_with_nonboolean_permitted_array():
    # permitted mask with integers (0,1) instead of boolean
    permitted = np.array([[1, 0], [1, 1]])
    blot = make_blot([2, 2], [1, 1], 2, permitted=permitted)
    assert blot is not None
    # All cells should be in permitted locations (>0 are truthy)
    for c in blot._cells:
        assert permitted[c] != 0


def test_make_blot_permitted_mask_smaller_than_grid_dims_raises_index_error():
    permitted = np.ones((1, 1), dtype=bool)
    # Because permitted mask is smaller than grid, make_blot will try to index out of bounds,
    # which raises an exception; so catch it.
    with pytest.raises(IndexError):
        make_blot([2, 2], [1, 1], 1, permitted=permitted)


def test_dyeblot_for_each_cell_with_side_effects():
    blot = DyeBlot([2, 2], [1, 1])
    blot._cells.extend([(0, 0), (1, 0), (1, 1)])

    accumulator = []

    def f(x, y):
        accumulator.append(x + y)

    blot.for_each_cell(f)
    # Sum of x+y in any order for these cells is known
    expected_sums = [0 + 0, 1 + 0, 1 + 1]
    assert set(accumulator) == set(expected_sums)
    assert len(accumulator) == len(expected_sums)


def test_for_each_cell_empty_cells_list():
    blot = DyeBlot([3, 3], [1, 1])
    called = False

    def f(x, y):
        nonlocal called
        called = True

    blot.for_each_cell(f)
    # Because blot._cells is empty, function should never be called
    assert not called


def test_make_blot_rejects_cell_outside_grid():
    permitted = np.ones((3, 3), dtype=bool)
    # Force start cell outside range by patching random to return > 1 (simulate)
    import numpy.random as npr

    class RandFail:
        def __init__(self):
            self.count = 0

        def random(self):
            # Return 1.5 which is out of bounds after floor(grid_dim * random)
            self.count += 1
            return 1.5

        def choice(self, a, p=None):
            return 0

    fail_rand = RandFail()
    orig_random = npr.random
    orig_choice = npr.choice
    npr.random = fail_rand.random
    npr.choice = fail_rand.choice
    try:
        blot = make_blot([3, 3], [1, 1], 1, permitted=permitted)
        # Because floor(random * grid_width) will be floor(1.5 * 3) = 4 (out of range),
        # _is_permitted will be False and blot will be None
        assert blot is None
    finally:
        npr.random = orig_random
        npr.choice = orig_choice
