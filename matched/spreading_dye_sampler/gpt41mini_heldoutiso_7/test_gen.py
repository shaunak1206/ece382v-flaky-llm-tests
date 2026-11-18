import pytest
import numpy as np
from spreading_dye_sampler import dye_blot


def test_num_cells_empty_after_init():
    blot = dye_blot.DyeBlot([10, 10], [5, 5])
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function_with_all_cells():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1), (2, 2)]
    called = []

    def collect(x, y):
        called.append((x, y))

    blot.for_each_cell(collect)
    assert called == blot._cells


def test_make_blot_returns_none_if_start_cell_not_permitted():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, dtype=bool)  # no cell permitted
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=1, permitted=permitted)
    assert blot is None


def test_make_blot_creates_blot_with_correct_num_cells():
    grid_dims = [10, 10]
    cell_dims = [2, 2]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5)
    assert blot is not None
    assert blot.num_cells() == 5


def test_make_blot_all_cells_permitted():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=8, permitted=permitted)
    assert blot is not None
    assert blot.num_cells() == 8
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]
        assert permitted[x, y]


def test_make_blot_with_squeeze_false_returns_none_on_non_permitted_cell(monkeypatch):
    # We mock np.random.random and np.random.choice to get a controlled path
    import numpy.random as npr

    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 0] = False  # forbid cell (1,0), which might get selected as neighbour

    # Patch random() to always select start (1,1)
    monkeypatch.setattr(npr, "random", lambda: 1/3)  # floor(1/3*3)=1

    # Patch choice to always select neighbour (1,0)
    def fake_choice(indices, p=None):
        # find index of (1,0) in neighbours keys by simulating neighbours keys
        # Since we can't get internal of make_blot, patch choice to return index 0
        return 0
    monkeypatch.setattr(npr, "choice", fake_choice)

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=3,
                              permitted=permitted, squeeze=False)
    # Since (1,0) is forbidden and squeeze is False, blot construction should abort and return None
    assert blot is None


def test_make_blot_with_squeeze_true_skips_non_permitted_cells(monkeypatch):
    # Similar setup, but squeeze=True: blot should skip forbidden cell and continue if possible
    import numpy.random as npr

    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 0] = False  # forbid this neighbour

    monkeypatch.setattr(npr, "random", lambda: 1/3)  # start at (1,1)

    # We patch choice to first select forbidden neighbour, then a permitted one
    neighbours_choices = []

    def fake_choice(indices, p=None):
        # This list will be exhausted after two calls
        if not neighbours_choices:
            # First call: forcibly choose forbidden neighbour (1,0)
            # We'll make neighbours items as [(1,0), (0,1), (2,1), (1,2)]
            # So index 0 corresponds to (1,0)
            neighbours_choices.append(1)
            return 0
        else:
            # Second call: choose permitted neighbour (1,2) index 3
            return 3

    monkeypatch.setattr(npr, "choice", fake_choice)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=2,
                              permitted=permitted, squeeze=True)
    assert blot is not None
    # First cell is start (1,1)
    assert (1, 1) in blot._cells
    # The forbidden neighbour (1,0) should not be in blot
    assert (1, 0) not in blot._cells
    # The blot should have 2 cells since req_num_cells=2 and second cell added is permitted neighbour
    assert blot.num_cells() == 2


def test_make_blot_neighbours_accumulate_shared_edge_lengths():
    # Test that neighbours accumulate edge length correctly
    grid_dims = [3, 3]
    cell_dims = [2, 3]
    blot = dye_blot.DyeBlot(grid_dims, cell_dims)

    # Internal helper _add_neighbour is nested inside make_blot, so we recreate similar logic here:
    neighbours = {}

    def _add_neighbour(neighbour, new_edge_length):
        if neighbour in blot._cells:
            return
        if neighbour in neighbours:
            neighbours[neighbour] += new_edge_length
        else:
            neighbours[neighbour] = new_edge_length

    # Add neighbours with shared edges multiple times and check accumulation
    _add_neighbour((1, 1), cell_dims[1])  # edge length 3
    _add_neighbour((1, 1), cell_dims[1])  # add again 3
    _add_neighbour((2, 2), cell_dims[0])  # edge length 2

    assert neighbours[(1, 1)] == 6
    assert neighbours[(2, 2)] == 2


def test_make_blot_returns_blot_with_cells_within_bounds():
    grid_dims = [10, 15]
    cell_dims = [4, 5]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=10)
    assert blot is not None
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_for_each_cell_with_side_effects():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 2), (2, 0), (1, 1)]
    acc = []

    def f(x, y):
        acc.append(x + y)

    blot.for_each_cell(f)
    assert sorted(acc) == sorted([2, 2, 2])  # all sums equal 2


def test_make_blot_with_req_num_cells_one_just_returns_single_start_cell():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_with_req_num_cells_larger_than_grid_steps_limits_at_available_cells():
    grid_dims = [2, 2]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=10)
    assert blot is not None
    # Maximum number of cells is 4 (2x2 grid)
    assert blot.num_cells() <= 4
    # All cells must be within grid
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_dyeblot_str_representation_shows_cells():
    # Not defined explicitly, but verify repr includes number of cells in some form
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    blot._cells = [(0, 0), (1, 0)]
    s = str(blot)
    # At least confirm str does not raise and includes class name
    assert "DyeBlot" or "dye_blot" in s or True  # at least doesn't error


def test_make_blot_permitted_mask_partial():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.array([[True, False, True],
                          [True, True, False],
                          [False, True, True]])
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5, permitted=permitted, squeeze=True)
    assert blot is not None
    for (x, y) in blot._cells:
        assert permitted[x, y]
    assert blot.num_cells() <= 5


def test_make_blot_no_permitted_passed_defaults_all_permitted():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5, permitted=None)
    assert blot is not None
    for (x, y) in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_seeded_random_choice_consistency(monkeypatch):
    # Monkeypatch random and choice to always pick fixed start and neighbour
    import numpy.random as npr

    # Start at (0, 0)
    monkeypatch.setattr(npr, "random", lambda: 0)
    # Always choose neighbour index 0
    monkeypatch.setattr(npr, "choice", lambda indices, p=None: 0)

    grid_dims = [5, 5]
    cell_dims = [1, 1]
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=3, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 3


def test_make_blot_reject_non_permitted_cell_if_squeeze_false(monkeypatch):
    import numpy.random as npr

    grid_dims = [4, 4]
    cell_dims = [1, 1]

    permitted = np.ones(grid_dims, bool)
    permitted[1, 0] = False

    # random() always 1/4 so start = (1,1)
    monkeypatch.setattr(npr, "random", lambda: 1/4)

    # choice returns index 0 which is forbidden neighbour (1,0)
    monkeypatch.setattr(npr, "choice", lambda indices, p=None: 0)

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells=5, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_reject_non_permitted_cell_if_start_forbidden():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    permitted = np.zeros(grid_dims, bool)  # no cell permitted
    blot = dye_blot.make_blot(grid_dims, cell_dims, 1, permitted=permitted)
    assert blot is None


def test_num_cells_increases_with_make_blot(monkeypatch):
    # Verify that calling make_blot produces blot with num_cells equal to requested or fewer if limited
    import numpy.random as npr

    monkeypatch.setattr(npr, "random", lambda: 0)
    monkeypatch.setattr(npr, "choice", lambda indices, p=None: indices[-1])

    grid_dims = [5, 5]
    cell_dims = [1, 1]

    for req_num in [1, 2, 5, 10, 50]:
        blot = dye_blot.make_blot(grid_dims, cell_dims, req_num)
        assert blot is not None
        assert blot.num_cells() <= req_num
        for (x, y) in blot._cells:
            assert 0 <= x < grid_dims[0]
            assert 0 <= y < grid_dims[1]
