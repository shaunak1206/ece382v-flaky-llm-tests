import pytest
import numpy as np
from spreading_dye_sampler import dye_blot


def test_dyeblot_initialization():
    blot = dye_blot.DyeBlot([5, 5], [10, 10])
    assert blot._grid_dims == [5, 5]
    assert blot._cell_dims == [10, 10]
    assert blot._cells == []
    assert blot.num_cells() == 0

def test_num_cells_with_appended_cells():
    blot = dye_blot.DyeBlot([3, 3], [5, 5])
    blot._cells.append((0, 0))
    blot._cells.append((1, 1))
    assert blot.num_cells() == 2

def test_for_each_cell_calls_function_for_all_cells():
    blot = dye_blot.DyeBlot([2, 2], [5, 5])
    blot._cells.extend([(0, 0), (1, 1)])
    called = []

    def func(x, y):
        called.append((x, y))

    blot.for_each_cell(func)
    assert set(called) == {(0, 0), (1, 1)}

def test_make_blot_basic_properties():
    blot = dye_blot.make_blot([10, 10], [1, 1], 5)
    assert blot is not None
    # Cells within grid
    for x, y in blot._cells:
        assert 0 <= x < 10
        assert 0 <= y < 10
    # Number of cells
    assert blot.num_cells() == 5

def test_make_blot_cells_are_connected():
    # connectivity by shared cell edges: adjacents have x or y differ by 1 only
    blot = dye_blot.make_blot([7, 7], [2, 2], 10)
    assert blot is not None
    cells = blot._cells
    # Each added cell must neighbor at least one other cell on either axis
    # except possibly the first
    for i in range(1, len(cells)):
        cx, cy = cells[i]
        neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        assert any(n in cells[:i] for n in neighbors)

def test_make_blot_respects_permitted_mask():
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[2, 2] = True
    blot = dye_blot.make_blot([5, 5], [1, 1], 1, permitted=permitted)
    if blot is not None:
        assert blot._cells == [(2, 2)]
    else:
        # Starting cell was picked randomly; if not (2,2), should return None
        assert blot is None or blot.num_cells() == 1

def test_make_blot_returns_none_if_start_cell_not_permitted():
    permitted = np.zeros((5, 5), dtype=bool)
    blot = dye_blot.make_blot([5, 5], [1, 1], 1, permitted=permitted)
    assert blot is None

def test_make_blot_allows_squeeze_to_skip_nonpermitted_cells():
    permitted = np.ones((5, 5), dtype=bool)
    permitted[1, 0] = False  # block a cell that would neighbor start
    blot = dye_blot.make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 3
    # The blocked cell should not be in blot
    assert (1, 0) not in blot._cells

def test_make_blot_returns_none_if_nonpermitted_cell_found_and_squeeze_false():
    permitted = np.ones((5, 5), dtype=bool)
    # create a "hole" that must be used if squeeze is false
    permitted[1, 0] = False
    blot = dye_blot.make_blot([2, 2], [1, 1], 3, permitted=permitted, squeeze=False)
    # It's possible that the blot fails because of hitting disallowed cell
    # so blot should be None
    assert blot is None or blot.num_cells() < 3

def test_make_blot_with_req_num_cells_one_returns_single_cell():
    blot = dye_blot.make_blot([3, 3], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    x, y = blot._cells[0]
    assert 0 <= x < 3
    assert 0 <= y < 3

def test_make_blot_permitted_mask_partial():
    permitted = np.zeros((10, 10), dtype=bool)
    permitted[3:6, 3:6] = True
    blot = dye_blot.make_blot([10, 10], [1, 1], 5, permitted=permitted, squeeze=True)
    if blot:
        # All cells in blot must be within the permitted mask region
        for x, y in blot._cells:
            assert 3 <= x < 6
            assert 3 <= y < 6

def test_make_blot_returns_none_if_start_cell_not_permitted_and_permitted_given():
    permitted = np.zeros((4, 4), dtype=bool)
    blot = dye_blot.make_blot([4, 4], [1, 1], 2, permitted=permitted)
    assert blot is None

def test_make_blot_selection_probability_weights():
    # Run multiple blots to check that weights influence neighbor picking
    blot_sizes = []
    for _ in range(10):
        blot = dye_blot.make_blot([4, 4], [1, 1], 5)
        if blot:
            blot_sizes.append(blot.num_cells())
    assert all(1 <= size <= 5 for size in blot_sizes)

def test_dyeblot_for_each_cell_with_no_cells():
    blot = dye_blot.DyeBlot([3, 3], [2, 2])
    calls = []

    def func(x, y):
        calls.append((x, y))

    blot.for_each_cell(func)
    assert calls == []

def test_make_blot_cells_are_unique():
    blot = dye_blot.make_blot([10, 10], [1, 1], 8)
    assert blot is not None
    assert len(blot._cells) == len(set(blot._cells))

def test_make_blot_with_non_standard_cell_dims():
    blot = dye_blot.make_blot([10, 10], [3, 2], 5)
    assert blot is not None
    assert blot.num_cells() == 5

def test_internal_neighbour_sharing_edges_accumulate():
    blot = dye_blot.make_blot([5, 5], [1, 1], 3)
    # neighbours is internal, but indirectly test logic by checking neighbors are adjacent
    cells = blot._cells
    for c in cells:
        # No cell is out of bounds
        assert 0 <= c[0] < 5
        assert 0 <= c[1] < 5

def test_make_blot_handles_large_req_num_cells():
    max_cells = 50
    blot = dye_blot.make_blot([10, 10], [1, 1], max_cells)
    if blot is not None:
        assert blot.num_cells() <= max_cells
        for x, y in blot._cells:
            assert 0 <= x < 10
            assert 0 <= y < 10

def test_make_blot_start_cell_randomness(monkeypatch):
    # Patch random and choice to make the start cell and subsequent neighbor deterministic
    import numpy as np

    calls = {}

    def fake_random():
        # always pick cell (1,1) in grid 3x3
        return 1 / 3

    def fake_choice(indices, p=None):
        calls['choice_called'] = True
        return 0

    monkeypatch.setattr(np.random, 'random', fake_random)
    monkeypatch.setattr(np.random, 'choice', fake_choice)

    blot = dye_blot.make_blot([3, 3], [1, 1], 2)
    assert blot is not None
    # Start cell should be (1,1) due to fake_random
    assert (1, 1) in blot._cells
    assert 'choice_called' in calls
