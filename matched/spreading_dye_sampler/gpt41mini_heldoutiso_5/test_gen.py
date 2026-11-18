import pytest
import numpy as np
from spreading_dye_sampler import dye_blot

def test_dyeblot_initialization():
    blot = dye_blot.DyeBlot([10, 20], [2, 3])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []

def test_dyeblot_num_cells_empty():
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    assert blot.num_cells() == 0

def test_dyeblot_num_cells_non_empty():
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])
    assert blot.num_cells() == 3

def test_dyeblot_for_each_cell_calls_function(monkeypatch):
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 2), (2, 1)])

    called = []
    def f(x, y):
        called.append((x, y))
    blot.for_each_cell(f)
    assert called == [(0, 0), (1, 2), (2, 1)]

@pytest.mark.parametrize("grid_dims,cell_dims,req_num_cells", [
    ([5, 5], [1, 1], 1),
    ([5, 5], [2, 2], 5),
    ([10, 10], [1, 1], 10),
])
def test_make_blot_basic_correct_size(grid_dims, cell_dims, req_num_cells):
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == req_num_cells
    # All cells are within grid
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]

def test_make_blot_with_permitted_mask_all_true():
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 4
    permitted = np.ones(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is not None
    assert blot.num_cells() == req_num_cells
    for x, y in blot._cells:
        assert permitted[x, y]

def test_make_blot_with_permitted_mask_some_false_squeeze_true():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 6
    permitted = np.zeros(grid_dims, dtype=bool)
    # Permit a plus shape in center only
    permitted[2,1] = True
    permitted[2,2] = True
    permitted[2,3] = True
    permitted[1,2] = True
    permitted[3,2] = True
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    # The blot should never by bigger than permitted true cells (5 here)
    assert blot is not None
    # It must only include permitted cells
    for x, y in blot._cells:
        assert permitted[x, y]
    # As req_num_cells=6 > 5 permitted, blot cells will be <=5
    assert blot.num_cells() <= 5

def test_make_blot_with_permitted_mask_some_false_squeeze_false_none_returned():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 6
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[2,2] = True
    permitted[2,3] = True
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)
    # Because required cells > permitted cells and squeeze is False,
    # blot construction must fail and return None
    assert blot is None

def test_make_blot_start_cell_not_permitted_returns_none():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 1
    permitted = np.zeros(grid_dims, dtype=bool)  # no cell permitted at all
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    # No valid start cell, must return None
    assert blot is None

def test_make_blot_cells_are_unique():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 10
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    coords = blot._cells
    assert len(coords) == len(set(coords))

def test_make_blot_cells_are_adjacent():
    # Because add_neighbour method only adds neighbouring cells, all cells must be adjacent
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 15
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    cells = blot._cells

    def adjacent(c1, c2):
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

    # For each cell except the first, it must touch at least one previously added cell
    for i in range(1, len(cells)):
        prev = cells[:i]
        assert any(adjacent(cells[i], p) for p in prev)

def test_make_blot_with_non_boolean_permitted_coerced():
    # permitted array filled with 1 and 0 integers, should be treated as boolean
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.array([[1,0,1],
                          [0,1,0],
                          [1,0,1]])
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    assert blot is not None
    for x, y in blot._cells:
        assert permitted[x, y] == 1 or permitted[x, y] == True

def test_dyeblot_for_each_cell_no_cells():
    blot = dye_blot.DyeBlot([1, 1], [1, 1])
    called = []
    def f(x, y):
        called.append((x, y))
    blot.for_each_cell(f)
    assert called == []

def test_make_blot_permitted_partial_with_squeeze_true_at_least_one_cell():
    # If permitted allows only some cells, and squeeze True, must get at least one cell if possible
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 5
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1, 1] = True
    permitted[1, 2] = True
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    # Should not be None, since at least one permitted cell exists
    assert blot is not None
    # All cells are in permitted
    for x, y in blot._cells:
        assert permitted[x, y]
    assert blot.num_cells() <= req_num_cells

def test_make_blot_with_large_grid_and_req_num_cells():
    grid_dims = [50, 50]
    cell_dims = [1, 1]
    req_num_cells = 100
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == req_num_cells

def test_make_blot_req_num_cells_larger_than_grid_total_cells():
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 20  # larger than 9 total cells
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, squeeze=True)
    # It should return a blot with max number of cells possible (9)
    assert blot is not None
    assert blot.num_cells() <= 9

def test_make_blot_with_negative_grid_dims_fails():
    # Negative dims are nonsensical; test if code returns None or raises
    with pytest.raises(IndexError):
        # Because of negative indexing with numpy array in permitted access or similar
        blot = dye_blot.make_blot([-1, 5], [1, 1], 1)

def test_make_blot_with_zero_req_num_cells_returns_empty_blot():
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 0
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == 0
    # _cells should be empty
    assert blot._cells == []

def test_make_blot_with_one_cell_only():
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 1
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == 1
    cell = blot._cells[0]
    assert 0 <= cell[0] < grid_dims[0]
    assert 0 <= cell[1] < grid_dims[1]

def test_make_blot_probabilities_sum_to_one(monkeypatch):
    # Patch choice to a custom function that verifies probabilities sum to 1
    called_params = {}
    def fake_choice(indices, p):
        called_params['p_sum'] = sum(p)
        return indices[0]
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 3
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert abs(called_params['p_sum'] - 1.0) < 1e-10
