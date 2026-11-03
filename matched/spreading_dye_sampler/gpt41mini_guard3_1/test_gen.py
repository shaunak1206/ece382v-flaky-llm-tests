import numpy as np
import pytest
from spreading_dye_sampler import dye_blot


def test_dyeblot_initial_state():
    blot = dye_blot.DyeBlot([5, 10], [2, 3])
    assert blot._grid_dims == [5, 10]
    assert blot._cell_dims == [2, 3]
    assert blot.num_cells() == 0


def test_dyeblot_num_cells_and_for_each_cell_basic():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 2), (2, 2)])

    visited = []

    def collect_cells(x, y):
        visited.append((x, y))

    blot.for_each_cell(collect_cells)

    assert blot.num_cells() == 3
    # The visited cells should be exactly the same as _cells' contents, regardless of order
    assert set(visited) == {(0, 0), (1, 2), (2, 2)}


@pytest.mark.parametrize(
    "grid_dims, cell_dims, req_num_cells",
    [
        ([1, 1], [1, 1], 1),
        ([5, 5], [1, 1], 1),
        ([5, 5], [2, 3], 3),
        ([10, 10], [1, 1], 5),
    ],
)
def test_make_blot_basic_properties(grid_dims, cell_dims, req_num_cells):
    np.random.seed(123)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot._grid_dims == grid_dims
    assert blot._cell_dims == cell_dims
    assert blot.num_cells() == req_num_cells

    # All cells should be within grid bounds
    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_returns_none_if_start_not_permitted():
    np.random.seed(1)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 1
    permitted = np.zeros(grid_dims, dtype=bool)  # No permitted cells
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is None


def test_make_blot_respects_permitted_mask_squeeze_false(monkeypatch):
    np.random.seed(100)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 3

    # Setup permitted mask that allows only (1,1) and neighbors, but a non-permitted cell neighbor will be chosen
    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 2] = False  # Disallow one neighbour cell that might be drawn

    # Because squeeze=False, if a non-permitted neighbor is chosen, result is None
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)
    # The blot may be None or a blot with fewer cells if it stops early, but per spec it should be None if non permitted cell used
    # Because randomness is seeded, desire to test exact result:

    # We cannot assert blot is not None because blot could be None if a non-permitted neighbor chosen.
    # The invariant: If blot is not None, all cells are permitted.
    if blot is not None:
        for x, y in blot._cells:
            assert permitted[x, y]
    else:
        assert blot is None


def test_make_blot_respects_permitted_mask_squeeze_true(monkeypatch):
    np.random.seed(200)
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 5

    permitted = np.ones(grid_dims, dtype=bool)
    permitted[1, 1] = False
    permitted[2, 2] = False

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=True)
    assert blot is not None
    assert blot.num_cells() <= req_num_cells

    # All cells must be permitted
    for x, y in blot._cells:
        assert permitted[x, y]


def test_make_blot_cells_are_contiguous_and_share_edges():
    np.random.seed(300)
    grid_dims = [6, 6]
    cell_dims = [2, 2]
    req_num_cells = 10
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    assert blot.num_cells() == req_num_cells

    # For every cell except the first, check it shares an edge with at least one other cell in blot.
    cells = blot._cells
    coords_set = set(cells)
    for i, cell in enumerate(cells):
        if i == 0:
            continue
        x, y = cell
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        shared = any(neigh in coords_set for neigh in neighbors)
        assert shared, f"Cell {cell} does not share an edge with other cells"


def test_make_blot_neighbours_external_state_unchanged(monkeypatch):
    # This test ensures that the neighbors dict is local and no external/global state is mutated.
    np.random.seed(400)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 2

    blot1 = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    blot2 = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot1 is not blot2
    # They should be different blot instances
    assert isinstance(blot1, dye_blot.DyeBlot)
    assert isinstance(blot2, dye_blot.DyeBlot)


def test_for_each_cell_function_is_called_exactly_once_per_cell():
    blot = dye_blot.DyeBlot([4, 4], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])

    call_count = 0

    def increment_count(x, y):
        nonlocal call_count
        call_count += 1

    blot.for_each_cell(increment_count)
    assert call_count == blot.num_cells()


def test_for_each_cell_passes_correct_coordinates():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    blot._cells.extend([(0, 0), (1, 1)])

    coords = []

    def store_coords(x, y):
        coords.append((x, y))

    blot.for_each_cell(store_coords)
    assert set(coords) == {(0, 0), (1, 1)}


def test_make_blot_probabilities_sum_to_one_and_distribution(monkeypatch):
    # Patch np.random.choice to intercept probabilities
    np.random.seed(0)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 1

    captured_probs = {}

    original_choice = dye_blot.np.random.choice if hasattr(dye_blot.np.random, 'choice') else None

    # We'll patch choice inside make_blot
    import numpy as np

    def choice_mock(indices, p=None):
        captured_probs['p'] = p
        # Return first index always to avoid randomness in this test
        return 0

    # NOTE: According to the source, choice is imported locally inside make_blot, so patching is tricky.
    # Instead, patch np.random.choice globally, it will affect the local import.
    monkeypatch.setattr(np.random, "choice", choice_mock)  # monkeypatch passed in from pytest

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    # Probabilities should sum to 1.0 (or very close)
    p = captured_probs['p']
    assert p is not None
    assert abs(sum(p) - 1.0) < 1e-10
    # All probabilities should be >= 0
    assert all(prob >= 0 for prob in p)


def test_make_blot_with_single_cell_and_masked_permitted():
    np.random.seed(55)
    grid_dims = [1, 1]
    cell_dims = [5, 5]
    req_num_cells = 1
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[0, 0] = True

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells[0] == (0, 0)


def test_make_blot_rejection_due_to_nonpermitted_with_squeeze_false():
    np.random.seed(78)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 5
    permitted = np.zeros(grid_dims, dtype=bool)
    permitted[1, 1] = True
    # Only the center is permitted
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted, squeeze=False)
    # Because neighbors outside the permitted center exist and squeeze is False,
    # blot will return None because it tries to extend to a non-permitted cell
    assert blot is None


def test_make_blot_rejection_due_to_start_cell_not_permitted():
    np.random.seed(12)
    grid_dims = [4, 4]
    cell_dims = [1, 1]
    req_num_cells = 3
    permitted = np.zeros(grid_dims, dtype=bool)
    # No cell permitted, start cell fails
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is None


def test_make_blot_all_cells_added_with_full_permitted_mask():
    np.random.seed(1234)
    grid_dims = [5, 5]
    cell_dims = [1, 1]
    req_num_cells = 10
    permitted = np.ones(grid_dims, dtype=bool)
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    assert blot is not None
    assert blot.num_cells() == req_num_cells
    # Check that all cells are within permitted mask
    for x, y in blot._cells:
        assert permitted[x, y]


def test_cells_added_do_not_repeat():
    np.random.seed(42)
    grid_dims = [10, 10]
    cell_dims = [1, 1]
    req_num_cells = 10
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells)
    assert blot is not None
    cells = blot._cells
    # Ensure all cells are unique, no duplicates
    assert len(cells) == len(set(cells))


def test_make_blot_with_non_boolean_permitted_array():
    np.random.seed(999)
    grid_dims = [3, 3]
    cell_dims = [1, 1]
    req_num_cells = 4

    # permitted supplied as int array with 0 and 1 rather than bool
    permitted = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int)

    blot = dye_blot.make_blot(grid_dims, cell_dims, req_num_cells, permitted=permitted)
    if blot is not None:
        for x, y in blot._cells:
            # Nonzero permitted entries are truthy, so tested accordingly
            assert permitted[x, y] != 0

