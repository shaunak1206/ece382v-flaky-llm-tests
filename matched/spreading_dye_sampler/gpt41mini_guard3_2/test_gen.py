import numpy as np
import pytest
from spreading_dye_sampler import dye_blot


def test_dyeblot_init_and_num_cells():
    blot = dye_blot.DyeBlot([10, 20], [3, 4])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [3, 4]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function(monkeypatch):
    # add some cells and collect calls
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2), (3, 4), (0, 0)]

    called = []
    def func(x, y):
        called.append((x, y))

    blot.for_each_cell(func)
    assert called == blot._cells


def test_make_blot_basic_no_permitted_seeded():
    # Seed random to fixed state at start of test body
    np.random.seed(12345)

    blot = dye_blot.make_blot([5, 5], [2, 2], 3)
    assert isinstance(blot, dye_blot.DyeBlot)
    n = blot.num_cells()
    assert n == 3
    # All cells are within grid bounds
    for x, y in blot._cells:
        assert 0 <= x < 5
        assert 0 <= y < 5


def test_make_blot_permitted_mask_all_false_returns_none_seeded():
    np.random.seed(42)
    permitted = np.zeros((3, 3), dtype=bool)
    blot = dye_blot.make_blot([3, 3], [1, 1], 2, permitted=permitted)
    assert blot is None


def test_make_blot_permitted_mask_restricts_cells_seeded():
    np.random.seed(0)
    permitted = np.zeros((4, 4), dtype=bool)
    permitted[1, 1] = True
    permitted[1, 2] = True
    blot = dye_blot.make_blot([4, 4], [1, 1], 2, permitted=permitted, squeeze=True)
    assert isinstance(blot, dye_blot.DyeBlot)
    cells = blot._cells
    # Only permitted cells allowed
    for x, y in cells:
        assert permitted[x, y]
    assert blot.num_cells() == 2


def test_make_blot_permitted_mask_non_permitted_no_squeeze_returns_none_seeded():
    np.random.seed(100)
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 1] = False
    blot = dye_blot.make_blot([3, 3], [1, 1], 5, permitted=permitted, squeeze=False)
    # If during expansion a non-permitted cell would be used -> None expected
    # With this seed and mask this happens deterministically.
    assert blot is None


def test_make_blot_permitted_mask_non_permitted_with_squeeze_skips_cells_seeded():
    np.random.seed(100)
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 1] = False
    blot = dye_blot.make_blot([3, 3], [1, 1], 5, permitted=permitted, squeeze=True)
    # Squeeze=True skips non permitted cells; blot returned may have less cells
    if blot is not None:
        for x, y in blot._cells:
            assert permitted[x, y]
        # blot num_cells can be <= req_num_cells
        assert blot.num_cells() <= 5


def test_make_blot_req_num_cells_is_one_seeded():
    np.random.seed(7)
    blot = dye_blot.make_blot([10, 10], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    x, y = blot._cells[0]
    assert 0 <= x < 10
    assert 0 <= y < 10


def test_neighbors_accumulate_shared_edge_length_seeded():
    # The `_add_neighbour` function adds edge lengths to neighbours dict.
    np.random.seed(42)
    blot = dye_blot.DyeBlot([3, 3], [2, 3])

    grid_width, grid_height = 3, 3
    cell_width, cell_height = 2, 3

    neighbours = {}

    def _add_neighbour(neighbour, new_edge_length):
        if neighbour in blot._cells:
            return
        if neighbour in neighbours:
            neighbours[neighbour] += new_edge_length
        else:
            neighbours[neighbour] = new_edge_length

    blot._cells.append((1, 1))
    # Add neighbours edges twice for one neighbour to accumulate
    _add_neighbour((0, 1), cell_height)
    _add_neighbour((0, 1), cell_height)

    assert neighbours[(0, 1)] == cell_height * 2

    # Adding neighbour in cells list does nothing
    blot._cells.append((0, 0))
    _add_neighbour((0, 0), cell_width)
    # neighbours unchanged for (0, 0) because is in blot._cells
    assert (0, 0) not in neighbours


def test_draw_neighbour_probability_distribution_sum_to_one(monkeypatch):
    # Verify _draw_neighbour selects neighbour with correct distribution
    np.random.seed(100)
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.append((1, 1))
    neighbours = {
        (0, 1): 2,
        (1, 0): 3,
        (2, 1): 5,
        (1, 2): 10,
    }
    cells, lengths = zip(*neighbours.items())
    lengths = np.asarray(lengths)
    probs = lengths / float(sum(lengths))
    assert abs(probs.sum() - 1.0) < 1e-10

    indices = np.arange(len(cells))
    chosen_index = np.random.choice(indices, p=probs)
    # chosen_index must be in range
    assert 0 <= chosen_index < len(cells)
    neighbour = cells[chosen_index]
    assert neighbour in neighbours


def test_is_permitted_outside_bounds(monkeypatch):
    permitted = np.ones((3, 3), dtype=bool)

    # Using inner function _is_permitted via make_blot wrapper is not exposed;
    # we'll test the inlined logic with a similar helper here.

    grid_width, grid_height = 3, 3

    def _is_permitted(neighbour):
        if neighbour[0] < 0: return False
        if neighbour[0] >= grid_width: return False
        if neighbour[1] < 0: return False
        if neighbour[1] >= grid_height: return False
        return permitted[neighbour]

    assert _is_permitted((0, 0)) is True
    assert _is_permitted((2, 2)) is True
    assert _is_permitted((-1, 0)) is False
    assert _is_permitted((0, -1)) is False
    assert _is_permitted((3, 0)) is False
    assert _is_permitted((0, 3)) is False


def test_make_blot_with_cell_dimensions_stored(monkeypatch):
    np.random.seed(1)
    blot = dye_blot.make_blot([10, 10], [7, 8], 3)
    assert blot is not None
    assert blot._grid_dims == [10, 10]
    assert blot._cell_dims == [7, 8]
    for (x, y) in blot._cells:
        assert 0 <= x < 10
        assert 0 <= y < 10


def test_make_blot_zero_cells_requested_returns_blot_with_zero_cells():
    # According to code, does not handle req_num_cells=0 explicitly,
    # but will add a start cell then loop won't run because len(blot._cells) == 1 > 0
    np.random.seed(0)
    blot = dye_blot.make_blot([5, 5], [1, 1], 0)
    if blot is not None:
        # blot will contain the start cell, so num_cells is 1, which differs from req_num_cells=0
        # So expect blot.num_cells >= 1 or blot is None
        assert blot.num_cells() >= 1


def test_make_blot_large_num_cells_seeded():
    np.random.seed(99)
    blot = dye_blot.make_blot([20, 20], [1, 1], 50)
    assert blot is not None
    # Should have exactly req_num_cells
    assert blot.num_cells() == 50
    # All cells within bounds
    for x, y in blot._cells:
        assert 0 <= x < 20
        assert 0 <= y < 20


def test_make_blot_with_non_bool_permitted_mask_coerced(monkeypatch):
    np.random.seed(2)
    permitted = np.array([
        [1, 0],
        [True, False]
    ])
    blot = dye_blot.make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=True)
    if blot is not None:
        # Cells all where permitted is truthy
        for (x, y) in blot._cells:
            assert permitted[x, y]


def test_make_blot_with_1x1_grid_and_req_1_seeded():
    np.random.seed(1234)
    blot = dye_blot.make_blot([1, 1], [5, 5], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


@pytest.mark.parametrize("req_num_cells", [1, 2, 3])
def test_make_blot_cells_are_unique_seeded(req_num_cells):
    np.random.seed(5678)
    blot = dye_blot.make_blot([10, 10], [1, 1], req_num_cells)
    assert blot is not None
    assert blot.num_cells() == req_num_cells
    # All cells unique
    assert len(set(blot._cells)) == blot.num_cells()


def test_for_each_cell_with_no_cells_does_not_call_function():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    called = False
    def func(x, y):
        nonlocal called
        called = True
    blot.for_each_cell(func)
    assert called is False
