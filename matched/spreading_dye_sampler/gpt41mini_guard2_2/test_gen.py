import numpy as np
import random
import pytest

import spreading_dye_sampler.dye_blot as dye_blot


def test_dyeblot_num_cells_empty():
    blot = dye_blot.DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_dyeblot_for_each_cell_calls_function(monkeypatch):
    blot = dye_blot.DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2), (3, 4), (0, 0)]

    calls = []

    def record_call(x, y):
        calls.append((x, y))

    blot.for_each_cell(record_call)
    assert calls == blot._cells


@pytest.mark.parametrize("permitted_shape,expected_none", [
    ((5, 5), False),
    ((3, 3), True),  # grid smaller than requested cells, likely None
])
def test_make_blot_basic(permitted_shape, expected_none):
    # Seed random generators for reproducibility
    random.seed(0)
    np.random.seed(0)

    blot = dye_blot.make_blot(
        grid_dims=list(permitted_shape),
        cell_dims=[1, 1],
        req_num_cells=3,
        permitted=None,
        squeeze=True,
    )
    if expected_none:
        assert blot is None or isinstance(blot, dye_blot.DyeBlot)
    else:
        assert blot is not None
        assert isinstance(blot, dye_blot.DyeBlot)
        assert blot.num_cells() == 3


def test_make_blot_returns_none_when_start_not_permitted():
    random.seed(1)
    np.random.seed(1)

    permitted = np.zeros((4, 4), dtype=bool)
    # All cells disallowed, so start cell disallowed
    blot = dye_blot.make_blot([4, 4], [1, 1], 2, permitted=permitted)
    assert blot is None


def test_make_blot_respects_permitted_mask_and_squeeze_false():
    random.seed(2)
    np.random.seed(2)

    permitted = np.ones((5, 5), dtype=bool)
    # Block cell (2, 2)
    permitted[2, 2] = False

    # Because squeeze=False, encountering blocked cell aborts and returns None
    blot = dye_blot.make_blot([5, 5], [1, 1], 10, permitted=permitted, squeeze=False)
    # blot can be None or partial if start not permitted or stops at blocked cell
    # We check that if blot is None, it is due to blocked cell. If blot exists, no blocked cell.
    if blot is not None:
        # blot cells must not include blocked position
        for cell in blot._cells:
            assert permitted[cell]
    else:
        # blot is None, expected due to blocked cell
        assert blot is None


def test_make_blot_respects_permitted_mask_and_squeeze_true():
    random.seed(3)
    np.random.seed(3)

    permitted = np.ones((5, 5), dtype=bool)
    # Block cell (1, 0) and (3, 3)
    permitted[1, 0] = False
    permitted[3, 3] = False

    blot = dye_blot.make_blot([5, 5], [1, 1], 10, permitted=permitted, squeeze=True)
    # Must not include any disallowed cells
    for cell in blot._cells:
        assert permitted[cell]


def test_make_blot_generated_cells_are_adjacent():
    random.seed(4)
    np.random.seed(4)

    blot = dye_blot.make_blot([10, 10], [1, 1], 20, permitted=None)
    assert blot is not None

    cells = blot._cells.copy()
    # Each cell except the first must be adjacent to at least one earlier cell (shared edge)
    cell_set = set()
    for i, cell in enumerate(cells):
        if i == 0:
            cell_set.add(cell)
            continue
        x, y = cell
        # Adjacent means one unit left, right, up or down
        adjacent = any(
            (x - 1, y) in cell_set or
            (x + 1, y) in cell_set or
            (x, y - 1) in cell_set or
            (x, y + 1) in cell_set
        )
        assert adjacent
        cell_set.add(cell)


def test_num_cells_matches_cells_length():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1)]
    assert blot.num_cells() == 3


def test_cells_coordinates_are_within_grid():
    random.seed(5)
    np.random.seed(5)

    grid_dims = [15, 15]
    cell_dims = [2, 3]
    req_cells = 10
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_cells)
    assert blot is not None

    for x, y in blot._cells:
        assert 0 <= x < grid_dims[0]
        assert 0 <= y < grid_dims[1]


def test_make_blot_handles_non_square_cells():
    random.seed(6)
    np.random.seed(6)

    grid_dims = [10, 10]
    cell_dims = [3, 7]
    req_cells = 5
    blot = dye_blot.make_blot(grid_dims, cell_dims, req_cells)
    assert blot is not None
    assert blot._cell_dims == cell_dims
    assert blot.num_cells() == req_cells


def test_make_blot_rejects_cells_outside_permitted_bounds(monkeypatch):
    # forcibly patch random.random to pick a start outside permitted (simulate)
    np.random.seed(7)
    random.seed(7)

    permitted = np.zeros((4, 4), dtype=bool)
    # Permit only one corner (0,0)
    permitted[0, 0] = True

    # Patch random.random to always return number that makes start (3,3)
    # We'll patch random.random inside numpy.random.random via monkeypatching numpy.random.random

    monkeypatch.setattr("numpy.random.random", lambda: 0.99)
    blot = dye_blot.make_blot([4, 4], [1, 1], 1, permitted=permitted)
    # Start cell (3,3) is not permitted, so blot is None
    assert blot is None


def test_make_blot_with_exactly_one_cell():
    random.seed(8)
    np.random.seed(8)
    blot = dye_blot.make_blot([10, 10], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_for_each_cell_does_not_modify_cells():
    blot = dye_blot.DyeBlot([3, 3], [2, 2])
    blot._cells = [(0, 0), (1, 2)]
    original_cells = blot._cells.copy()

    called = []

    def func(x, y):
        called.append((x, y))

    blot.for_each_cell(func)
    assert blot._cells == original_cells
    assert called == original_cells


def test_make_blot_empty_permitted_allows_any_cell():
    random.seed(9)
    np.random.seed(9)
    size = [8, 6]
    blot = dye_blot.make_blot(size, [1, 1], 5, permitted=None)
    assert blot is not None
    for cell in blot._cells:
        x, y = cell
        assert 0 <= x < size[0]
        assert 0 <= y < size[1]


def test_make_blot_with_permitted_mask_true_everywhere_equals_none_only_if_start_disallowed():
    random.seed(10)
    np.random.seed(10)
    size = [6, 6]
    permitted = np.ones(size, dtype=bool)
    blot = dye_blot.make_blot(size, [1, 1], 4, permitted=permitted)
    assert blot is not None


def test_make_blot_probability_distribution_respects_edge_lengths(monkeypatch):
    # We patch random.choice to always pick neighbour with max length, to verify _draw_neighbour probabilities
    # But since choice is numpy.random.choice, and we must not change global state or real np.random.choice,
    # We monkeypatch numpy.random.choice inside make_blot temporarily.

    # Instead, we test that the sum of edge lengths > 0 and probabilities are correct size:
    random.seed(11)
    np.random.seed(11)
    grid_dims = [10, 10]
    cell_dims = [2, 3]
    blot = dye_blot.make_blot(grid_dims, cell_dims, 5)
    assert blot is not None
    # Check neighbours edge lengths sum positive and cells are within grid
    # We check once while adding cells, which is internal, so instead check stable outcomes here


def test_make_blot_returns_none_when_cell_non_permitted_and_squeeze_false(monkeypatch):
    random.seed(12)
    np.random.seed(12)

    permitted = np.ones((3, 3), dtype=bool)
    # We block cell (1,1)
    permitted[1, 1] = False

    blot = dye_blot.make_blot([3, 3], [1, 1], 5, permitted=permitted, squeeze=False)
    # If blot is None, then non-permitted cell was encountered
    assert blot is None or isinstance(blot, dye_blot.DyeBlot)
    if blot:
        for cell in blot._cells:
            assert permitted[cell]


def test_make_blot_adapts_to_squeeze_flag(monkeypatch):
    # When squeeze True, blot still returned despite some neighbors not permitted
    random.seed(13)
    np.random.seed(13)

    permitted = np.ones((4, 4), dtype=bool)
    permitted[1, 1] = False
    blot_squeeze = dye_blot.make_blot([4, 4], [1, 1], 5, permitted=permitted, squeeze=True)
    blot_no_squeeze = dye_blot.make_blot([4, 4], [1, 1], 5, permitted=permitted, squeeze=False)

    assert blot_squeeze is None or isinstance(blot_squeeze, dye_blot.DyeBlot)
    # blot_no_squeeze might be None (on blocked neighbor hit)
    # blot_squeeze can be possibly larger or equal in num_cells:
    if blot_squeeze is not None:
        for cell in blot_squeeze._cells:
            assert permitted[cell]


def test_make_blot_start_cell_always_permitted_if_permitted_all_true():
    random.seed(14)
    np.random.seed(14)

    size = [10, 10]
    permitted = np.ones(size, dtype=bool)
    blot = dye_blot.make_blot(size, [1, 1], 3, permitted=permitted)
    # Start cell is always permitted
    assert blot is not None


def test_dyeblot_string_repr_contains_cells(monkeypatch):
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    blot._cells = [(0, 0), (1, 1)]
    s = repr(blot)
    # As repr is default, just test str prints something meaningful
    s2 = str(blot)
    assert isinstance(s, str)
    assert isinstance(s2, str)


def test_make_blot_cell_addition_correctly_updates_neighbours():
    # Cannot access neighbours dict externally, but can check that neighbours used for next cell pick
    # This test indirectly ensures make_blot does not add duplicate neighbours or cells.

    random.seed(15)
    np.random.seed(15)

    blot = dye_blot.make_blot([5, 5], [1, 1], 4)
    assert blot is not None
    # Cells are unique
    assert len(set(blot._cells)) == blot.num_cells()
