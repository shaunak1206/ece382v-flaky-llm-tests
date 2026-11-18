import numpy as np
import pytest

import spreading_dye_sampler.dye_blot as dye_blot


def _patch_random(monkeypatch, values):
    values = iter(values)

    def fake_random():
        return next(values)

    monkeypatch.setattr(dye_blot.np.random, "random", fake_random)


def _patch_choice(monkeypatch, chooser):
    def fake_choice(indices, p=None):
        return chooser(indices, p)

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)


def test_dyeblot_constructor_stores_dimensions_and_starts_empty():
    blot = dye_blot.DyeBlot([12, 8], [3, 2])

    assert blot._grid_dims == [12, 8]
    assert blot._cell_dims == [3, 2]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_reflects_current_cell_list_length():
    blot = dye_blot.DyeBlot([4, 4], [1, 1])
    blot._cells.extend([(0, 0), (1, 0), (1, 1)])

    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = dye_blot.DyeBlot([4, 4], [1, 1])
    blot._cells = [(2, 1), (0, 0), (3, 3)]

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(2, 1), (0, 0), (3, 3)]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == []


def test_make_blot_with_no_permitted_mask_can_create_single_cell(monkeypatch):
    _patch_random(monkeypatch, [0.2, 0.7])  # start at (0, 1)

    blot = dye_blot.make_blot([3, 3], [1, 1], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 1)]


def test_make_blot_returns_none_when_start_cell_is_masked_out(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)
    permitted = np.zeros((2, 2), dtype=bool)

    blot = dye_blot.make_blot([2, 2], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_respects_permitted_mask_for_valid_start(monkeypatch):
    _patch_random(monkeypatch, [0.49, 0.49])  # start at (0, 0)
    permitted = np.zeros((2, 2), dtype=bool)
    permitted[0, 0] = True

    blot = dye_blot.make_blot([2, 2], [1, 1], 1, permitted=permitted)

    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_when_drawn_neighbor_is_forbidden_and_squeeze_false(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)

    # Force the only neighbor to be the right cell (1, 0).
    _patch_choice(monkeypatch, lambda indices, p=None: 0)

    permitted = np.ones((2, 2), dtype=bool)
    permitted[1, 0] = False

    blot = dye_blot.make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_true_skips_forbidden_neighbor_and_continues(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)

    # First choose forbidden (1, 0), then choose permitted (0, 1).
    choices = iter([0, 0])

    def chooser(indices, p=None):
        return next(choices)

    _patch_choice(monkeypatch, chooser)

    permitted = np.ones((2, 2), dtype=bool)
    permitted[1, 0] = False

    blot = dye_blot.make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells[0] == (0, 0)
    assert (0, 1) in blot._cells


def test_make_blot_can_extend_horizontally_and_uses_requested_size(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)

    # Prefer the right neighbor on the first draw.
    def chooser(indices, p=None):
        return 2  # (1, 0) after adding left, up, right, down

    _patch_choice(monkeypatch, chooser)

    blot = dye_blot.make_blot([3, 3], [1, 1], 2)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_can_extend_vertically_when_cell_dims_make_that_neighbor_more_likely(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)

    seen_probabilities = []

    def chooser(indices, p=None):
        seen_probabilities.append(tuple(np.round(p, 6)))
        # For a 2x2 grid, choose the down neighbor (index 3) if possible.
        return 3

    _patch_choice(monkeypatch, chooser)

    blot = dye_blot.make_blot([2, 2], [5, 1], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (0, 1)]
    assert len(seen_probabilities) == 1
    # neighbor weights are [cell_height, cell_width, cell_height, cell_width] = [1, 5, 1, 5]
    assert seen_probabilities[0] == pytest.approx((1 / 12, 5 / 12, 1 / 12, 5 / 12))


def test_make_blot_stops_when_required_number_of_cells_is_reached(monkeypatch):
    _patch_random(monkeypatch, [0.1, 0.1])  # start at (0, 0)

    # Any valid choice; we only need one extension.
    _patch_choice(monkeypatch, lambda indices, p=None: 2)

    blot = dye_blot.make_blot([3, 3], [1, 1], 2)

    assert blot is not None
    assert blot.num_cells() == 2


def test_make_blot_handles_boundary_cells_without_leaving_grid(monkeypatch):
    _patch_random(monkeypatch, [0.99, 0.99])  # start at max valid index in 2x2 -> (1,1)

    # Prefer left neighbor first.
    _patch_choice(monkeypatch, lambda indices, p=None: 0)

    blot = dye_blot.make_blot([2, 2], [1, 1], 2)

    assert blot is not None
    assert blot._cells[0] == (1, 1)
    assert blot._cells[1] in {(0, 1), (1, 0)}
    for x, y in blot._cells:
        assert 0 <= x < 2
        assert 0 <= y < 2


def test_make_blot_returns_blot_object_with_correct_dimensions(monkeypatch):
    _patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)

    blot = dye_blot.make_blot([5, 4], [7, 9], 1)

    assert isinstance(blot, dye_blot.DyeBlot)
    assert blot._grid_dims == [5, 4]
    assert blot._cell_dims == [7, 9]


def test_for_each_cell_with_mutating_callback_sees_cells_as_stored():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1)]

    seen = []

    def callback(x, y):
        seen.append((x, y))
        blot._cells.append((2, 2))  # mutate during iteration; should not affect current loop

    blot.for_each_cell(callback)

    assert seen == [(0, 0), (1, 1)]
    assert (2, 2) in blot._cells
