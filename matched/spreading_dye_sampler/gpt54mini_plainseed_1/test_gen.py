import numpy as np
import pytest

import spreading_dye_sampler.dye_blot as dye_blot


def test_dye_blot_initialization_stores_dimensions_and_starts_empty():
    blot = dye_blot.DyeBlot([12, 34], [5, 6])

    assert blot._grid_dims == [12, 34]
    assert blot._cell_dims == [5, 6]
    assert blot.num_cells() == 0
    assert blot._cells == []


def test_num_cells_reflects_internal_cell_list_length():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    blot._cells.extend([(0, 0), (1, 0), (1, 1)])

    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 1), (2, 2), (1, 0)]
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(0, 1), (2, 2), (1, 0)]


def test_for_each_cell_on_empty_blot_does_not_call_function():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    calls = []

    blot.for_each_cell(lambda x, y: calls.append((x, y)))

    assert calls == []


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)
    permitted = np.zeros((3, 3), dtype=bool)

    blot = dye_blot.make_blot([3, 3], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_single_cell_without_permitted_uses_default_all_true(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)

    blot = dye_blot.make_blot([3, 4], [1, 1], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_stops_after_required_number_of_cells(monkeypatch):
    random_values = iter([0.0, 0.0])  # start at (0, 0)
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    def fake_choice(indices, p=None):
        # Only one neighbour exists in this setup, so always pick it.
        return 0

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    blot = dye_blot.make_blot([3, 3], [1, 1], 1, permitted=permitted)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_can_extend_to_adjacent_neighbour(monkeypatch):
    random_values = iter([0.0, 0.0])  # start at (0, 0)
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    def fake_choice(indices, p=None):
        # Neighbours are added in order:
        # (-1,0), (0,-1), (1,0), (0,1)
        # In-bounds permitted neighbour (1,0) is index 2.
        return 2

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    blot = dye_blot.make_blot([3, 3], [2, 1], 2, permitted=permitted)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_returns_none_when_unpermitted_neighbour_encountered_and_not_squeeze(monkeypatch):
    random_values = iter([0.0, 0.0])  # start at (0, 0)
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    def fake_choice(indices, p=None):
        # Pick the only neighbour: (1, 0), which is unpermitted.
        return 2

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False

    blot = dye_blot.make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_skips_unpermitted_neighbour_and_continues(monkeypatch):
    random_values = iter([0.0, 0.0])  # start at (0, 0)
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    # First neighbour is unpermitted, second is permitted.
    choices = iter([2, 3])

    def fake_choice(indices, p=None):
        return next(choices)

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False

    blot = dye_blot.make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (0, 1)]


def test_make_blot_chooses_start_cell_from_random_values(monkeypatch):
    # floor(0.7 * 4) == 2, floor(0.25 * 5) == 1
    random_values = iter([0.7, 0.25])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    blot = dye_blot.make_blot([4, 5], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(2, 1)]


def test_make_blot_uses_cell_dimensions_when_creating_neighbour_weights(monkeypatch):
    # Start at (1, 1) in a 3x3 grid.
    random_values = iter([0.4, 0.4])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    captured = {}

    def fake_choice(indices, p=None):
        captured["p"] = np.array(p, dtype=float)
        # Choose the neighbour above/below? The order is:
        # (-1,1), (1,0), (2,1), (1,2) etc. Here just pick a valid index.
        return 2

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    blot = dye_blot.make_blot([3, 3], [2, 5], 2, permitted=permitted)

    assert blot is not None
    assert "p" in captured
    # From (1,1), all four neighbours are in-bounds and valid in this test.
    # Weights should reflect shared edge lengths:
    # left/right -> cell_height = 5, up/down -> cell_width = 2
    assert np.allclose(captured["p"], np.array([5, 2, 5, 2], dtype=float) / 14.0)


def test_make_blot_returns_blot_object_with_expected_dimensions(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)

    blot = dye_blot.make_blot([7, 8], [3, 4], 1)

    assert isinstance(blot, dye_blot.DyeBlot)
    assert blot._grid_dims == [7, 8]
    assert blot._cell_dims == [3, 4]


def test_for_each_cell_propagates_callback_exception():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1)]

    def callback(x, y):
        raise ValueError(f"bad cell {x},{y}")

    with pytest.raises(ValueError, match="bad cell 0,0"):
        blot.for_each_cell(callback)
