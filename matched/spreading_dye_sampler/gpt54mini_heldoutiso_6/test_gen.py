import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dyeblot_constructor_stores_grid_and_cell_dims_and_starts_empty():
    blot = DyeBlot([12, 34], [3, 4])

    assert blot._grid_dims == [12, 34]
    assert blot._cell_dims == [3, 4]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_reflects_current_cells_list_length():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])

    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(2, 3), (4, 5), (6, 7)]
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(2, 3), (4, 5), (6, 7)]


def test_for_each_cell_passes_two_arguments_each_time():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(9, 8)]
    received = []

    def func(x, y):
        received.append((x, y))

    blot.for_each_cell(func)

    assert received == [(9, 8)]


def test_for_each_cell_on_empty_blot_does_not_call_function():
    blot = DyeBlot([5, 5], [1, 1])
    called = []

    blot.for_each_cell(lambda x, y: called.append((x, y)))

    assert called == []


def test_make_blot_returns_single_cell_when_req_num_cells_is_one(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.1)

    blot = make_blot([10, 10], [2, 3], 1)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(1, 1)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.2)
    permitted = np.zeros((4, 4), dtype=bool)

    blot = make_blot([4, 4], [1, 1], 3, permitted=permitted)

    assert blot is None


def test_make_blot_uses_permitted_mask_for_start_cell(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.75)
    permitted = np.zeros((4, 4), dtype=bool)
    permitted[3, 3] = True

    blot = make_blot([4, 4], [1, 1], 1, permitted=permitted)

    assert blot is not None
    assert blot._cells == [(3, 3)]


def test_make_blot_returns_none_when_non_permitted_neighbour_used_and_squeeze_false(monkeypatch):
    # Start at (1, 1). Choose the left neighbour (0, 1) next, which is not permitted.
    random_values = iter([0.3, 0.3])

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [2, 5], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_skips_non_permitted_neighbour_when_squeeze_true(monkeypatch):
    # Start at (1, 1). First draw left neighbour (0, 1), which is blocked.
    # Then draw right neighbour (2, 1), which is permitted.
    random_values = iter([0.3, 0.3])
    choices = iter([0, 1])

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: next(choices))

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [2, 5], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(1, 1), (2, 1)]


def test_make_blot_stops_when_requested_cell_count_is_reached(monkeypatch):
    # Start at (1, 1), then add one neighbour and stop at req_num_cells=2.
    random_values = iter([0.3, 0.3])

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([3, 3], [4, 7], 2, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(1, 1), (0, 1)]


def test_make_blot_returns_single_cell_when_grid_is_one_by_one(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    blot = make_blot([1, 1], [9, 9], 10)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_respects_bounds_when_neighbour_out_of_grid(monkeypatch):
    # Start at (0, 0). The chosen neighbour is left of the grid, so it should
    # be rejected; with squeeze=False, construction returns None.
    random_values = iter([0.0, 0.0])

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([2, 2], [1, 1], 2, squeeze=False)

    assert blot is None


def test_make_blot_with_squeeze_true_can_continue_after_out_of_bounds_choice(monkeypatch):
    # Start at (0, 0). First neighbour chosen is out of bounds and removed.
    # Second neighbour chosen is permitted and added.
    random_values = iter([0.0, 0.0, 0.0])
    choices = iter([0, 2])

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: next(choices))

    blot = make_blot([2, 2], [3, 4], 2, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_passes_weighted_probabilities_to_numpy_choice(monkeypatch):
    # Start at (0, 0) in a 3x3 grid with cell dims (2, 5).
    # The neighbor edge lengths should be [5, 2, 5, 2] for
    # left, up, right, down respectively, so probabilities are normalized.
    random_values = iter([0.0, 0.0])
    captured = {}

    def fake_choice(indices, p=None):
        captured["indices"] = indices
        captured["p"] = np.asarray(p)
        return 2  # choose right neighbour

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [2, 5], 2, squeeze=True)

    assert blot is not None
    assert np.array_equal(captured["indices"], np.arange(0, 4))
    assert np.allclose(captured["p"], np.array([5, 2, 5, 2], dtype=float) / 14.0)
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_uses_cell_width_for_vertical_edges_and_cell_height_for_horizontal_edges(monkeypatch):
    # Start at (1, 1) to get all four neighbours in-bounds.
    # The four edge lengths should be [cell_height, cell_width, cell_height, cell_width].
    random_values = iter([0.5, 0.5])
    captured = {}

    def fake_choice(indices, p=None):
        captured["p"] = np.asarray(p)
        return 0

    monkeypatch.setattr("numpy.random.random", lambda: next(random_values))
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    _ = make_blot([3, 3], [7, 11], 2, squeeze=True)

    assert np.allclose(captured["p"], np.array([11, 7, 11, 7], dtype=float) / 36.0)
