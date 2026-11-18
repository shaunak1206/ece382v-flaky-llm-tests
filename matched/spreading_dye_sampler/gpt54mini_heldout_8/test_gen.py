import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dyeblot_initialization_stores_dims_and_starts_empty():
    blot = DyeBlot([12, 34], [5, 6])

    assert blot._grid_dims == [12, 34]
    assert blot._cell_dims == [5, 6]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_tracks_appended_cells():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells.append((0, 0))
    blot._cells.append((1, 0))
    blot._cells.append((1, 1))

    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(2, 1), (0, 0), (1, 2)]

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(2, 1), (0, 0), (1, 2)]


def test_for_each_cell_with_no_cells_does_not_call_function():
    blot = DyeBlot([3, 3], [1, 1])

    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))

    assert calls == []


def test_make_blot_single_cell_without_permitted_uses_start_cell(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.2)
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    blot = make_blot([5, 5], [2, 3], 1)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(1, 1)]


def test_make_blot_returns_none_if_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.9)

    permitted = np.zeros((4, 4), dtype=bool)
    blot = make_blot([4, 4], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_extends_to_requested_number_of_cells(monkeypatch):
    # Start at (0, 0)
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))

    # First pick right neighbor, then down neighbor.
    choices = iter([2, 1])
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: next(choices))

    blot = make_blot([3, 3], [2, 5], 3)

    assert blot is not None
    assert blot.num_cells() == 3
    assert blot._cells == [(0, 0), (1, 0), (1, 1)]


def test_make_blot_grows_with_weighted_neighbor_selection(monkeypatch):
    # Start at center so all four neighbors are available.
    random_values = iter([0.5, 0.5])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))

    captured = {}

    def fake_choice(indices, p=None):
        captured["indices"] = list(indices)
        captured["probabilities"] = list(p)
        return 0

    monkeypatch.setattr(np.random, "choice", fake_choice)

    blot = make_blot([3, 3], [2, 7], 2)

    assert blot is not None
    assert blot._cells[0] == (1, 1)
    assert set(captured["indices"]) == {0, 1, 2, 3}
    assert captured["probabilities"] == pytest.approx([7 / 18, 2 / 18, 7 / 18, 2 / 18])


def test_make_blot_returns_none_when_nonpermitted_neighbour_and_squeeze_false(monkeypatch):
    # Start at (1, 1), then try to add left neighbor (0, 1), which is masked out.
    random_values = iter([0.5, 0.5])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_skips_nonpermitted_neighbour_when_squeeze_true(monkeypatch):
    # Start at (1, 1), then choose a masked neighbor, then another valid one.
    random_values = iter([0.5, 0.5])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))

    choices = iter([0, 1])  # left neighbor masked, then top neighbor allowed
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: next(choices))

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot._cells == [(1, 1), (1, 0)]


def test_make_blot_stops_when_requested_size_reached_even_if_neighbours_remain(monkeypatch):
    random_values = iter([0.5, 0.5])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 2)

    blot = make_blot([4, 4], [1, 1], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(2, 2)]


def test_make_blot_with_permitted_mask_can_select_only_truthy_cells(monkeypatch):
    # Start at (0, 0); only (1, 0) is permitted as a neighbor.
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))

    def fake_choice(indices, p=None):
        # In the order added by _add_cell, right neighbor is index 2.
        return 2

    monkeypatch.setattr(np.random, "choice", fake_choice)

    permitted = np.zeros((2, 2), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = True

    blot = make_blot([2, 2], [1, 1], 2, permitted=permitted)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_result_supports_for_each_cell_after_construction(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 2)

    blot = make_blot([3, 3], [1, 1], 2)

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(0, 0), (1, 0)]


def test_make_blot_does_not_modify_permitted_array(monkeypatch):
    random_values = iter([0.5, 0.5])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    permitted = np.ones((3, 3), dtype=bool)
    permitted_copy = permitted.copy()

    _ = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True)

    assert np.array_equal(permitted, permitted_copy)
