import numpy as np
import pytest

import spreading_dye_sampler.dye_blot as dye_blot


def test_dyeblot_initial_state_and_attrs():
    blot = dye_blot.DyeBlot([5, 7], [2, 3])

    assert blot._grid_dims == [5, 7]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 1), (2, 2), (1, 0)]

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(0, 1), (2, 2), (1, 0)]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])

    called = []
    blot.for_each_cell(lambda x, y: called.append((x, y)))

    assert called == []


def test_make_blot_with_permitted_none_creates_single_cell_blot(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)

    blot = dye_blot.make_blot([4, 5], [2, 3], 10)

    assert isinstance(blot, dye_blot.DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_if_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)
    permitted = np.zeros((3, 3), dtype=bool)

    blot = dye_blot.make_blot([3, 3], [1, 1], 3, permitted=permitted)

    assert blot is None


def test_make_blot_returns_single_cell_when_req_num_cells_is_one(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)

    blot = dye_blot.make_blot([10, 10], [4, 5], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_adds_only_permitted_neighbor_when_masked(monkeypatch):
    random_values = iter([0.0, 0.0])  # start at (0,0), then no further random calls
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    def fake_choice(indices, p=None):
        assert list(indices) == [0, 1]
        assert p is not None
        assert np.allclose(p, [0.5, 0.5])
        return 1  # choose (0, 1), not (1, 0)

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.array(
        [
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )

    blot = dye_blot.make_blot([2, 2], [2, 3], 2, permitted=permitted, squeeze=False)

    assert blot is not None
    assert blot._cells == [(0, 0), (0, 1)]
    assert blot.num_cells() == 2


def test_make_blot_returns_none_when_non_permitted_neighbor_used_and_no_squeeze(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(dye_blot.np.random, "choice", lambda indices, p=None: 0)

    permitted = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    blot = dye_blot.make_blot([2, 2], [2, 3], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_skips_non_permitted_neighbor_and_continues(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    choice_calls = []

    def fake_choice(indices, p=None):
        choice_calls.append((list(indices), list(p)))
        return 0  # choose (1, 0), which will be masked out

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.array(
        [
            [True, False],
            [False, False],
        ],
        dtype=bool,
    )

    blot = dye_blot.make_blot([2, 2], [2, 3], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot._cells == [(0, 0)]
    assert blot.num_cells() == 1
    assert len(choice_calls) == 1


def test_make_blot_uses_grid_dims_for_starting_cell_bounds(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.999999)

    blot = dye_blot.make_blot([4, 6], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(3, 5)]


def test_make_blot_uses_cell_dimensions_in_neighbour_weighting(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    recorded = {}

    def fake_choice(indices, p=None):
        recorded["indices"] = list(indices)
        recorded["probabilities"] = list(p)
        return 0

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.ones((2, 2), dtype=bool)
    blot = dye_blot.make_blot([2, 2], [4, 1], 2, permitted=permitted, squeeze=False)

    assert blot is not None
    assert recorded["indices"] == [0, 1]
    assert np.allclose(recorded["probabilities"], [0.2, 0.8])


def test_make_blot_neighbor_is_removed_after_masked_selection(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: next(random_values))

    choice_returns = iter([0, 1])

    def fake_choice(indices, p=None):
        return next(choice_returns)

    monkeypatch.setattr(dye_blot.np.random, "choice", fake_choice)

    permitted = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    blot = dye_blot.make_blot([2, 2], [1, 1], 3, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_for_each_cell_with_mutable_callback_collects_coordinates():
    blot = dye_blot.DyeBlot([2, 2], [1, 1])
    blot._cells = [(1, 0), (0, 1)]

    coords = []

    def collect(x, y):
        coords.append((x, y))

    blot.for_each_cell(collect)

    assert coords == [(1, 0), (0, 1)]


def test_make_blot_returns_blot_object_with_expected_metadata(monkeypatch):
    monkeypatch.setattr(dye_blot.np.random, "random", lambda: 0.0)

    blot = dye_blot.make_blot([8, 9], [3, 4], 1)

    assert blot._grid_dims == [8, 9]
    assert blot._cell_dims == [3, 4]
    assert isinstance(blot._cells, list)


def test_num_cells_tracks_internal_cell_list():
    blot = dye_blot.DyeBlot([3, 3], [1, 1])
    blot._cells.extend([(0, 0), (1, 1), (2, 2)])

    assert blot.num_cells() == 3
