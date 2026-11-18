import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dye_blot_initial_state_and_num_cells():
    blot = DyeBlot([3, 4], [10, 20])
    assert blot._grid_dims == [3, 4]
    assert blot._cell_dims == [10, 20]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function_in_insertion_order():
    blot = DyeBlot([3, 4], [10, 20])
    blot._cells = [(1, 2), (0, 0), (2, 3)]
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == [(1, 2), (0, 0), (2, 3)]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = DyeBlot([3, 4], [10, 20])
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == []


def test_make_blot_single_cell_when_req_num_cells_is_one(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([3, 4], [10, 20], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    permitted = np.zeros((3, 4), dtype=bool)
    permitted[1, 1] = True

    blot = make_blot([3, 4], [10, 20], 2, permitted=permitted)

    assert blot is None


def test_make_blot_builds_horizontal_chain_with_weighted_neighbor_choice(monkeypatch):
    # Start at (1, 1). Then choose left neighbor first, then right neighbor.
    monkeypatch.setattr("numpy.random.random", lambda: 0.4)

    choices = []

    def fake_choice(indices, p=None):
        choices.append((tuple(indices), tuple(np.round(p, 8))))
        return 0

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [10, 20], 3)

    assert blot is not None
    assert blot._cells == [(1, 1), (0, 1), (2, 1)]
    assert blot.num_cells() == 3
    assert len(choices) == 2
    assert choices[0][0] == (0, 1, 2, 3)


def test_make_blot_builds_vertical_chain_with_weighted_neighbor_choice(monkeypatch):
    # Start at (1, 1). Then choose up neighbor first, then down neighbor.
    monkeypatch.setattr("numpy.random.random", lambda: 0.4)

    call_count = {"n": 0}

    def fake_choice(indices, p=None):
        call_count["n"] += 1
        return 1 if call_count["n"] == 1 else 3

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [10, 20], 3)

    assert blot is not None
    assert blot._cells == [(1, 1), (1, 0), (1, 2)]
    assert blot.num_cells() == 3


def test_make_blot_squeeze_false_returns_none_on_masked_neighbor(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.4)

    # Start at (1, 1), then first neighbor chosen is (0, 1), which is not permitted.
    def fake_choice(indices, p=None):
        return 0

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [10, 20], 3, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_true_skips_masked_neighbor_and_continues(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.4)

    # Choose left masked neighbor first, then right, then down.
    choices = [0, 1, 2]

    def fake_choice(indices, p=None):
        return choices.pop(0)

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 1] = False

    blot = make_blot([3, 3], [10, 20], 3, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot._cells == [(1, 1), (2, 1), (1, 2)]
    assert blot.num_cells() == 3


def test_make_blot_respects_grid_bounds_and_returns_none_if_start_outside_permitted(monkeypatch):
    # Use random values that map to the last cell in each dimension.
    monkeypatch.setattr("numpy.random.random", lambda: 0.999999)

    permitted = np.ones((2, 2), dtype=bool)
    permitted[1, 1] = False

    blot = make_blot([2, 2], [10, 20], 1, permitted=permitted)

    assert blot is None


def test_make_blot_with_all_permitted_false_returns_none(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    permitted = np.zeros((2, 2), dtype=bool)

    blot = make_blot([2, 2], [10, 20], 1, permitted=permitted)

    assert blot is None


def test_make_blot_does_not_exceed_required_cell_count(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.4)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([3, 3], [10, 20], 2)

    assert blot is not None
    assert blot.num_cells() == 2
    assert len(blot._cells) == 2


def test_make_blot_returns_blot_object_with_expected_dimensions(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([5, 6], [7, 8], 1)

    assert blot is not None
    assert isinstance(blot, DyeBlot)
    assert blot._grid_dims == [5, 6]
    assert blot._cell_dims == [7, 8]
    assert blot._cells == [(0, 0)]
