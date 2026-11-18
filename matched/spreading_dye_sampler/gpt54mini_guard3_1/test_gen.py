import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dyeblot_initial_state_and_num_cells():
    blot = DyeBlot([3, 4], [10, 20])
    assert blot._grid_dims == [3, 4]
    assert blot._cell_dims == [10, 20]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_for_each_cell_visits_cells_in_insertion_order():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(1, 2), (0, 0), (2, 1)]
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == [(1, 2), (0, 0), (2, 1)]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = DyeBlot([2, 2], [1, 1])
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == []


def test_make_blot_single_cell_selected_when_start_is_permitted(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.2)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([4, 5], [10, 20], req_num_cells=1)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 1)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True

    blot = make_blot([3, 3], [1, 1], req_num_cells=1, permitted=permitted)
    assert blot is None


def test_make_blot_respects_permitted_mask_for_start_and_growth(monkeypatch):
    # Start at (0, 0), then choose the only rightward neighbor.
    draws = iter([0.0, 0.0])

    def fake_random():
        return next(draws)

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 2)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = True

    blot = make_blot([3, 3], [2, 5], req_num_cells=2, permitted=permitted)
    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]
    assert blot.num_cells() == 2


def test_make_blot_returns_none_when_nonpermitted_neighbor_is_drawn_and_squeeze_false(monkeypatch):
    draws = iter([0.0, 0.0])

    def fake_random():
        return next(draws)

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 2)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False

    blot = make_blot([3, 3], [2, 5], req_num_cells=2, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_with_squeeze_true_skips_nonpermitted_neighbor(monkeypatch):
    draws = iter([0.0, 0.0])

    def fake_random():
        return next(draws)

    # First draw right neighbor, then draw down neighbor after the forbidden one is removed.
    choice_calls = iter([2, 1])

    def fake_choice(indices, p=None):
        return next(choice_calls)

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False
    permitted[0, 1] = True

    blot = make_blot([3, 3], [2, 5], req_num_cells=2, permitted=permitted, squeeze=True)
    assert blot is not None
    assert blot._cells == [(0, 0), (0, 1)]
    assert blot.num_cells() == 2


def test_make_blot_uses_cell_dimensions_for_neighbor_weights(monkeypatch):
    # Start at (1, 1) in a 3x3 grid, then choose the left neighbor.
    draws = iter([0.4, 0.4])

    def fake_random():
        return next(draws)

    captured = {}

    def fake_choice(indices, p=None):
        captured["indices"] = list(indices)
        captured["p"] = np.array(p)
        return 0  # left neighbor is first in insertion order here

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [2, 6], req_num_cells=2)
    assert blot is not None
    assert blot._cells[0] == (1, 1)
    assert blot._cells[1] == (0, 1)

    # Neighbor order after starting at (1,1): left, up, right, down
    assert captured["indices"] == [0, 1, 2, 3]
    assert pytest.approx(captured["p"].tolist()) == [6 / 16, 2 / 16, 6 / 16, 2 / 16]


def test_make_blot_stops_when_reaching_requested_size(monkeypatch):
    draws = iter([0.0, 0.0])

    def fake_random():
        return next(draws)

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 2)

    blot = make_blot([4, 4], [1, 1], req_num_cells=1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_returns_blot_object_with_correct_metadata(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.25)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([8, 6], [3, 7], req_num_cells=1)

    assert isinstance(blot, DyeBlot)
    assert blot._grid_dims == [8, 6]
    assert blot._cell_dims == [3, 7]
    assert blot.num_cells() == 1


def test_make_blot_can_construct_from_large_requested_size_when_space_allows(monkeypatch):
    # Build a small deterministic chain: (0,0) -> (1,0) -> (1,1)
    draws = iter([0.0, 0.0])

    def fake_random():
        return next(draws)

    choice_calls = iter([2, 1])

    def fake_choice(indices, p=None):
        return next(choice_calls)

    monkeypatch.setattr("numpy.random.random", fake_random)
    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.ones((2, 2), dtype=bool)
    blot = make_blot([2, 2], [1, 1], req_num_cells=3, permitted=permitted)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0), (1, 1)]
    assert blot.num_cells() == 3


def test_for_each_cell_callback_receives_integer_coordinates():
    blot = DyeBlot([2, 2], [1, 1])
    blot._cells = [(0, 1), (1, 0)]
    seen_types = []

    def capture(x, y):
        seen_types.append((type(x), type(y)))

    blot.for_each_cell(capture)
    assert seen_types == [(int, int), (int, int)]


def test_make_blot_defaults_to_all_permitted_when_mask_is_none(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: 0)

    blot = make_blot([2, 2], [1, 1], req_num_cells=1, permitted=None)
    assert blot is not None
    assert blot._cells == [(0, 0)]
