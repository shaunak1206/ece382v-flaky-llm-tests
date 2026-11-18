import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dye_blot_constructor_stores_dims_and_starts_empty():
    blot = DyeBlot([3, 4], [10, 20])
    assert blot._grid_dims == [3, 4]
    assert blot._cell_dims == [10, 20]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_reflects_number_of_internal_cells():
    blot = DyeBlot([2, 2], [1, 1])
    blot._cells.extend([(0, 0), (1, 0), (1, 1)])
    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_every_cell_in_order():
    blot = DyeBlot([2, 2], [1, 1])
    blot._cells = [(1, 2), (3, 4), (5, 6)]
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == [(1, 2), (3, 4), (5, 6)]


def test_for_each_cell_with_no_cells_calls_nothing():
    blot = DyeBlot([2, 2], [1, 1])
    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == []


def test_make_blot_returns_single_cell_when_req_num_cells_is_one(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.2)
    called = {"choice": 0}

    def fake_choice(*args, **kwargs):
        called["choice"] += 1
        raise AssertionError("choice should not be called for req_num_cells=1")

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([5, 5], [2, 3], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(1, 1)]
    assert called["choice"] == 0


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    permitted = np.zeros((3, 3), dtype=bool)

    blot = make_blot([3, 3], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_with_all_permitted_and_monkeypatched_choice_builds_requested_size(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    def fake_choice(indices, p=None):
        # Always choose the first available neighbor.
        return indices[0]

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_for_each_cell_iterates_over_constructed_cells(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: indices[0])

    blot = make_blot([3, 3], [1, 1], 3)

    seen = []
    blot.for_each_cell(lambda x, y: seen.append((x, y)))
    assert seen == blot._cells
    assert blot.num_cells() == 3


def test_make_blot_returns_none_when_nonpermitted_neighbor_encountered_and_squeeze_false(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    choices = iter([0])  # pick the first neighbor, which will be (1, 0)

    def fake_choice(indices, p=None):
        return indices[next(choices)]

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False

    blot = make_blot([3, 3], [1, 2], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_skips_nonpermitted_neighbor_and_continues_when_squeeze_true(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    # First pick the disallowed neighbor to the right, then the allowed one below.
    picks = iter([0, 0])

    def fake_choice(indices, p=None):
        return indices[next(picks)]

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False
    permitted[0, 1] = True

    blot = make_blot([3, 3], [1, 2], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (0, 1)]


def test_make_blot_respects_grid_boundaries_for_negative_or_out_of_range_cells(monkeypatch):
    # Start at bottom-right corner in a 2x2 grid.
    monkeypatch.setattr("numpy.random.random", lambda: 0.9999)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: indices[0])

    blot = make_blot([2, 2], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(1, 1)]


def test_make_blot_uses_cell_dimensions_to_define_weighting(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    captured = {}

    def fake_choice(indices, p=None):
        captured["indices"] = list(indices)
        captured["p"] = np.asarray(p)
        return indices[0]

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert set(captured["indices"]) == {1, 3}  # right and down from (0, 0)
    assert np.allclose(captured["p"], np.array([5.0, 2.0]) / 7.0)


def test_make_blot_weighting_changes_with_non_square_cell_dimensions(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)

    captured = {}

    def fake_choice(indices, p=None):
        captured["p"] = np.asarray(p)
        return indices[-1]

    monkeypatch.setattr("numpy.random.choice", fake_choice)

    blot = make_blot([3, 3], [7, 1], 2)

    assert blot is not None
    assert np.allclose(captured["p"], np.array([1.0, 7.0]) / 8.0)
    assert blot.num_cells() == 2


def test_make_blot_can_return_more_than_two_cells(monkeypatch):
    monkeypatch.setattr("numpy.random.random", lambda: 0.0)
    monkeypatch.setattr("numpy.random.choice", lambda indices, p=None: indices[0])

    blot = make_blot([4, 4], [1, 1], 4)

    assert blot is not None
    assert blot.num_cells() == 4
    assert blot._cells == [(0, 0), (1, 0), (2, 0), (3, 0)]
