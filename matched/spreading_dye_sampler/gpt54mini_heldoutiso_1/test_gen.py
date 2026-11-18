import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _patch_random_sequence(monkeypatch, values):
    values = iter(values)
    monkeypatch.setattr(np.random, "random", lambda: next(values))


def _patch_choice_sequence(monkeypatch, returns):
    returns = iter(returns)

    def _choice(indices, p=None):
        return next(returns)

    monkeypatch.setattr(np.random, "choice", _choice)


def test_dyeblot_init_stores_dims_and_starts_empty():
    blot = DyeBlot([10, 20], [2, 3])

    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []


def test_num_cells_returns_zero_for_new_blot():
    blot = DyeBlot([4, 4], [1, 1])

    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_does_not_call_function():
    blot = DyeBlot([4, 4], [1, 1])
    calls = []

    blot.for_each_cell(lambda x, y: calls.append((x, y)))

    assert calls == []


def test_for_each_cell_calls_function_with_each_cell_in_order():
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 1), (2, 3), (4, 5)]
    calls = []

    blot.for_each_cell(lambda x, y: calls.append((x, y)))

    assert calls == [(0, 1), (2, 3), (4, 5)]


def test_make_blot_creates_single_cell_blot_when_requested(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])

    blot = make_blot([3, 3], [1, 1], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])
    permitted = np.zeros((2, 2), dtype=bool)

    blot = make_blot([2, 2], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_can_grow_to_a_second_cell(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])
    _patch_choice_sequence(monkeypatch, [2])  # choose (1, 0)

    blot = make_blot([3, 3], [1, 1], 2)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_returns_none_when_selected_neighbor_is_disallowed_and_squeeze_false(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])
    _patch_choice_sequence(monkeypatch, [2])  # choose (1, 0)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_true_skips_disallowed_neighbor_and_continues(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])
    _patch_choice_sequence(monkeypatch, [2, 3])  # (1, 0) rejected, then (0, 1)

    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (0, 1)]


def test_make_blot_uses_cell_dimensions_in_neighbor_weighting(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])

    captured = {}

    def _choice(indices, p=None):
        captured["indices"] = list(indices)
        captured["p"] = np.asarray(p, dtype=float)
        return 2  # choose (1, 0)

    monkeypatch.setattr(np.random, "choice", _choice)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert captured["indices"] == [0, 1, 2, 3]
    assert captured["p"] == pytest.approx([5 / 14, 2 / 14, 5 / 14, 2 / 14])


def test_make_blot_respects_grid_boundaries_for_start_cell(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.999999, 0.999999])

    blot = make_blot([3, 4], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(2, 3)]


def test_for_each_cell_works_on_cells_constructed_by_make_blot(monkeypatch):
    _patch_random_sequence(monkeypatch, [0.0, 0.0])
    _patch_choice_sequence(monkeypatch, [3])

    blot = make_blot([3, 3], [1, 1], 2)
    calls = []

    blot.for_each_cell(lambda x, y: calls.append((x, y)))

    assert calls == [(0, 0), (0, 1)]
