import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


class SequenceRandom:
    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self):
        if self._i >= len(self._values):
            raise AssertionError("random() called more times than expected")
        value = self._values[self._i]
        self._i += 1
        return value


class ChoiceRecorder:
    def __init__(self, return_values):
        self._return_values = list(return_values)
        self._i = 0
        self.calls = []

    def __call__(self, indices, p=None):
        self.calls.append((np.array(indices), np.array(p)))
        if self._i >= len(self._return_values):
            raise AssertionError("choice() called more times than expected")
        value = self._return_values[self._i]
        self._i += 1
        return value


def test_dye_blot_constructor_stores_dimensions_and_starts_empty():
    blot = DyeBlot([4, 5], [2, 3])
    assert blot._grid_dims == [4, 5]
    assert blot._cell_dims == [2, 3]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 1), (2, 2), (1, 0)]
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(0, 1), (2, 2), (1, 0)]


def test_for_each_cell_passes_x_and_y_as_separate_arguments():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(7, 9)]
    captured = []

    def func(x, y):
        captured.append((x, y))

    blot.for_each_cell(func)
    assert captured == [(7, 9)]


def test_make_blot_single_cell_when_start_permitted(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.2]))
    blot = make_blot([4, 5], [2, 3], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 1)]


def test_make_blot_returns_none_if_start_cell_not_permitted(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.4, 0.6]))
    permitted = np.zeros((4, 5), dtype=bool)
    permitted[2, 3] = True

    blot = make_blot([4, 5], [2, 3], 1, permitted=permitted)
    assert blot is None


def test_make_blot_defaults_to_all_permitted_when_none(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.75, 0.99]))
    blot = make_blot([4, 5], [1, 1], 1, permitted=None)

    assert blot is not None
    assert blot._cells == [(3, 4)]


def test_make_blot_adds_horizontal_neighbour_with_correct_probability_weights(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1, 0.1]))
    choice = ChoiceRecorder([0])
    monkeypatch.setattr(np.random, "choice", choice)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]
    assert len(choice.calls) == 1
    indices, probabilities = choice.calls[0]
    assert list(indices) == [0, 1]
    assert np.allclose(probabilities, [5 / 9.0, 4 / 9.0])


def test_make_blot_adds_vertical_neighbour_with_correct_probability_weights(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1, 0.1]))
    choice = ChoiceRecorder([0])
    monkeypatch.setattr(np.random, "choice", choice)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]
    assert len(choice.calls) == 1


def test_make_blot_returns_none_when_non_permitted_neighbour_and_squeeze_false(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1, 0.1]))
    monkeypatch.setattr(np.random, "choice", ChoiceRecorder([0]))

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False

    blot = make_blot([3, 3], [2, 3], 2, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_skips_non_permitted_neighbour_when_squeeze_true(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1, 0.1, 0.1]))
    monkeypatch.setattr(np.random, "choice", ChoiceRecorder([0, 0, 0]))

    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    permitted[1, 0] = False
    permitted[0, 1] = True

    blot = make_blot([3, 3], [2, 3], 2, permitted=permitted, squeeze=True)
    assert blot is not None
    assert blot._cells == [(0, 0), (0, 1)]


def test_make_blot_stops_when_no_neighbours_remain_before_requesting_more_cells(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1]))
    monkeypatch.setattr(np.random, "choice", ChoiceRecorder([]))

    blot = make_blot([1, 1], [1, 1], 10)
    assert blot is not None
    assert blot._cells == [(0, 0)]
    assert blot.num_cells() == 1


def test_make_blot_does_not_add_duplicate_cells(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.1, 0.1, 0.1, 0.1]))
    monkeypatch.setattr(np.random, "choice", ChoiceRecorder([0, 1]))

    blot = make_blot([2, 2], [1, 1], 3)

    assert blot is not None
    assert blot.num_cells() == len(set(blot._cells))
    assert blot._cells == [(0, 0), (1, 0), (1, 1)]


def test_make_blot_respects_grid_bounds_for_start_cell(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.9999, 0.9999]))
    blot = make_blot([2, 3], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(1, 2)]


def test_make_blot_can_construct_multiple_cells_without_mask(monkeypatch):
    monkeypatch.setattr(np.random, "random", SequenceRandom([0.2, 0.2, 0.2, 0.2, 0.2]))
    monkeypatch.setattr(np.random, "choice", ChoiceRecorder([0, 1]))

    blot = make_blot([3, 3], [2, 2], 3)

    assert blot is not None
    assert blot.num_cells() == 3
    assert blot._cells[0] == (0, 0)
    assert all(0 <= x < 3 and 0 <= y < 3 for x, y in blot._cells)
