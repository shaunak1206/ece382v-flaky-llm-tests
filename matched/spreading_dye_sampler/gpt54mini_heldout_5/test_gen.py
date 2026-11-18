import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def test_dye_blot_initialization_stores_dimensions_and_starts_empty():
    blot = DyeBlot([12, 34], [5, 6])

    assert blot._grid_dims == [12, 34]
    assert blot._cell_dims == [5, 6]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_counts_stored_cells():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1)]

    assert blot.num_cells() == 3


def test_for_each_cell_calls_function_for_each_cell_in_order():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(2, 4), (5, 6), (7, 8)]
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(2, 4), (5, 6), (7, 8)]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = DyeBlot([3, 3], [1, 1])
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == []


def test_make_blot_with_zero_required_cells_returns_empty_blot(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.1)
    blot = make_blot([4, 4], [2, 2], 0)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 0


def test_make_blot_with_permitted_none_uses_all_cells_by_default(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    blot = make_blot([5, 6], [2, 3], 1, permitted=None)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    permitted = np.zeros((3, 3), dtype=bool)

    blot = make_blot([3, 3], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_can_build_single_cell_blot(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    blot = make_blot([4, 4], [2, 2], 1)

    assert isinstance(blot, DyeBlot)
    assert blot._cells == [(0, 0)]
    assert blot.num_cells() == 1


def test_make_blot_adds_horizontal_neighbor_with_deterministic_choice(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    blot = make_blot([4, 4], [2, 3], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]
    assert blot.num_cells() == 2


def test_make_blot_adds_vertical_neighbor_with_deterministic_choice(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 1)

    blot = make_blot([4, 4], [2, 3], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (0, 1)]
    assert blot.num_cells() == 2


def test_make_blot_skips_nonpermitted_cell_when_squeeze_true(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    permitted = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    blot = make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_make_blot_returns_none_when_nonpermitted_neighbor_is_selected_and_squeeze_false(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    permitted = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    blot = make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_uses_grid_dimensions_to_limit_start_cell(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.999999)

    blot = make_blot([3, 7], [1, 1], 1)

    assert blot is not None
    assert blot._cells == [(2, 6)]


def test_make_blot_does_not_modify_permitted_array(monkeypatch):
    random_values = iter([0.0, 0.0])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: 0)

    permitted = np.array(
        [
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )
    original = permitted.copy()

    blot = make_blot([2, 2], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert np.array_equal(permitted, original)


def test_make_blot_can_return_requested_number_of_cells_with_multiple_steps(monkeypatch):
    random_values = iter([0.0, 0.0, 0.0])
    choice_calls = iter([0, 1])
    monkeypatch.setattr(np.random, "random", lambda: next(random_values))
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: next(choice_calls))

    blot = make_blot([4, 4], [2, 2], 3)

    assert blot is not None
    assert blot.num_cells() == 3
    assert blot._cells == [(0, 0), (1, 0), (1, 1)]
