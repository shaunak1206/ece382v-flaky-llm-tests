import numpy as np
import pytest

from spreading_dye_sampler import dye_blot
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _is_connected(cells):
    """Each cell after the first must touch some earlier cell (insertion order)."""
    for i in range(1, len(cells)):
        x, y = cells[i]
        if not any(abs(x - a) + abs(y - b) == 1 for a, b in cells[:i]):
            return False
    return True


def test_new_blot_is_empty():
    blot = DyeBlot([10, 20], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([10, 20], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_in_order():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells.extend([(1, 2), (3, 4), (5, 6)])
    assert blot.num_cells() == 3
    assert _collect(blot) == [(1, 2), (3, 4), (5, 6)]


def test_for_each_cell_accepts_any_callable_object():
    class Recorder:
        def __init__(self):
            self.total = 0

        def __call__(self, x, y):
            self.total += x * 10 + y

    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(1, 2), (3, 4)])
    rec = Recorder()
    blot.for_each_cell(rec)
    assert rec.total == 12 + 34


def test_make_blot_single_cell_on_one_by_one_grid():
    blot = make_blot([1, 1], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert _collect(blot) == [(0, 0)]


def test_make_blot_single_cell_within_bounds():
    for _ in range(20):
        blot = make_blot([7, 4], [1, 1], 1)
        assert blot is not None
        cells = _collect(blot)
        assert len(cells) == 1
        x, y = cells[0]
        assert 0 <= x < 7 and 0 <= y < 4


def test_make_blot_stores_dimensions():
    blot = make_blot([3, 3], [2, 5], 1)
    assert list(blot._grid_dims) == [3, 3]
    assert list(blot._cell_dims) == [2, 5]


def test_squeeze_reaches_requested_size_with_unique_in_bounds_cells():
    for _ in range(20):
        blot = make_blot([5, 5], [1, 1], 10, squeeze=True)
        assert blot is not None
        cells = _collect(blot)
        assert blot.num_cells() == 10
        assert len(set(cells)) == 10
        assert all(0 <= x < 5 and 0 <= y < 5 for x, y in cells)


def test_squeeze_blot_is_connected():
    for _ in range(20):
        blot = make_blot([6, 6], [1, 1], 12, squeeze=True)
        assert _is_connected(_collect(blot))


def test_request_larger_than_grid_with_squeeze_fills_grid():
    blot = make_blot([3, 2], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert sorted(_collect(blot)) == sorted((x, y) for x in range(3) for y in range(2))


def test_non_square_grid_coordinates_respect_dimensions():
    for _ in range(10):
        blot = make_blot([8, 3], [1, 1], 10, squeeze=True)
        assert blot.num_cells() == 10
        assert all(0 <= x < 8 and 0 <= y < 3 for x, y in _collect(blot))


def test_all_false_mask_returns_none():
    permitted = np.zeros([4, 4], dtype=bool)
    assert make_blot([4, 4], [1, 1], 3, permitted=permitted) is None
    assert make_blot([4, 4], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_non_squeeze_interior_blot_succeeds(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    blot = make_blot([100, 100], [1, 1], 5)
    assert blot is not None
    cells = _collect(blot)
    assert cells[0] == (50, 50)
    assert len(set(cells)) == 5
    assert _is_connected(cells)


def test_isolated_permitted_cell_non_squeeze_returns_none(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    permitted = np.zeros([5, 5], dtype=bool)
    permitted[0, 0] = True
    assert make_blot([5, 5], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_isolated_permitted_cell_squeeze_returns_single_cell(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    permitted = np.zeros([5, 5], dtype=bool)
    permitted[0, 0] = True
    blot = make_blot([5, 5], [1, 1], 2, permitted=permitted, squeeze=True)
    assert blot is not None
    assert _collect(blot) == [(0, 0)]


def test_squeeze_only_uses_permitted_cells(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    permitted = np.zeros([6, 6], dtype=bool)
    permitted[3:, :] = True
    for _ in range(20):
        blot = make_blot([6, 6], [1, 1], 8, permitted=permitted, squeeze=True)
        assert blot is not None
        cells = _collect(blot)
        assert len(cells) == 8
        assert all(x >= 3 for x, y in cells)


def test_integer_mask_is_treated_as_truthy_values(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    permitted = np.zeros([4, 4], dtype=int)
    permitted[0:2, 0:2] = 1
    blot = make_blot([4, 4], [1, 1], 10, permitted=permitted, squeeze=True)
    assert blot is not None
    assert sorted(_collect(blot)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
