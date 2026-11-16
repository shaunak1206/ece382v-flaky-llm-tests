import numpy as np
import pytest

from spreading_dye_sampler import dye_blot
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _cells_of(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def test_new_blot_is_empty():
    np.random.seed(0)
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
    np.random.seed(0)
    blot = DyeBlot([10, 10], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_visits_cells_in_order_with_x_then_y():
    np.random.seed(0)
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells.extend([(3, 4), (0, 1), (7, 2)])
    assert _cells_of(blot) == [(3, 4), (0, 1), (7, 2)]
    assert blot.num_cells() == 3


def test_for_each_cell_ignores_return_value_of_function():
    np.random.seed(0)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells.extend([(1, 1), (2, 2)])
    count = []

    def fn(x, y):
        count.append(1)
        return "ignored"

    assert blot.for_each_cell(fn) is None
    assert len(count) == 2


def test_make_blot_squeeze_gives_requested_number_of_cells():
    np.random.seed(1)
    blot = make_blot([10, 8], [1, 1], 15, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 15
    assert len(_cells_of(blot)) == 15


def test_make_blot_cells_unique_and_in_bounds():
    np.random.seed(2)
    blot = make_blot([7, 5], [2, 3], 20, squeeze=True)
    cells = _cells_of(blot)
    assert len(set(cells)) == len(cells) == 20
    for x, y in cells:
        assert 0 <= x < 7
        assert 0 <= y < 5


def test_make_blot_cells_form_connected_growth():
    np.random.seed(3)
    blot = make_blot([8, 8], [1, 1], 25, squeeze=True)
    cells = _cells_of(blot)
    for i in range(1, len(cells)):
        x, y = cells[i]
        assert any(abs(x - cx) + abs(y - cy) == 1 for cx, cy in cells[:i])


def test_make_blot_request_larger_than_grid_fills_grid_when_squeezing():
    np.random.seed(4)
    blot = make_blot([3, 2], [1, 1], 100, squeeze=True)
    assert blot.num_cells() == 6
    assert set(_cells_of(blot)) == {(x, y) for x in range(3) for y in range(2)}


def test_make_blot_single_cell_request_returns_one_cell():
    np.random.seed(5)
    blot = make_blot([5, 5], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_zero_request_still_contains_start_cell():
    np.random.seed(6)
    blot = make_blot([5, 5], [1, 1], 0)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_stores_dims():
    np.random.seed(7)
    blot = make_blot([6, 4], [2, 3], 3, squeeze=True)
    assert blot._grid_dims == [6, 4]
    assert blot._cell_dims == [2, 3]


def test_make_blot_nothing_permitted_returns_none():
    np.random.seed(8)
    permitted = np.zeros((5, 5), dtype=bool)
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted) is None
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_non_squeeze_on_tiny_grid_hits_edge_and_returns_none():
    np.random.seed(9)
    # Single cell grid: every neighbour is out of bounds.
    assert make_blot([1, 1], [1, 1], 5) is None


def test_make_blot_single_cell_grid_with_single_request_succeeds():
    np.random.seed(10)
    blot = make_blot([1, 1], [1, 1], 1)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_start_cell_follows_random_draw(monkeypatch):
    np.random.seed(11)
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    blot = make_blot([5, 5], [1, 1], 1)
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_forced_start_on_masked_cell_returns_none(monkeypatch):
    np.random.seed(12)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (2, 2)
    permitted = np.ones((5, 5), dtype=bool)
    permitted[2, 2] = False
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted) is None
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_squeeze_stays_inside_permitted_region(monkeypatch):
    np.random.seed(13)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (2, 2)
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[1:4, 1:4] = True
    blot = make_blot([5, 5], [1, 1], 9, permitted=permitted, squeeze=True)
    assert blot is not None
    assert set(_cells_of(blot)) == {(x, y) for x in range(1, 4) for y in range(1, 4)}


def test_make_blot_mask_is_indexed_by_x_then_y(monkeypatch):
    np.random.seed(14)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (2, 1) on 4x3
    permitted = np.zeros((4, 3), dtype=bool)
    permitted[2, :] = True
    blot = make_blot([4, 3], [1, 1], 3, permitted=permitted, squeeze=True)
    assert blot is not None
    assert set(_cells_of(blot)) == {(2, 0), (2, 1), (2, 2)}


def test_make_blot_non_squeeze_returns_none_when_masked_neighbour_drawn(monkeypatch):
    np.random.seed(15)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (2, 2)
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[2, 2] = True
    assert make_blot([5, 5], [1, 1], 2, permitted=permitted) is None


def test_make_blot_neighbour_probabilities_weighted_by_shared_edge(monkeypatch):
    np.random.seed(16)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (2, 2)
    seen = {}

    def fake_choice(indices, p=None):
        seen["p"] = np.asarray(p)
        return indices[0]

    monkeypatch.setattr(np.random, "choice", fake_choice)
    blot = make_blot([5, 5], [2, 3], 2)
    # Neighbour order: left, up, right, down; left/right share the cell height
    # (3), up/down share the cell width (2).
    assert seen["p"] == pytest.approx([0.3, 0.2, 0.3, 0.2])
    assert float(seen["p"].sum()) == pytest.approx(1.0)
    assert _cells_of(blot) == [(2, 2), (1, 2)]


def test_make_blot_same_seed_gives_same_blot():
    np.random.seed(17)
    first = _cells_of(make_blot([9, 9], [1, 1], 12, squeeze=True))
    np.random.seed(17)
    second = _cells_of(make_blot([9, 9], [1, 1], 12, squeeze=True))
    assert first == second
