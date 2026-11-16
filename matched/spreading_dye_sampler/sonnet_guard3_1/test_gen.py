import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _assert_connected(cells):
    for i in range(1, len(cells)):
        assert any(_adjacent(cells[i], cells[j]) for j in range(i)), cells[i]


def test_new_blot_is_empty_and_stores_dims():
    np.random.seed(0)
    blot = DyeBlot([10, 20], [3, 4])
    assert blot.num_cells() == 0
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [3, 4]


def test_for_each_cell_on_empty_blot_never_calls_function():
    np.random.seed(0)
    blot = DyeBlot([5, 5], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_visits_cells_in_order():
    np.random.seed(0)
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2), (3, 4), (0, 0)]
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(1, 2), (3, 4), (0, 0)]
    assert blot.num_cells() == 3


def test_for_each_cell_passes_positional_x_then_y():
    np.random.seed(0)
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(7, 9)]
    received = []

    def fn(*args, **kwargs):
        received.append((args, kwargs))

    blot.for_each_cell(fn)
    assert received == [((7, 9), {})]


def test_make_blot_single_cell_always_succeeds():
    np.random.seed(1)
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([6, 4], [1, 1], 1)
        assert isinstance(blot, DyeBlot)
        assert blot.num_cells() == 1
        x, y = blot._cells[0]
        assert 0 <= x < 6
        assert 0 <= y < 4


def test_make_blot_one_by_one_grid_no_squeeze():
    np.random.seed(2)
    blot = make_blot([1, 1], [1, 1], 1)
    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_make_blot_one_by_one_grid_larger_request():
    np.random.seed(3)
    # Without squeeze an out-of-grid neighbour gets drawn -> None.
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None
    # With squeeze the out-of-grid neighbours are dropped.
    blot = make_blot([1, 1], [1, 1], 2, squeeze=True)
    assert blot is not None
    assert blot._cells == [(0, 0)]


def test_squeeze_gives_exact_size_unique_connected_in_bounds():
    np.random.seed(4)
    for seed in range(15):
        np.random.seed(seed)
        blot = make_blot([8, 6], [1, 1], 20, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 20
        cells = blot._cells
        assert len(set(cells)) == 20
        for x, y in cells:
            assert 0 <= x < 8
            assert 0 <= y < 6
        _assert_connected(cells)


def test_squeeze_with_oversized_request_fills_whole_grid():
    np.random.seed(5)
    blot = make_blot([3, 3], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 9
    assert set(blot._cells) == {(x, y) for x in range(3) for y in range(3)}


def test_no_squeeze_oversized_request_returns_none():
    np.random.seed(6)
    for seed in range(10):
        np.random.seed(seed)
        assert make_blot([2, 2], [1, 1], 10, squeeze=False) is None


def test_same_seed_gives_same_blot():
    np.random.seed(7)
    first = make_blot([10, 10], [2, 3], 15, squeeze=True)
    np.random.seed(7)
    second = make_blot([10, 10], [2, 3], 15, squeeze=True)
    assert first is not None and second is not None
    assert first._cells == second._cells


def test_all_false_mask_returns_none():
    np.random.seed(8)
    mask = np.zeros((4, 4), dtype=bool)
    for squeeze in (False, True):
        for seed in range(5):
            np.random.seed(seed)
            assert make_blot([4, 4], [1, 1], 3, permitted=mask, squeeze=squeeze) is None


def test_start_cell_uses_width_for_x_and_height_for_y(monkeypatch):
    np.random.seed(9)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    blot = make_blot([4, 10], [1, 1], 1)
    assert blot is not None
    # floor(0.5 * 4) = 2, floor(0.5 * 10) = 5
    assert blot._cells == [(2, 5)]


def test_unpermitted_start_returns_none(monkeypatch):
    np.random.seed(10)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    assert make_blot([5, 5], [1, 1], 3, permitted=mask, squeeze=True) is None


def test_single_permitted_island_squeeze_vs_no_squeeze(monkeypatch):
    np.random.seed(11)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert make_blot([5, 5], [1, 1], 3, permitted=mask, squeeze=False) is None
    blot = make_blot([5, 5], [1, 1], 3, permitted=mask, squeeze=True)
    assert blot is not None
    assert blot._cells == [(2, 2)]


def test_squeeze_respects_mask(monkeypatch):
    np.random.seed(12)
    monkeypatch.setattr(np.random, "random", lambda: 0.1)  # start at (0, 0)
    mask = np.zeros((6, 6), dtype=bool)
    mask[:3, :] = True  # columns x < 3 permitted (18 cells)
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 10, permitted=mask, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 10
        assert len(set(blot._cells)) == 10
        assert all(x < 3 for x, _ in blot._cells)
        _assert_connected(blot._cells)


def test_neighbour_probabilities_weighted_by_shared_edge(monkeypatch):
    np.random.seed(13)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    recorded = []

    def fake_choice(indices, p=None):
        recorded.append((list(indices), None if p is None else list(p)))
        return 0

    monkeypatch.setattr(np.random, "choice", fake_choice)
    # cell width 3, height 5; start (2, 2)
    blot = make_blot([5, 5], [3, 5], 2)
    assert blot is not None
    assert len(recorded) == 1
    indices, p = recorded[0]
    assert indices == [0, 1, 2, 3]
    # neighbours in insertion order: left, up, right, down
    # left/right share edges of cell_height (5); up/down of cell_width (3)
    expected = [5 / 16.0, 3 / 16.0, 5 / 16.0, 3 / 16.0]
    assert p == pytest.approx(expected)
    # index 0 is the left neighbour
    assert blot._cells == [(2, 2), (1, 2)]


def test_make_blot_does_not_modify_permitted_mask():
    np.random.seed(14)
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False
    original = mask.copy()
    make_blot([5, 5], [1, 1], 8, permitted=mask, squeeze=True)
    assert np.array_equal(mask, original)
