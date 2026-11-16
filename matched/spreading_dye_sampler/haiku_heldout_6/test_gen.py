import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect_cells(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _patch_random(monkeypatch, value):
    # make_blot imports numpy.random.random at call time, so patching the
    # numpy.random module attribute controls the chosen start cell.
    monkeypatch.setattr(np.random, "random", lambda: value)


# ---------------------------------------------------------------------------
# DyeBlot
# ---------------------------------------------------------------------------

def test_new_blot_is_empty():
    blot = DyeBlot([4, 3], [10, 10])
    assert blot.num_cells() == 0


def test_new_blot_for_each_cell_calls_nothing():
    blot = DyeBlot([4, 3], [10, 10])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_in_order():
    blot = DyeBlot([4, 3], [10, 10])
    blot._cells = [(0, 0), (2, 1), (3, 2)]
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(0, 0), (2, 1), (3, 2)]


def test_for_each_cell_call_count_matches_num_cells():
    blot = DyeBlot([4, 3], [10, 10])
    blot._cells = [(0, 0), (1, 1), (2, 2), (3, 0)]
    count = []
    blot.for_each_cell(lambda x, y: count.append(1))
    assert len(count) == blot.num_cells() == 4


def test_for_each_cell_propagates_exceptions():
    blot = DyeBlot([4, 3], [10, 10])
    blot._cells = [(0, 0)]

    def boom(x, y):
        raise ValueError("stop")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


# ---------------------------------------------------------------------------
# make_blot: basic construction
# ---------------------------------------------------------------------------

def test_make_blot_returns_dyeblot_instance():
    blot = make_blot([5, 5], [10, 10], 1)
    assert isinstance(blot, DyeBlot)


def test_make_blot_single_cell_request(monkeypatch):
    _patch_random(monkeypatch, 0.0)
    blot = make_blot([5, 5], [10, 10], 1)
    assert blot.num_cells() == 1
    assert _collect_cells(blot) == [(0, 0)]


def test_make_blot_start_cell_chosen_from_random(monkeypatch):
    # 0.999 * 4 -> x = 3, 0.999 * 3 -> y = 2
    _patch_random(monkeypatch, 0.999)
    blot = make_blot([4, 3], [10, 10], 1)
    assert _collect_cells(blot) == [(3, 2)]


def test_make_blot_reaches_requested_size_on_large_grid():
    np.random.seed(0)
    blot = make_blot([10, 10], [5, 5], 7)
    assert blot.num_cells() == 7


def test_make_blot_cells_are_unique_and_in_bounds():
    np.random.seed(1)
    blot = make_blot([6, 5], [5, 5], 12)
    cells = _collect_cells(blot)
    assert len(cells) == len(set(cells))
    for x, y in cells:
        assert 0 <= x < 6
        assert 0 <= y < 5


def test_make_blot_cell_coordinates_are_ints():
    np.random.seed(2)
    blot = make_blot([5, 5], [5, 5], 4)
    for x, y in _collect_cells(blot):
        assert isinstance(x, int)
        assert isinstance(y, int)


def test_make_blot_fills_small_grid_completely():
    np.random.seed(3)
    blot = make_blot([3, 3], [5, 5], 9)
    cells = set(_collect_cells(blot))
    expected = {(x, y) for x in range(3) for y in range(3)}
    assert cells == expected


def test_make_blot_stops_when_no_neighbours_remain_on_1x1_grid(monkeypatch):
    # Only one in-bounds cell exists; with squeeze the out-of-bounds
    # neighbours are removed and growth must terminate.
    _patch_random(monkeypatch, 0.0)
    blot = make_blot([1, 1], [5, 5], 10, squeeze=True)
    assert blot.num_cells() == 1


def test_make_blot_is_reproducible_with_same_seed():
    np.random.seed(42)
    first = _collect_cells(make_blot([6, 6], [5, 5], 10))
    np.random.seed(42)
    second = _collect_cells(make_blot([6, 6], [5, 5], 10))
    assert first == second


# ---------------------------------------------------------------------------
# make_blot: out-of-grid neighbours and squeeze behaviour
# ---------------------------------------------------------------------------

def test_out_of_grid_neighbour_returns_none_without_squeeze(monkeypatch):
    # On a 1x1 grid every neighbour is out of bounds, so the first draw
    # is non-permitted and squeeze=False must abort with None.
    _patch_random(monkeypatch, 0.0)
    result = make_blot([1, 1], [5, 5], 2, squeeze=False)
    assert result is None


def test_squeeze_true_on_1x1_grid_keeps_single_cell(monkeypatch):
    _patch_random(monkeypatch, 0.0)
    blot = make_blot([1, 1], [5, 5], 2, squeeze=True)
    assert blot is not None
    assert _collect_cells(blot) == [(0, 0)]


def test_squeeze_true_on_2x1_grid_grows_to_both_cells(monkeypatch):
    _patch_random(monkeypatch, 0.0)
    blot = make_blot([2, 1], [5, 5], 2, squeeze=True)
    assert set(_collect_cells(blot)) == {(0, 0), (1, 0)}


# ---------------------------------------------------------------------------
# make_blot: permitted mask
# ---------------------------------------------------------------------------

def test_non_permitted_start_returns_none(monkeypatch):
    _patch_random(monkeypatch, 0.0)
    permitted = np.ones((4, 3), dtype=bool)
    permitted[0, 0] = False
    assert make_blot([4, 3], [5, 5], 3, permitted=permitted) is None


def test_all_non_permitted_returns_none():
    permitted = np.zeros((4, 3), dtype=bool)
    assert make_blot([4, 3], [5, 5], 3, permitted=permitted) is None


def test_non_permitted_neighbour_returns_none_without_squeeze(monkeypatch):
    # Only (1, 0) is permitted in a 3x1 grid, so its neighbours are blocked.
    _patch_random(monkeypatch, 0.0)
    permitted = np.array([[False], [True], [False]])
    # Start is forced to (0,0) which is not permitted; use 0.5 -> x=1.
    _patch_random(monkeypatch, 0.5)
    result = make_blot([3, 1], [5, 5], 3, permitted=permitted, squeeze=False)
    assert result is None


def test_squeeze_true_removes_non_permitted_neighbours(monkeypatch):
    _patch_random(monkeypatch, 0.5)  # start at (1, 0)
    permitted = np.array([[False], [True], [False]])
    blot = make_blot([3, 1], [5, 5], 3, permitted=permitted, squeeze=True)
    assert blot is not None
    assert _collect_cells(blot) == [(1, 0)]


def test_blot_never_uses_masked_cells(monkeypatch):
    _patch_random(monkeypatch, 0.0)  # start at (0, 0)
    permitted = np.ones((4, 3), dtype=bool)
    permitted[2, :] = False  # block column x = 2
    blot = make_blot([4, 3], [5, 5], 100, permitted=permitted, squeeze=True)
    cells = set(_collect_cells(blot))
    expected = {(x, y) for x in range(2) for y in range(3)}
    assert cells == expected
    assert blot.num_cells() == 6
