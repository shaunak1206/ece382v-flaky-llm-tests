import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _is_connected(cells):
    cells = set(cells)
    if not cells:
        return True
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen == cells


def _collect(blot):
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    return calls


def _fix_start(monkeypatch, value):
    monkeypatch.setattr(np.random, "random", lambda: value)


def test_new_blot_is_empty():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls():
    blot = DyeBlot([10, 10], [1, 1])
    calls = _collect(blot)
    assert calls == []


def test_for_each_cell_calls_in_order_with_x_y():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 2), (3, 4), (5, 6)]
    assert blot.num_cells() == 3
    assert _collect(blot) == [(1, 2), (3, 4), (5, 6)]


def test_for_each_cell_passes_two_positional_args():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(7, 8)]
    received = []

    def f(*args, **kwargs):
        received.append((args, kwargs))

    blot.for_each_cell(f)
    assert received == [((7, 8), {})]


def test_make_blot_returns_requested_number_of_cells_with_squeeze():
    np.random.seed(0)
    blot = make_blot([10, 10], [1, 1], 15, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 15


def test_make_blot_cells_unique_in_bounds_and_connected():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([8, 12], [2, 3], 20, squeeze=True)
        cells = _collect(blot)
        assert len(cells) == 20
        assert len(set(cells)) == 20
        for x, y in cells:
            assert 0 <= x < 8
            assert 0 <= y < 12
        assert _is_connected(cells)


def test_make_blot_single_cell_request():
    np.random.seed(1)
    blot = make_blot([5, 5], [1, 1], 1)
    assert blot.num_cells() == 1


def test_make_blot_zero_requested_still_contains_start_cell(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    blot = make_blot([11, 11], [1, 1], 0)
    assert _collect(blot) == [(5, 5)]


def test_make_blot_starts_at_position_given_by_random(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    blot = make_blot([11, 11], [1, 1], 1)
    assert _collect(blot) == [(5, 5)]


def test_make_blot_second_cell_is_adjacent_to_start(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([11, 11], [1, 1], 2)
        cells = _collect(blot)
        assert cells[0] == (5, 5)
        assert len(cells) == 2
        dx = abs(cells[1][0] - 5)
        dy = abs(cells[1][1] - 5)
        assert dx + dy == 1


def test_make_blot_fills_whole_nonsquare_grid_when_squeezed():
    np.random.seed(3)
    blot = make_blot([4, 6], [1, 1], 24, squeeze=True)
    cells = _collect(blot)
    assert len(cells) == 24
    assert set(cells) == {(x, y) for x in range(4) for y in range(6)}


def test_make_blot_request_exceeding_grid_with_squeeze_stops_at_grid_size():
    np.random.seed(4)
    blot = make_blot([3, 3], [1, 1], 100, squeeze=True)
    assert blot.num_cells() == 9


def test_make_blot_without_squeeze_returns_none_when_leaving_grid():
    np.random.seed(5)
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_all_masked_returns_none():
    permitted = np.zeros((5, 5), dtype=bool)
    np.random.seed(6)
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_with_mask_only_uses_permitted_cells(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((6, 6), dtype=bool)
    permitted[:3, :] = True
    np.random.seed(7)
    blot = make_blot([6, 6], [1, 1], 100, permitted=permitted, squeeze=True)
    cells = _collect(blot)
    assert len(cells) == 18
    assert all(x < 3 for x, _ in cells)


def test_make_blot_mask_without_squeeze_returns_none_on_masked_neighbour(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    np.random.seed(8)
    assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_isolated_permitted_cell_with_squeeze(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, 0] = True
    np.random.seed(9)
    blot = make_blot([3, 3], [1, 1], 5, permitted=permitted, squeeze=True)
    assert _collect(blot) == [(0, 0)]


def test_make_blot_start_on_masked_cell_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.ones((4, 4), dtype=bool)
    permitted[0, 0] = False
    assert make_blot([4, 4], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_for_each_cell_on_made_blot_matches_num_cells():
    np.random.seed(10)
    blot = make_blot([10, 10], [1, 1], 7, squeeze=True)
    calls = _collect(blot)
    assert len(calls) == blot.num_cells() == 7
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in calls)
