import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect(blot):
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    return calls


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


def test_new_blot_is_empty():
    blot = DyeBlot([10, 20], [1, 2])
    assert blot.num_cells() == 0
    assert _collect(blot) == []


def test_blot_stores_dimensions():
    blot = DyeBlot([10, 20], [1, 2])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [1, 2]


def test_for_each_cell_calls_in_order_with_coordinates():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 1), (2, 3), (4, 0)]
    assert blot.num_cells() == 3
    assert _collect(blot) == [(0, 1), (2, 3), (4, 0)]


def test_for_each_cell_passes_two_positional_args():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2)]
    received = []

    def func(*args, **kwargs):
        received.append((args, kwargs))

    blot.for_each_cell(func)
    assert received == [((1, 2), {})]


def test_for_each_cell_ignores_return_value():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 1), (2, 2)]
    count = []
    result = blot.for_each_cell(lambda x, y: count.append(1) or 99)
    assert result is None
    assert len(count) == 2


def test_make_blot_single_cell_no_squeeze():
    np.random.seed(0)
    blot = make_blot([10, 10], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1


def test_make_blot_start_cell_from_random(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    blot = make_blot([4, 6], [1, 1], 1)
    assert _collect(blot) == [(2, 3)]


def test_make_blot_returns_none_if_start_not_permitted(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    permitted = np.ones((4, 6), dtype=bool)
    permitted[2, 3] = False
    assert make_blot([4, 6], [1, 1], 1, permitted=permitted) is None
    assert make_blot([4, 6], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_all_masked_returns_none():
    np.random.seed(1)
    permitted = np.zeros((5, 5), dtype=bool)
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_squeeze_reaches_requested_size_unique_in_bounds():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([5, 4], [1, 1], 12, squeeze=True)
        assert blot is not None
        cells = _collect(blot)
        assert len(cells) == 12
        assert len(set(cells)) == 12
        for x, y in cells:
            assert 0 <= x < 5
            assert 0 <= y < 4


def test_make_blot_cells_are_connected():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([8, 6], [1, 1], 15, squeeze=True)
        assert blot is not None
        assert _is_connected(_collect(blot))


def test_make_blot_request_larger_than_grid_fills_grid_with_squeeze():
    np.random.seed(3)
    blot = make_blot([3, 2], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert set(_collect(blot)) == {(x, y) for x in range(3) for y in range(2)}


def test_make_blot_one_by_one_grid():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 1)
    assert _collect(blot) == [(0, 0)]


def test_make_blot_one_by_one_grid_no_squeeze_returns_none_when_too_big():
    np.random.seed(0)
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_one_by_one_grid_squeeze_returns_partial_blot():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 2, squeeze=True)
    assert blot is not None
    assert _collect(blot) == [(0, 0)]


def test_make_blot_squeeze_respects_mask():
    permitted = np.zeros((6, 3), dtype=bool)
    permitted[:3, :] = True  # left half permitted
    successes = 0
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([6, 3], [1, 1], 5, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        successes += 1
        cells = _collect(blot)
        assert len(cells) == 5
        assert all(permitted[c] for c in cells)
        assert all(x < 3 for x, _ in cells)
    assert successes > 0


def test_make_blot_squeeze_with_mask_stops_when_region_exhausted():
    permitted = np.zeros((6, 3), dtype=bool)
    permitted[:2, :2] = True  # 4 permitted cells
    found = False
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([6, 3], [1, 1], 10, permitted=permitted, squeeze=True)
        if blot is not None:
            found = True
            assert set(_collect(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert found


def test_make_blot_no_squeeze_never_uses_masked_cells():
    permitted = np.ones((6, 4), dtype=bool)
    permitted[3:, :] = False
    for seed in range(40):
        np.random.seed(seed)
        blot = make_blot([6, 4], [1, 1], 6, permitted=permitted, squeeze=False)
        if blot is not None:
            assert all(permitted[c] for c in _collect(blot))


def test_make_blot_no_squeeze_returns_none_when_blocked():
    # Only one permitted cell; asking for 2 cells must hit a masked/out-of-grid
    # neighbour, so no result can ever be a blot of size 2.
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True
    for seed in range(20):
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_stores_dims_on_blot():
    np.random.seed(0)
    blot = make_blot([7, 9], [2, 3], 3, squeeze=True)
    assert blot._grid_dims == [7, 9]
    assert blot._cell_dims == [2, 3]
    assert blot.num_cells() == 3
