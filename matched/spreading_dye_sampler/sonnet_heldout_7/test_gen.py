import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _start_at_origin(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.0)


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
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls():
    blot = DyeBlot([10, 10], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_calls_in_order_with_coordinates():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 2), (3, 4), (5, 6)]
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(1, 2), (3, 4), (5, 6)]
    assert blot.num_cells() == 3


def test_make_blot_returns_requested_number_of_cells_with_squeeze():
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([8, 8], [1, 1], 10, squeeze=True)
        assert isinstance(blot, DyeBlot)
        assert blot.num_cells() == 10


def test_make_blot_cells_unique_in_bounds_and_connected():
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([7, 5], [2, 3], 12, squeeze=True)
        cells = []
        blot.for_each_cell(lambda x, y: cells.append((x, y)))
        assert len(cells) == len(set(cells)) == 12
        for x, y in cells:
            assert 0 <= x < 7
            assert 0 <= y < 5
        assert _is_connected(cells)


def test_single_cell_blot_at_forced_start(monkeypatch):
    _start_at_origin(monkeypatch)
    blot = make_blot([5, 5], [1, 1], 1)
    assert blot is not None
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    assert cells == [(0, 0)]


def test_single_cell_blot_without_squeeze_never_fails():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 1, squeeze=False)
        assert blot is not None
        assert blot.num_cells() == 1


def test_zero_requested_cells_still_places_start_cell():
    np.random.seed(0)
    blot = make_blot([4, 4], [1, 1], 0)
    assert blot is not None
    assert blot.num_cells() == 1


def test_request_larger_than_grid_with_squeeze_fills_grid():
    np.random.seed(1)
    blot = make_blot([2, 3], [1, 1], 100, squeeze=True)
    assert blot is not None
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    assert sorted(cells) == [(x, y) for x in range(2) for y in range(3)]


def test_leaving_grid_without_squeeze_returns_none():
    np.random.seed(2)
    # A 1x1 grid has only out-of-bounds neighbours.
    assert make_blot([1, 1], [1, 1], 3, squeeze=False) is None


def test_all_forbidden_mask_returns_none():
    mask = np.zeros((5, 5), dtype=bool)
    np.random.seed(3)
    assert make_blot([5, 5], [1, 1], 3, permitted=mask, squeeze=True) is None
    assert make_blot([5, 5], [1, 1], 3, permitted=mask, squeeze=False) is None


def test_isolated_permitted_cell_with_squeeze_gives_one_cell(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (1, 1)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    blot = make_blot([3, 3], [1, 1], 3, permitted=mask, squeeze=True)
    assert blot is not None
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    assert cells == [(1, 1)]


def test_isolated_permitted_cell_without_squeeze_returns_none(monkeypatch):
    monkeypatch.setattr(np.random, "random", lambda: 0.5)  # start (1, 1)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    assert make_blot([3, 3], [1, 1], 3, permitted=mask, squeeze=False) is None


def test_mask_is_respected_with_squeeze(monkeypatch):
    _start_at_origin(monkeypatch)
    mask = np.zeros((6, 4), dtype=bool)
    mask[:3, :] = True
    for seed in range(15):
        np.random.seed(seed)
        blot = make_blot([6, 4], [1, 1], 6, permitted=mask, squeeze=True)
        assert blot is not None
        cells = []
        blot.for_each_cell(lambda x, y: cells.append((x, y)))
        assert len(cells) == 6
        assert all(x < 3 for x, _ in cells)
        assert _is_connected(cells)


def test_start_cell_not_permitted_returns_none(monkeypatch):
    _start_at_origin(monkeypatch)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    assert make_blot([4, 4], [1, 1], 3, permitted=mask, squeeze=True) is None


def test_neighbour_probabilities_weighted_by_shared_edge(monkeypatch):
    _start_at_origin(monkeypatch)
    recorded = []

    def fake_choice(a, size=None, replace=True, p=None):
        recorded.append(np.array(p))
        return 0

    monkeypatch.setattr(np.random, "choice", fake_choice)
    blot = make_blot([3, 3], [2, 3], 2, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 2
    # Neighbours of (0, 0) in order: (-1,0) h=3, (0,-1) w=2, (1,0) h=3, (0,1) w=2
    assert recorded[0] == pytest.approx([0.3, 0.2, 0.3, 0.2])
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    assert cells == [(0, 0), (1, 0)]
