import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


@pytest.fixture(autouse=True)
def _seed_rng():
    np.random.seed(12345)
    yield


def _cells_of(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


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


def test_new_dyeblot_is_empty():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([10, 10], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_calls_in_insertion_order_with_coordinates():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(1, 2), (3, 4), (0, 0)])
    assert _cells_of(blot) == [(1, 2), (3, 4), (0, 0)]
    assert blot.num_cells() == 3


def test_for_each_cell_returns_none():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.append((2, 2))
    assert blot.for_each_cell(lambda x, y: 42) is None


def test_for_each_cell_propagates_exceptions_from_callback():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.append((2, 2))

    def boom(x, y):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


def test_make_blot_single_cell_grid_one_cell():
    blot = make_blot([1, 1], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_single_cell_grid_without_squeeze_returns_none_when_too_big():
    # Neighbours fall outside the grid, so growth must fail.
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_single_cell_grid_with_squeeze_stops_at_one_cell():
    blot = make_blot([1, 1], [1, 1], 5, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_squeeze_fills_whole_grid_when_requested():
    blot = make_blot([3, 3], [1, 1], 9, squeeze=True)
    assert blot.num_cells() == 9
    assert set(_cells_of(blot)) == {(x, y) for x in range(3) for y in range(3)}


def test_make_blot_squeeze_caps_at_grid_size():
    blot = make_blot([2, 3], [1, 1], 100, squeeze=True)
    assert blot.num_cells() == 6


@pytest.mark.parametrize("seed", range(10))
def test_make_blot_cells_unique_in_bounds_and_connected(seed):
    np.random.seed(seed)
    blot = make_blot([8, 6], [2, 3], 12, squeeze=True)
    assert blot is not None
    cells = _cells_of(blot)
    assert len(cells) == 12
    assert len(set(cells)) == 12
    for x, y in cells:
        assert 0 <= x < 8
        assert 0 <= y < 6
    assert _is_connected(cells)


def test_make_blot_zero_requested_cells_still_places_start_cell():
    blot = make_blot([4, 4], [1, 1], 0)
    assert blot.num_cells() == 1


def test_make_blot_fully_masked_grid_returns_none():
    permitted = np.zeros((4, 4), dtype=bool)
    assert make_blot([4, 4], [1, 1], 3, permitted=permitted) is None
    assert make_blot([4, 4], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_single_permitted_cell_with_squeeze():
    permitted = np.zeros((3, 4), dtype=bool)
    permitted[1, 2] = True
    results = []
    for seed in range(200):
        np.random.seed(seed)
        results.append(make_blot([3, 4], [1, 1], 5, permitted=permitted, squeeze=True))
    for blot in results:
        assert blot is None or _cells_of(blot) == [(1, 2)]
    assert any(b is not None for b in results)


def test_make_blot_respects_mask_with_squeeze():
    permitted = np.zeros((6, 6), dtype=bool)
    permitted[:, :3] = True  # only the lower half is allowed
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 8, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        for x, y in _cells_of(blot):
            assert permitted[x, y]
            assert y < 3


def test_make_blot_without_squeeze_never_includes_masked_cells():
    permitted = np.ones((5, 5), dtype=bool)
    permitted[2, 2] = False
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 10, permitted=permitted, squeeze=False)
        if blot is None:
            continue
        assert (2, 2) not in _cells_of(blot)
        assert blot.num_cells() == 10


def test_make_blot_is_reproducible_with_same_seed():
    np.random.seed(7)
    first = _cells_of(make_blot([10, 10], [1, 1], 15, squeeze=True))
    np.random.seed(7)
    second = _cells_of(make_blot([10, 10], [1, 1], 15, squeeze=True))
    assert first == second


def test_make_blot_accepts_tuple_dims():
    blot = make_blot((3, 3), (1, 1), 4, squeeze=True)
    assert blot.num_cells() == 4
