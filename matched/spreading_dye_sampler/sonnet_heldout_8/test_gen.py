import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


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


def test_empty_blot_has_no_cells():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_never_calls_function():
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


def test_for_each_cell_passes_two_positional_args():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(7, 8)]
    received = []

    def fn(*args):
        received.append(args)

    blot.for_each_cell(fn)
    assert received == [(7, 8)]


def test_make_blot_single_cell_in_bounds():
    np.random.seed(0)
    blot = make_blot([20, 20], [1, 1], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    (x, y), = _cells_of(blot)
    assert 0 <= x < 20 and 0 <= y < 20


def test_make_blot_zero_requested_still_has_start_cell():
    np.random.seed(1)
    blot = make_blot([5, 5], [1, 1], 0)
    assert blot is not None
    assert blot.num_cells() == 1


def test_make_blot_squeeze_gives_exact_requested_size():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([20, 20], [1, 1], 10, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 10
        assert len(_cells_of(blot)) == 10


def test_make_blot_cells_unique_and_in_bounds():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([5, 5], [2, 3], 12, squeeze=True)
        cells = _cells_of(blot)
        assert len(cells) == len(set(cells))
        for x, y in cells:
            assert 0 <= x < 5
            assert 0 <= y < 5


def test_make_blot_is_connected():
    for seed in range(15):
        np.random.seed(seed)
        blot = make_blot([15, 15], [1, 1], 20, squeeze=True)
        assert _is_connected(_cells_of(blot))


def test_make_blot_non_square_grid_respects_dimensions():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([8, 3], [1, 1], 10, squeeze=True)
        assert blot is not None
        for x, y in _cells_of(blot):
            assert 0 <= x < 8
            assert 0 <= y < 3


def test_make_blot_larger_than_grid_with_squeeze_fills_grid():
    np.random.seed(3)
    blot = make_blot([3, 3], [1, 1], 100, squeeze=True)
    assert blot is not None
    assert set(_cells_of(blot)) == {(x, y) for x in range(3) for y in range(3)}


def test_make_blot_one_cell_grid_squeeze_vs_no_squeeze():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 5, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]
    np.random.seed(0)
    assert make_blot([1, 1], [1, 1], 5, squeeze=False) is None


def test_make_blot_no_squeeze_returns_none_or_full_blot():
    results = []
    for seed in range(30):
        np.random.seed(seed)
        results.append(make_blot([2, 2], [1, 1], 4, squeeze=False))
    for blot in results:
        if blot is not None:
            assert blot.num_cells() == 4
            assert set(_cells_of(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert any(b is None for b in results)


@pytest.mark.parametrize("squeeze", [True, False])
def test_make_blot_fully_masked_returns_none(squeeze):
    permitted = np.zeros((6, 6), dtype=bool)
    for seed in range(5):
        np.random.seed(seed)
        assert make_blot([6, 6], [1, 1], 3, permitted=permitted,
                         squeeze=squeeze) is None


def test_make_blot_squeeze_respects_mask():
    permitted = np.zeros((10, 10), dtype=bool)
    permitted[:5, :] = True
    found = 0
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 8, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        found += 1
        assert blot.num_cells() == 8
        for x, y in _cells_of(blot):
            assert permitted[x, y]
    assert found > 0


def test_make_blot_no_squeeze_respects_mask_when_successful():
    permitted = np.ones((10, 10), dtype=bool)
    permitted[0, :] = False
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 1, permitted=permitted, squeeze=False)
        if blot is not None:
            (x, y), = _cells_of(blot)
            assert permitted[x, y]


def test_make_blot_does_not_modify_mask():
    permitted = np.ones((6, 6), dtype=bool)
    permitted[2, 2] = False
    original = permitted.copy()
    np.random.seed(5)
    make_blot([6, 6], [1, 1], 10, permitted=permitted, squeeze=True)
    assert np.array_equal(permitted, original)


def test_make_blot_same_seed_is_reproducible():
    np.random.seed(42)
    a = _cells_of(make_blot([20, 20], [1, 1], 15, squeeze=True))
    np.random.seed(42)
    b = _cells_of(make_blot([20, 20], [1, 1], 15, squeeze=True))
    assert a == b


def test_make_blot_stores_dimensions():
    np.random.seed(0)
    blot = make_blot([12, 9], [3, 4], 2, squeeze=True)
    assert blot._grid_dims == [12, 9]
    assert blot._cell_dims == [3, 4]
