import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect(blot):
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
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in cells and n not in seen:
                seen.add(n)
                stack.append(n)
    return seen == cells


def test_new_blot_is_empty():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0
    assert _collect(blot) == []


def test_for_each_cell_calls_function_in_order_with_coordinates():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells.extend([(1, 2), (3, 4), (5, 6)])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(1, 2), (3, 4), (5, 6)]
    assert blot.num_cells() == 3


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([4, 4], [1, 1])
    called = []
    blot.for_each_cell(lambda x, y: called.append(1))
    assert called == []


def test_for_each_cell_accepts_any_callable_object():
    class Recorder:
        def __init__(self):
            self.total = 0

        def __call__(self, x, y):
            self.total += x + y

    blot = DyeBlot([4, 4], [1, 1])
    blot._cells.extend([(1, 1), (2, 3)])
    rec = Recorder()
    blot.for_each_cell(rec)
    assert rec.total == 7


def test_for_each_cell_propagates_exceptions():
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells.append((0, 0))

    def boom(x, y):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


def test_make_blot_single_cell():
    np.random.seed(0)
    blot = make_blot([5, 5], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    x, y = _collect(blot)[0]
    assert 0 <= x < 5 and 0 <= y < 5


def test_make_blot_has_requested_number_of_cells():
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 10)
        assert blot.num_cells() == 10


def test_make_blot_cells_unique_in_bounds_and_connected():
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([8, 6], [2, 3], 15)
        cells = _collect(blot)
        assert len(cells) == len(set(cells)) == 15
        for x, y in cells:
            assert 0 <= x < 8
            assert 0 <= y < 6
        assert _is_connected(cells)


def test_make_blot_request_larger_than_grid_fills_grid():
    np.random.seed(1)
    blot = make_blot([3, 3], [1, 1], 100)
    assert blot.num_cells() == 9
    assert set(_collect(blot)) == {(x, y) for x in range(3) for y in range(3)}


def test_make_blot_non_square_grid_respects_bounds():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([6, 2], [1, 1], 12)
        assert blot.num_cells() == 12
        for x, y in _collect(blot):
            assert 0 <= x < 6
            assert 0 <= y < 2


def test_make_blot_stores_dimensions():
    np.random.seed(0)
    blot = make_blot([4, 5], [2, 3], 3)
    assert blot._grid_dims == [4, 5]
    assert blot._cell_dims == [2, 3]


def test_make_blot_same_seed_is_reproducible():
    np.random.seed(42)
    a = _collect(make_blot([10, 10], [1, 1], 12))
    np.random.seed(42)
    b = _collect(make_blot([10, 10], [1, 1], 12))
    assert a == b


def test_make_blot_all_forbidden_returns_none():
    permitted = np.zeros((5, 5), dtype=bool)
    for seed in range(10):
        np.random.seed(seed)
        assert make_blot([5, 5], [1, 1], 3, permitted=permitted) is None
        assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_make_blot_no_squeeze_with_single_permitted_cell_always_none():
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True
    for seed in range(30):
        np.random.seed(seed)
        # Either the start is forbidden, or a forbidden neighbour is drawn.
        assert make_blot([3, 3], [1, 1], 2, permitted=permitted) is None


def test_make_blot_squeeze_with_single_permitted_cell():
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[1, 1] = True
    results = []
    for seed in range(100):
        np.random.seed(seed)
        results.append(make_blot([3, 3], [1, 1], 4, permitted=permitted, squeeze=True))
    successes = [r for r in results if r is not None]
    assert successes
    for r in successes:
        assert _collect(r) == [(1, 1)]


def test_make_blot_squeeze_only_uses_permitted_cells():
    permitted = np.zeros((6, 6), dtype=bool)
    permitted[:3, :] = True
    successes = 0
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 8, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        successes += 1
        cells = _collect(blot)
        assert 1 <= len(cells) <= 8
        for x, y in cells:
            assert permitted[x, y]
    assert successes > 0


def test_make_blot_squeeze_fills_connected_permitted_region():
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[1:4, 1:3] = True  # 3x2 block, 6 cells
    expected = {(x, y) for x in range(1, 4) for y in range(1, 3)}
    found = False
    for seed in range(200):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 6, permitted=permitted, squeeze=True)
        if blot is not None:
            found = True
            assert set(_collect(blot)) == expected
    assert found


def test_make_blot_no_squeeze_result_never_includes_forbidden_cells():
    permitted = np.ones((6, 6), dtype=bool)
    permitted[3:, :] = False
    for seed in range(50):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 5, permitted=permitted)
        if blot is not None:
            for x, y in _collect(blot):
                assert permitted[x, y]


def test_make_blot_zero_requested_cells_gives_only_start_cell():
    np.random.seed(3)
    blot = make_blot([5, 5], [1, 1], 0)
    assert blot is not None
    assert blot.num_cells() == 1
