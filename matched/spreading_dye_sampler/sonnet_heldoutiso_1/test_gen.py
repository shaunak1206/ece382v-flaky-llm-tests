import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def collect_cells(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def is_connected(cells):
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


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([10, 10], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_calls_in_order_with_x_y_arguments():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells.extend([(1, 2), (3, 4), (5, 6)])
    assert blot.num_cells() == 3
    assert collect_cells(blot) == [(1, 2), (3, 4), (5, 6)]


def test_for_each_cell_returns_none():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells.append((0, 0))
    assert blot.for_each_cell(lambda x, y: 42) is None


def test_make_blot_returns_blot_with_requested_size():
    np.random.seed(1)
    blot = make_blot([50, 50], [1, 1], 10, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 10
    assert len(collect_cells(blot)) == 10


def test_make_blot_single_cell():
    np.random.seed(2)
    blot = make_blot([20, 20], [1, 1], 1)
    assert blot.num_cells() == 1


def test_cells_are_unique_in_bounds_and_connected():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([30, 20], [2, 3], 25, squeeze=True)
        assert blot is not None
        cells = collect_cells(blot)
        assert len(cells) == 25
        assert len(set(cells)) == 25
        for x, y in cells:
            assert 0 <= x < 30
            assert 0 <= y < 20
        assert is_connected(cells)


def test_non_square_grid_respects_width_and_height():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([6, 2], [1, 1], 12, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 12
        for x, y in collect_cells(blot):
            assert 0 <= x < 6
            assert 0 <= y < 2


def test_squeeze_fills_whole_small_grid():
    for seed in range(5):
        np.random.seed(seed)
        blot = make_blot([3, 3], [1, 1], 9, squeeze=True)
        assert blot is not None
        assert set(collect_cells(blot)) == {(x, y) for x in range(3) for y in range(3)}


def test_request_larger_than_grid_with_squeeze_yields_whole_grid():
    np.random.seed(3)
    blot = make_blot([3, 3], [1, 1], 20, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 9


def test_single_cell_grid_without_squeeze_returns_none_when_more_requested():
    np.random.seed(4)
    assert make_blot([1, 1], [1, 1], 5, squeeze=False) is None


def test_single_cell_grid_with_squeeze_returns_one_cell():
    np.random.seed(4)
    blot = make_blot([1, 1], [1, 1], 5, squeeze=True)
    assert blot is not None
    assert collect_cells(blot) == [(0, 0)]


@pytest.mark.parametrize("squeeze", [True, False])
def test_fully_masked_grid_returns_none(squeeze):
    np.random.seed(5)
    permitted = np.zeros((10, 10), dtype=bool)
    assert make_blot([10, 10], [1, 1], 5, permitted=permitted, squeeze=squeeze) is None


def test_squeeze_with_mask_only_uses_permitted_cells():
    permitted = np.zeros((10, 10), dtype=bool)
    permitted[:5, :] = True
    successes = 0
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 20, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        successes += 1
        assert blot.num_cells() == 20
        for x, y in collect_cells(blot):
            assert permitted[x, y]
    assert successes > 0


def test_no_squeeze_with_mask_never_uses_forbidden_cells():
    permitted = np.zeros((10, 10), dtype=bool)
    permitted[:5, :] = True
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 20, permitted=permitted, squeeze=False)
        if blot is None:
            continue
        for x, y in collect_cells(blot):
            assert permitted[x, y]


def test_squeeze_limited_by_size_of_permitted_region():
    permitted = np.zeros((10, 10), dtype=bool)
    permitted[4:6, 4:6] = True
    successes = 0
    for seed in range(60):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 10, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        successes += 1
        assert set(collect_cells(blot)) == {(4, 4), (4, 5), (5, 4), (5, 5)}
    assert successes > 0


def test_blot_stores_nothing_extra_after_none_result():
    # A None result for a masked start must not raise.
    permitted = np.zeros((4, 4), dtype=bool)
    result = make_blot([4, 4], [1, 1], 3, permitted=permitted)
    assert result is None


def test_cell_dimensions_bias_growth_direction():
    # Horizontal neighbours share an edge of length cell_height, so a very
    # tall cell makes horizontal growth far more likely than vertical.
    horizontal = 0
    trials = 50
    for seed in range(trials):
        np.random.seed(seed)
        blot = make_blot([1000, 1000], [1, 1000], 2, squeeze=True)
        assert blot is not None
        cells = collect_cells(blot)
        assert len(cells) == 2
        if cells[0][1] == cells[1][1]:
            horizontal += 1
    assert horizontal >= 40
