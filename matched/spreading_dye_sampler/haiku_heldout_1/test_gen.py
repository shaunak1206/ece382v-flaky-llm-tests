import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _cells_of(blot):
    collected = []
    blot.for_each_cell(lambda x, y: collected.append((x, y)))
    return collected


def test_dyeblot_init_stores_dims_and_starts_empty():
    blot = DyeBlot([10, 20], [3, 4])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [3, 4]
    assert blot._cells == []
    assert blot.num_cells() == 0


def test_num_cells_on_empty_blot_is_zero():
    assert DyeBlot([5, 5], [1, 1]).num_cells() == 0


def test_for_each_cell_on_empty_blot_calls_nothing():
    calls = []
    DyeBlot([5, 5], [1, 1]).for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_in_insertion_order():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(0, 0), (3, 7), (9, 2)]
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(0, 0), (3, 7), (9, 2)]


def test_for_each_cell_unpacks_arguments_as_two_positional_values():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(4, 5)]
    received = []

    def record(*args):
        received.append(args)

    blot.for_each_cell(record)
    assert received == [(4, 5)]


def test_for_each_cell_does_not_modify_blot():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 1), (2, 2)]
    blot.for_each_cell(lambda x, y: None)
    assert blot._cells == [(1, 1), (2, 2)]


def test_make_blot_returns_dyeblot_instance():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 1)
    assert isinstance(blot, DyeBlot)


def test_make_blot_1x1_requesting_one_cell_returns_origin():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 1)
    assert _cells_of(blot) == [(0, 0)]
    assert blot.num_cells() == 1


def test_make_blot_1x1_squeeze_true_returns_all_available_cells():
    np.random.seed(0)
    blot = make_blot([1, 1], [1, 1], 5, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_1x1_squeeze_false_returns_none_when_out_of_grid_drawn():
    # The only neighbours of (0, 0) are all outside the grid, so the
    # first drawn neighbour is non-permitted and construction must fail.
    np.random.seed(0)
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_returns_none_when_start_cell_not_permitted():
    permitted = np.zeros((3, 3), dtype=bool)
    for seed in range(20):
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 1, permitted=permitted) is None


def test_make_blot_squeeze_true_full_grid_reaches_requested_size():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 20, squeeze=True)
        assert blot is not None
        assert blot.num_cells() == 20


def test_make_blot_cells_are_unique_in_bounds_and_permitted():
    permitted = np.ones((6, 6), dtype=bool)
    permitted[2, :] = False
    permitted[:, 4] = False
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([6, 6], [1, 1], 8, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        cells = _cells_of(blot)
        assert len(cells) == len(set(cells))
        for x, y in cells:
            assert 0 <= x < 6
            assert 0 <= y < 6
            assert permitted[x, y]


def test_make_blot_squeeze_false_never_includes_masked_cells():
    permitted = np.ones((5, 5), dtype=bool)
    permitted[1, 1] = False
    permitted[3, 3] = False
    for seed in range(40):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 4, permitted=permitted, squeeze=False)
        if blot is None:
            continue
        for x, y in _cells_of(blot):
            assert permitted[x, y]


def test_make_blot_column_mask_squeeze_true_caps_size_at_column_length():
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, :] = True  # only column x == 0 is permitted
    successes = 0
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([3, 3], [1, 1], 10, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        successes += 1
        cells = _cells_of(blot)
        assert blot.num_cells() == 3
        assert all(x == 0 for x, _ in cells)
        assert sorted(y for _, y in cells) == [0, 1, 2]
    assert successes > 0


def test_make_blot_column_mask_squeeze_false_returns_none_or_full_column():
    permitted = np.zeros((3, 3), dtype=bool)
    permitted[0, :] = True
    for seed in range(30):
        np.random.seed(seed)
        blot = make_blot([3, 3], [1, 1], 3, permitted=permitted, squeeze=False)
        if blot is None:
            continue
        cells = _cells_of(blot)
        assert len(cells) == 3
        assert all(x == 0 for x, _ in cells)


def test_make_blot_same_seed_gives_same_cells():
    np.random.seed(123)
    first = _cells_of(make_blot([8, 8], [1, 1], 12, squeeze=True))
    np.random.seed(123)
    second = _cells_of(make_blot([8, 8], [1, 1], 12, squeeze=True))
    assert first == second


def test_make_blot_none_permitted_matches_explicit_all_true_mask():
    np.random.seed(7)
    default_cells = _cells_of(make_blot([6, 4], [1, 1], 10, squeeze=True))
    np.random.seed(7)
    explicit_cells = _cells_of(
        make_blot([6, 4], [1, 1], 10,
                  permitted=np.ones((6, 4), dtype=bool), squeeze=True)
    )
    assert default_cells == explicit_cells


def test_make_blot_non_square_grid_respects_width_and_height():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([4, 2], [1, 1], 8, squeeze=True)
        assert blot is not None
        cells = _cells_of(blot)
        assert len(cells) == 8
        assert len(set(cells)) == 8
        for x, y in cells:
            assert 0 <= x < 4
            assert 0 <= y < 2


def test_make_blot_num_cells_matches_for_each_cell_count():
    np.random.seed(99)
    blot = make_blot([10, 10], [2, 2], 15, squeeze=True)
    count = []
    blot.for_each_cell(lambda x, y: count.append(1))
    assert len(count) == blot.num_cells() == 15
