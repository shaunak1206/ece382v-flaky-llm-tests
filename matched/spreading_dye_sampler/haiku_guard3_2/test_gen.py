"""Unit tests for spreading_dye_sampler/dye_blot.py (DyeBlot and make_blot)."""

from collections import deque

import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


@pytest.fixture(autouse=True)
def _preserve_numpy_global_random_state():
    """Save and restore numpy's global RNG state so tests leave no trace."""
    saved = np.random.get_state()
    yield
    np.random.set_state(saved)


def _collect(blot):
    """Return the cells of a blot as a list of (x, y) tuples, in visit order."""
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _is_connected(cells):
    """Return True if the cells form one 4-connected group."""
    if not cells:
        return True
    remaining = set(cells)
    start = cells[0]
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nb in remaining and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return seen == remaining


# ---------------------------------------------------------------------------
# DyeBlot construction and for_each_cell
# ---------------------------------------------------------------------------

def test_dyeblot_init_stores_dims_and_starts_empty():
    np.random.seed(1001)
    blot = DyeBlot([4, 5], [1, 2])
    assert blot._grid_dims == [4, 5]
    assert blot._cell_dims == [1, 2]
    assert blot._cells == []


def test_num_cells_is_zero_for_new_blot():
    np.random.seed(1002)
    blot = DyeBlot([10, 10], [30, 30])
    assert blot.num_cells() == 0


def test_for_each_cell_on_empty_blot_calls_nothing():
    np.random.seed(1003)
    blot = DyeBlot([3, 3], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_in_insertion_order():
    np.random.seed(1004)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(2, 3), (0, 1), (3, 3)]
    received = []
    blot.for_each_cell(lambda x, y: received.append((x, y)))
    assert received == [(2, 3), (0, 1), (3, 3)]
    # for_each_cell must not modify the blot.
    assert blot.num_cells() == 3


def test_for_each_cell_propagates_exceptions_from_callback():
    np.random.seed(1005)
    blot = DyeBlot([4, 4], [1, 1])
    blot._cells = [(0, 0), (1, 1)]

    def boom(x, y):
        raise ValueError("callback failure")

    with pytest.raises(ValueError, match="callback failure"):
        blot.for_each_cell(boom)


# ---------------------------------------------------------------------------
# make_blot: deterministic outcomes
# ---------------------------------------------------------------------------

def test_make_blot_request_of_one_cell_returns_single_cell():
    np.random.seed(2001)
    blot = make_blot([10, 10], [30, 30], 1)
    assert blot is not None
    assert blot.num_cells() == 1
    x, y = _collect(blot)[0]
    assert 0 <= x < 10 and 0 <= y < 10


def test_make_blot_1x1_grid_squeeze_true_stops_with_one_cell():
    np.random.seed(2002)
    blot = make_blot([1, 1], [30, 30], 2, squeeze=True)
    assert blot is not None
    assert _collect(blot) == [(0, 0)]


def test_make_blot_1x1_grid_no_squeeze_returns_none_when_growing():
    np.random.seed(2003)
    # The only start cell is (0, 0). Every neighbour lies outside the grid,
    # so the first draw is non-permitted and squeeze=False must return None.
    blot = make_blot([1, 1], [30, 30], 2, squeeze=False)
    assert blot is None


def test_make_blot_2x1_grid_squeeze_fills_both_cells():
    np.random.seed(2004)
    blot = make_blot([2, 1], [30, 30], 2, squeeze=True)
    assert blot is not None
    assert sorted(_collect(blot)) == [(0, 0), (1, 0)]


def test_make_blot_2x1_grid_request_larger_than_grid_caps_at_grid_size():
    np.random.seed(2005)
    blot = make_blot([2, 1], [30, 30], 5, squeeze=True)
    assert blot is not None
    assert sorted(_collect(blot)) == [(0, 0), (1, 0)]


def test_make_blot_returns_none_when_no_start_cell_is_permitted():
    np.random.seed(2006)
    permitted = np.zeros((3, 3), dtype=bool)
    blot = make_blot([3, 3], [30, 30], 4, permitted=permitted, squeeze=True)
    assert blot is None


def test_make_blot_squeeze_false_returns_none_when_masked_neighbour_drawn():
    np.random.seed(2007)
    # Grid is 3 wide, 1 high; middle cell is masked. Any start cell has
    # only non-permitted neighbours, so growing to 2 cells must fail.
    permitted = np.array([[True], [False], [True]])
    blot = make_blot([3, 1], [30, 30], 2, permitted=permitted, squeeze=False)
    assert blot is None


def test_make_blot_squeeze_true_never_includes_masked_cells():
    np.random.seed(2008)
    permitted = np.array([[True], [False], [True]])
    blot = make_blot([3, 1], [30, 30], 3, permitted=permitted, squeeze=True)
    # Either the start was masked (None), or the blot is the single
    # permitted start cell, since its only in-grid neighbour is masked.
    if blot is None:
        return
    cells = _collect(blot)
    assert len(cells) == 1
    assert cells[0] in {(0, 0), (2, 0)}


def test_make_blot_full_grid_squeeze_reaches_exact_requested_size():
    np.random.seed(2009)
    blot = make_blot([6, 6], [30, 30], 10, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 10


def test_make_blot_result_stores_grid_and_cell_dims():
    np.random.seed(2010)
    blot = make_blot([1, 1], [30, 40], 1)
    assert blot is not None
    assert blot._grid_dims == [1, 1]
    assert blot._cell_dims == [30, 40]


# ---------------------------------------------------------------------------
# make_blot: properties that hold for every random outcome
# ---------------------------------------------------------------------------

def test_make_blot_cells_are_unique_and_within_grid():
    np.random.seed(2011)
    blot = make_blot([8, 8], [30, 30], 20, squeeze=True)
    assert blot is not None
    cells = _collect(blot)
    assert len(cells) == len(set(cells)) == 20
    for x, y in cells:
        assert 0 <= x < 8
        assert 0 <= y < 8


def test_make_blot_cells_are_four_connected():
    np.random.seed(2012)
    blot = make_blot([8, 8], [30, 30], 20, squeeze=True)
    assert blot is not None
    assert _is_connected(_collect(blot))


def test_make_blot_without_squeeze_is_none_or_exact_size():
    np.random.seed(2013)
    blot = make_blot([8, 8], [30, 30], 12, squeeze=False)
    assert blot is None or blot.num_cells() == 12


def test_make_blot_squeeze_result_respects_mask_for_every_outcome():
    np.random.seed(2014)
    permitted = np.ones((6, 6), dtype=bool)
    permitted[2, :] = False  # block the entire column x == 2
    blot = make_blot([6, 6], [30, 30], 8, permitted=permitted, squeeze=True)
    if blot is None:
        return
    cells = _collect(blot)
    assert len(cells) == 8
    for x, y in cells:
        assert permitted[x, y]


def test_make_blot_same_seed_gives_identical_blot():
    np.random.seed(2015)
    first = make_blot([8, 8], [30, 30], 15, squeeze=True)

    np.random.seed(2015)
    second = make_blot([8, 8], [30, 30], 15, squeeze=True)

    assert first is not None and second is not None
    assert _collect(first) == _collect(second)


def test_for_each_cell_on_made_blot_matches_internal_cell_list():
    np.random.seed(2016)
    blot = make_blot([8, 8], [30, 30], 10, squeeze=True)
    assert blot is not None
    collected = _collect(blot)
    assert collected == blot._cells
    assert len(collected) == blot.num_cells() == 10
    for x, y in collected:
        assert isinstance(x, int) and isinstance(y, int)
