import numpy as np
import numpy.random
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


# ---------- helpers ----------

def _patch_start(monkeypatch, value):
    """Make the random draws used for the start cell deterministic.

    make_blot does `from numpy.random import random` at call time, so
    replacing the attribute on numpy.random takes effect.
    """
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: value)


def _patch_choice(monkeypatch, fn):
    monkeypatch.setattr(np.random, "choice", fn)


def _is_connected(cells):
    cell_set = set(cells)
    if not cell_set:
        return True
    start = next(iter(cell_set))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cell_set)


def _collect_cells(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


# ---------- DyeBlot basics ----------

def test_dyeblot_new_blot_has_no_cells():
    blot = DyeBlot([10, 10], [2, 2])
    assert blot.num_cells() == 0


def test_dyeblot_stores_grid_and_cell_dims():
    blot = DyeBlot([12, 7], [3, 4])
    assert blot._grid_dims == [12, 7]
    assert blot._cell_dims == [3, 4]


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([5, 5], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_then_y_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 4), (3, 1), (2, 2)])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == [(0, 4), (3, 1), (2, 2)]


def test_for_each_cell_calls_once_per_cell_and_count_matches():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(1, 1), (1, 2), (2, 1)])
    count = []
    blot.for_each_cell(lambda x, y: count.append(1))
    assert len(count) == blot.num_cells() == 3


def test_for_each_cell_does_not_modify_blot():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (1, 0)])
    before = list(blot._cells)
    blot.for_each_cell(lambda x, y: None)
    assert blot._cells == before


# ---------- make_blot construction ----------

def test_make_blot_returns_dyeblot_with_given_dims():
    blot = make_blot([6, 6], [2, 3], 4)
    assert isinstance(blot, DyeBlot)
    assert blot._grid_dims == [6, 6]
    assert blot._cell_dims == [2, 3]


def test_make_blot_start_cell_is_first_cell(monkeypatch):
    # random 0.5 on a 5-wide, 5-high grid -> start (2, 2)
    _patch_start(monkeypatch, 0.5)
    blot = make_blot([5, 5], [1, 1], 3, squeeze=True)
    assert _collect_cells(blot)[0] == (2, 2)


def test_make_blot_unconstrained_squeeze_reaches_exact_size():
    blot = make_blot([5, 5], [1, 1], 10, squeeze=True)
    assert blot.num_cells() == 10


def test_make_blot_cells_are_unique():
    blot = make_blot([6, 6], [1, 1], 15, squeeze=True)
    cells = _collect_cells(blot)
    assert len(cells) == len(set(cells))


def test_make_blot_cells_lie_within_grid():
    blot = make_blot([4, 7], [1, 1], 20, squeeze=True)
    for x, y in _collect_cells(blot):
        assert 0 <= x < 4
        assert 0 <= y < 7


def test_make_blot_cells_are_edge_connected():
    blot = make_blot([8, 8], [1, 1], 25, squeeze=True)
    assert _is_connected(_collect_cells(blot))


def test_make_blot_single_cell_request(monkeypatch):
    _patch_start(monkeypatch, 0.5)
    blot = make_blot([3, 3], [1, 1], 1, squeeze=True)
    assert _collect_cells(blot) == [(1, 1)]


def test_make_blot_returns_none_when_start_not_permitted():
    mask = np.zeros((3, 3), dtype=bool)
    assert make_blot([3, 3], [1, 1], 2, permitted=mask, squeeze=True) is None


def test_make_blot_growth_order_is_deterministic_with_fixed_choice(monkeypatch):
    _patch_start(monkeypatch, 0.5)  # start (1, 1)
    _patch_choice(monkeypatch, lambda a, p=None: a[0])
    blot = make_blot([3, 3], [1, 1], 2, squeeze=True)
    # neighbours inserted in order: (0,1), (1,0), (2,1), (1,2)
    assert _collect_cells(blot) == [(1, 1), (0, 1)]


def test_make_blot_draw_probabilities_proportional_to_shared_edge(monkeypatch):
    # cell_dims [2, 3]: left/right neighbours share an edge of height 3,
    # top/bottom neighbours share an edge of width 2.
    _patch_start(monkeypatch, 0.5)  # start (1, 1) on 3x3
    captured = []

    def fake_choice(a, p=None):
        captured.append(np.asarray(p))
        return a[0]

    _patch_choice(monkeypatch, fake_choice)
    make_blot([3, 3], [2, 3], 2, squeeze=True)
    assert len(captured) == 1
    # order (0,1):3, (1,0):2, (2,1):3, (1,2):2 normalised by total 10
    assert np.allclose(captured[0], [0.3, 0.2, 0.3, 0.2])


# ---------- permitted masks and squeeze behaviour ----------

def test_make_blot_squeeze_true_respects_mask(monkeypatch):
    _patch_start(monkeypatch, 0.5)  # start (2, 2) on 4x4
    mask = np.ones((4, 4), dtype=bool)
    mask[0, :] = False  # block x == 0 column
    blot = make_blot([4, 4], [1, 1], 16, permitted=mask, squeeze=True)
    cells = _collect_cells(blot)
    assert len(cells) == 12  # only the x>=1 region is reachable
    assert all(x >= 1 for x, _ in cells)


def test_make_blot_squeeze_false_returns_none_on_forbidden_neighbour(monkeypatch):
    _patch_start(monkeypatch, 0.5)  # start (2, 2) on 4x4
    mask = np.ones((4, 4), dtype=bool)
    mask[0, :] = False
    # Only 12 cells are permitted but 16 are requested, so a blocked
    # neighbour must eventually be drawn, which aborts construction.
    result = make_blot([4, 4], [1, 1], 16, permitted=mask, squeeze=False)
    assert result is None


def test_make_blot_squeeze_false_returns_none_on_out_of_grid_neighbour(monkeypatch):
    _patch_start(monkeypatch, 0.0)  # start corner (0, 0)
    _patch_choice(monkeypatch, lambda a, p=None: a[0])
    # first neighbour inserted is (-1, 0), which lies outside the grid
    assert make_blot([4, 4], [1, 1], 2, squeeze=False) is None


def test_make_blot_squeeze_false_single_cell_ignores_boundary(monkeypatch):
    _patch_start(monkeypatch, 0.5)  # start (1, 1) on 3x3
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    blot = make_blot([3, 3], [1, 1], 1, permitted=mask, squeeze=False)
    assert blot is not None
    assert _collect_cells(blot) == [(1, 1)]


def test_make_blot_unconstrained_no_squeeze_is_none_or_full_size():
    # Without a mask and without squeeze, a boundary draw may abort
    # construction; otherwise the blot must have exactly the requested size.
    np.random.seed(0)
    blot = make_blot([20, 20], [1, 1], 5, squeeze=False)
    assert blot is None or blot.num_cells() == 5


def test_for_each_cell_on_made_blot_visits_every_cell_once():
    blot = make_blot([6, 6], [1, 1], 12, squeeze=True)
    visited = _collect_cells(blot)
    assert len(visited) == blot.num_cells() == 12
    assert len(set(visited)) == 12
