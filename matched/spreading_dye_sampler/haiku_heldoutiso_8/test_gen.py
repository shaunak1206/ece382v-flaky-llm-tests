import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect(blot):
    """Return the cells of a blot as a list of (x, y) tuples, in iteration order."""
    out = []
    blot.for_each_cell(lambda x, y: out.append((x, y)))
    return out


def _patch_start(monkeypatch, x, y, w, h):
    """Force make_blot's two initial random() draws to select cell (x, y)."""
    values = iter([(x + 0.5) / w, (y + 0.5) / h])
    monkeypatch.setattr(np.random, "random", lambda *a, **k: next(values))


def _patch_choice(monkeypatch, pick):
    """
    Replace numpy.random.choice. `pick` receives the index array and returns
    the chosen index. Returns a list that records the probabilities passed in.
    """
    recorded = []

    def fake_choice(a, p=None):
        recorded.append(None if p is None else np.asarray(p, dtype=float))
        return pick(a)

    monkeypatch.setattr(np.random, "choice", fake_choice)
    return recorded


def _is_connected(cells):
    """Check 4-connectivity of a set of cells via flood fill."""
    if not cells:
        return True
    remaining = set(cells)
    stack = [next(iter(remaining))]
    seen = set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        x, y = c
        for n in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if n in remaining and n not in seen:
                stack.append(n)
    return seen == remaining


# ---------------------------------------------------------------- DyeBlot basics

def test_new_dyeblot_is_empty():
    blot = DyeBlot([10, 10], [2, 2])
    assert blot.num_cells() == 0


def test_new_dyeblot_stores_dims():
    grid = [7, 9]
    cell = [3, 4]
    blot = DyeBlot(grid, cell)
    assert blot._grid_dims is grid
    assert blot._cell_dims is cell


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([5, 5], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_coordinates_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(0, 0), (3, 1), (2, 4)])
    assert _collect(blot) == [(0, 0), (3, 1), (2, 4)]


def test_for_each_cell_count_matches_num_cells():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.extend([(1, 1), (1, 2)])
    assert len(_collect(blot)) == blot.num_cells() == 2


def test_for_each_cell_calls_with_two_positional_args():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells.append((4, 2))
    seen = []

    def f(*args):
        seen.append(args)

    blot.for_each_cell(f)
    assert seen == [(4, 2)]


# ---------------------------------------------------------------- make_blot

def test_make_blot_req_one_gives_single_start_cell(monkeypatch):
    _patch_start(monkeypatch, 0, 0, 3, 3)
    blot = make_blot([3, 3], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert _collect(blot) == [(0, 0)]
    assert blot.num_cells() == 1


def test_make_blot_returns_dyeblot_with_given_dims(monkeypatch):
    _patch_start(monkeypatch, 1, 1, 3, 3)
    blot = make_blot([3, 3], [10, 20], 1)
    assert blot._grid_dims == [3, 3]
    assert blot._cell_dims == [10, 20]


def test_make_blot_neighbour_probability_weighted_by_shared_edge(monkeypatch):
    # cell_dims = [width 10, height 20]. Left/right neighbours share an edge
    # of length cell_height (20); up/down neighbours share cell_width (10).
    _patch_start(monkeypatch, 0, 0, 3, 3)
    # Neighbour order from insertion: (-1,0), (0,-1), (1,0), (0,1).
    recorded = _patch_choice(monkeypatch, pick=lambda a: a[2])  # choose (1,0)

    blot = make_blot([3, 3], [10, 20], 2)

    assert _collect(blot) == [(0, 0), (1, 0)]
    assert len(recorded) == 1
    assert np.allclose(recorded[0], [20 / 60, 10 / 60, 20 / 60, 10 / 60])


def test_make_blot_squeeze_false_returns_none_on_out_of_bounds_draw(monkeypatch):
    _patch_start(monkeypatch, 0, 0, 3, 3)
    # Index 0 is neighbour (-1, 0), which lies outside the grid.
    _patch_choice(monkeypatch, pick=lambda a: a[0])
    assert make_blot([3, 3], [1, 1], 2, squeeze=False) is None


def test_make_blot_squeeze_false_returns_none_on_forbidden_draw(monkeypatch):
    permitted = np.ones((1, 3), dtype=bool)
    permitted[0, 1] = False
    _patch_start(monkeypatch, 0, 0, 1, 3)
    # Index 3 is neighbour (0, 1), which is forbidden.
    _patch_choice(monkeypatch, pick=lambda a: a[3])
    assert make_blot([1, 3], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_squeeze_true_drops_forbidden_cells(monkeypatch):
    permitted = np.ones((1, 3), dtype=bool)
    permitted[0, 1] = False
    _patch_start(monkeypatch, 0, 0, 1, 3)
    # Only neighbour (0,1) is in-grid and it is forbidden. Squeeze removes it,
    # leaving the blot with just the start cell even though 3 were requested.
    blot = make_blot([1, 3], [1, 1], 3, permitted=permitted, squeeze=True)
    assert blot is not None
    assert _collect(blot) == [(0, 0)]


def test_make_blot_returns_none_when_all_cells_forbidden_squeeze_true():
    permitted = np.zeros((2, 2), dtype=bool)
    assert make_blot([2, 2], [1, 1], 1, permitted=permitted, squeeze=True) is None


def test_make_blot_returns_none_when_all_cells_forbidden_squeeze_false():
    permitted = np.zeros((2, 2), dtype=bool)
    assert make_blot([2, 2], [1, 1], 1, permitted=permitted, squeeze=False) is None


def test_make_blot_returns_none_when_start_cell_forbidden(monkeypatch):
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 1] = False
    _patch_start(monkeypatch, 1, 1, 3, 3)
    assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True) is None
    _patch_start(monkeypatch, 1, 1, 3, 3)
    assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_default_mask_squeeze_fills_small_grid():
    # With no mask, a squeezed blot on a 3x2 grid can absorb every cell.
    np.random.seed(0)
    blot = make_blot([3, 2], [1, 1], 100, squeeze=True)
    assert blot is not None
    cells = _collect(blot)
    assert len(cells) == 6
    assert set(cells) == {(x, y) for x in range(3) for y in range(2)}


def test_make_blot_grows_to_requested_size_unique_in_bounds_connected():
    np.random.seed(42)
    blot = make_blot([10, 10], [2, 2], 10, squeeze=True)
    cells = _collect(blot)
    assert len(cells) == 10
    assert len(set(cells)) == 10
    assert all(0 <= x < 10 and 0 <= y < 10 for x, y in cells)
    assert _is_connected(cells)


def test_make_blot_respects_mask_with_squeeze(monkeypatch):
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False
    _patch_start(monkeypatch, 0, 0, 3, 3)
    np.random.seed(7)
    blot = make_blot([3, 3], [1, 1], 5, permitted=permitted, squeeze=True)
    cells = _collect(blot)
    assert blot.num_cells() == 5
    assert len(set(cells)) == 5
    assert (1, 0) not in cells
    assert all(permitted[c] for c in cells)
    assert _is_connected(cells)
