import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _fix_start(monkeypatch, u, v):
    """Force make_blot's two random() draws to return u then v."""
    values = iter([u, v])
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: next(values))


def _cells_of(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _is_connected(cells):
    cell_set = set(cells)
    if not cell_set:
        return True
    start = next(iter(cell_set))
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in cell_set and n not in visited:
                visited.add(n)
                stack.append(n)
    return visited == cell_set


# ---------------------------------------------------------------------------
# DyeBlot basics and for_each_cell
# ---------------------------------------------------------------------------

def test_new_blot_is_empty():
    blot = DyeBlot([5, 4], [1, 1])
    assert blot.num_cells() == 0


def test_new_blot_stores_dimensions():
    blot = DyeBlot([5, 4], [2, 3])
    assert blot._grid_dims == [5, 4]
    assert blot._cell_dims == [2, 3]


def test_for_each_cell_on_empty_blot_calls_nothing():
    blot = DyeBlot([5, 4], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_visits_cells_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1), (4, 3)]
    visited = _cells_of(blot)
    assert visited == [(0, 0), (1, 0), (1, 1), (4, 3)]


def test_for_each_cell_passes_two_positional_arguments():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(2, 3), (4, 0)]
    received = []
    blot.for_each_cell(lambda *args: received.append(args))
    assert received == [(2, 3), (4, 0)]
    assert all(len(args) == 2 for args in received)


def test_for_each_cell_call_count_matches_num_cells():
    np.random.seed(1)
    blot = make_blot([6, 6], [1, 1], 9, squeeze=True)
    count = []
    blot.for_each_cell(lambda x, y: count.append(1))
    assert len(count) == blot.num_cells() == 9


def test_for_each_cell_propagates_exceptions():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0), (1, 1)]

    def boom(x, y):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        blot.for_each_cell(boom)


# ---------------------------------------------------------------------------
# make_blot: construction on a fully permitted grid
# ---------------------------------------------------------------------------

def test_make_blot_returns_dyeblot_of_requested_size():
    np.random.seed(42)
    blot = make_blot([8, 8], [1, 1], 10, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 10


def test_make_blot_cells_are_unique_and_in_bounds():
    np.random.seed(7)
    grid_w, grid_h = 7, 5
    blot = make_blot([grid_w, grid_h], [1, 1], 20, squeeze=True)
    cells = _cells_of(blot)
    assert len(cells) == len(set(cells))
    for x, y in cells:
        assert 0 <= x < grid_w
        assert 0 <= y < grid_h


def test_make_blot_cells_form_a_connected_region():
    for seed in range(10):
        np.random.seed(seed)
        blot = make_blot([10, 10], [1, 1], 25, squeeze=True)
        assert _is_connected(_cells_of(blot))


def test_make_blot_same_seed_gives_same_cells():
    np.random.seed(123)
    first = _cells_of(make_blot([9, 9], [1, 1], 15, squeeze=True))
    np.random.seed(123)
    second = _cells_of(make_blot([9, 9], [1, 1], 15, squeeze=True))
    assert first == second


def test_make_blot_req_one_returns_only_the_start_cell(monkeypatch):
    # u=0.0 -> cx=0, v=0.0 -> cy=0
    _fix_start(monkeypatch, 0.0, 0.0)
    blot = make_blot([4, 3], [1, 1], 1)
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_start_mapping_uses_floor_of_scaled_draws(monkeypatch):
    # cx = floor(0.999 * 4) = 3, cy = floor(0.5 * 3) = 1
    _fix_start(monkeypatch, 0.999, 0.5)
    blot = make_blot([4, 3], [1, 1], 1)
    assert _cells_of(blot) == [(3, 1)]


def test_full_grid_fill_with_squeeze_covers_every_cell():
    np.random.seed(0)
    blot = make_blot([3, 2], [1, 1], 6, squeeze=True)
    assert set(_cells_of(blot)) == {(x, y) for x in range(3) for y in range(2)}


def test_explicit_all_true_mask_matches_default_mask():
    np.random.seed(99)
    default = _cells_of(make_blot([6, 6], [1, 1], 12, squeeze=True))
    np.random.seed(99)
    explicit = _cells_of(
        make_blot([6, 6], [1, 1], 12, permitted=np.ones((6, 6), dtype=bool), squeeze=True)
    )
    assert default == explicit


# ---------------------------------------------------------------------------
# make_blot: masks, squeeze, and None returns
# ---------------------------------------------------------------------------

def test_make_blot_returns_none_when_start_not_permitted(monkeypatch):
    _fix_start(monkeypatch, 0.0, 0.0)
    mask = np.zeros((3, 3), dtype=bool)
    assert make_blot([3, 3], [1, 1], 4, permitted=mask, squeeze=True) is None


def test_default_squeeze_false_returns_none_when_growth_is_blocked(monkeypatch):
    _fix_start(monkeypatch, 0.0, 0.0)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    assert make_blot([3, 3], [1, 1], 2, permitted=mask) is None


def test_squeeze_false_returns_none_when_blocked_cell_is_drawn(monkeypatch):
    _fix_start(monkeypatch, 0.0, 0.0)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 0] = True  # one permitted neighbour; the other is masked
    # With squeeze=False, drawing any masked neighbour aborts, so the result
    # must be None or contain only permitted cells of the requested size.
    result = None
    for seed in range(30):
        np.random.seed(seed)
        _fix_start(monkeypatch, 0.0, 0.0)
        result = make_blot([3, 3], [1, 1], 2, permitted=mask, squeeze=False)
        if result is not None:
            assert set(_cells_of(result)) <= {(0, 0), (1, 0)}
    # Every call either returned None or a valid blot; no exception occurred.


def test_squeeze_false_never_returns_masked_cells():
    mask = np.zeros((6, 4), dtype=bool)
    mask[:, 0::2] = True  # only even y rows permitted
    for seed in range(20):
        np.random.seed(seed)
        result = make_blot([6, 4], [1, 1], 5, permitted=mask, squeeze=False)
        if result is not None:
            cells = _cells_of(result)
            assert len(cells) == 5
            assert all(mask[x, y] for x, y in cells)


def test_squeeze_true_skips_masked_cells_and_stays_permitted(monkeypatch):
    mask = np.zeros((6, 4), dtype=bool)
    mask[:, 0::2] = True
    _fix_start(monkeypatch, 0.0, 0.0)  # start (0, 0) is permitted
    np.random.seed(5)
    blot = make_blot([6, 4], [1, 1], 8, permitted=mask, squeeze=True)
    assert blot is not None
    cells = _cells_of(blot)
    assert len(cells) <= 8
    assert all(mask[x, y] for x, y in cells)
    assert _is_connected(cells)


def test_squeeze_true_fills_reachable_permitted_region(monkeypatch):
    mask = np.zeros((3, 3), dtype=bool)
    mask[0:2, 0:2] = True  # a 2x2 block; the only reachable region from (0, 0)
    _fix_start(monkeypatch, 0.0, 0.0)
    blot = make_blot([3, 3], [1, 1], 10, permitted=mask, squeeze=True)
    assert set(_cells_of(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_squeeze_true_stops_early_when_no_neighbours_remain(monkeypatch):
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True  # isolated single permitted cell
    _fix_start(monkeypatch, 0.5, 0.5)  # cx=1, cy=1
    blot = make_blot([3, 3], [1, 1], 5, permitted=mask, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(1, 1)]


def test_one_by_one_grid_without_squeeze_returns_none_for_larger_blot(monkeypatch):
    _fix_start(monkeypatch, 0.0, 0.0)
    # The only neighbours are out of bounds, so the first draw aborts.
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_one_by_one_grid_with_squeeze_returns_single_cell(monkeypatch):
    _fix_start(monkeypatch, 0.0, 0.0)
    blot = make_blot([1, 1], [1, 1], 2, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_req_one_always_yields_exactly_one_cell_over_many_seeds():
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([5, 5], [1, 1], 1)
        assert blot is not None
        assert blot.num_cells() == 1
