import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _cells_of(blot):
    """Collect the cells of a blot through the public for_each_cell API."""
    collected = []
    blot.for_each_cell(lambda x, y: collected.append((x, y)))
    return collected


def _fix_start(monkeypatch, value):
    """Force make_blot's start-cell draws (numpy.random.random) to a constant."""
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: value)


def _is_connected(cells):
    """Return True if the cells form one 4-connected region."""
    cellset = set(cells)
    start = cells[0]
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in cellset and n not in seen:
                seen.add(n)
                stack.append(n)
    return len(seen) == len(cellset)


def test_num_cells_empty_blot():
    blot = DyeBlot([5, 5], [1, 1])
    assert blot.num_cells() == 0


def test_num_cells_matches_cell_list():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0), (1, 0), (1, 1)]
    assert blot.num_cells() == 3


def test_for_each_cell_visits_cells_in_order():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 2), (2, 1)]
    assert _cells_of(blot) == [(0, 0), (1, 2), (2, 1)]


def test_for_each_cell_empty_blot_makes_no_calls():
    blot = DyeBlot([3, 3], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append((x, y)))
    assert calls == []


def test_for_each_cell_passes_x_and_y_separately():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(2, 1)]
    received = []

    def record(x, y):
        received.append((x, y))

    blot.for_each_cell(record)
    assert received == [(2, 1)]


def test_for_each_cell_does_not_modify_blot():
    blot = DyeBlot([3, 3], [1, 1])
    blot._cells = [(0, 0), (1, 1)]
    blot.for_each_cell(lambda x, y: None)
    assert blot.num_cells() == 2


def test_make_blot_returns_dyeblot_instance():
    blot = make_blot([4, 4], [1, 1], 3)
    assert isinstance(blot, DyeBlot)


def test_make_blot_single_cell_request(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    blot = make_blot([5, 5], [30, 30], 1)
    assert _cells_of(blot) == [(0, 0)]
    assert blot.num_cells() == 1


def test_make_blot_start_at_far_corner(monkeypatch):
    _fix_start(monkeypatch, 0.999)
    blot = make_blot([5, 5], [30, 30], 1)
    assert _cells_of(blot) == [(4, 4)]


def test_make_blot_full_grid_exact_size_when_squeezed():
    np.random.seed(1)
    blot = make_blot([4, 3], [1, 1], 12, squeeze=True)
    cells = _cells_of(blot)
    assert len(cells) == 12
    assert set(cells) == {(x, y) for x in range(4) for y in range(3)}


def test_make_blot_cells_unique_and_in_bounds_non_square():
    np.random.seed(7)
    grid_w, grid_h = 6, 2
    blot = make_blot([grid_w, grid_h], [1, 1], 9, squeeze=True)
    cells = _cells_of(blot)
    assert len(cells) == len(set(cells)) == 9
    for x, y in cells:
        assert 0 <= x < grid_w
        assert 0 <= y < grid_h


def test_make_blot_cells_are_connected():
    np.random.seed(3)
    blot = make_blot([6, 5], [10, 10], 20, squeeze=True)
    assert _is_connected(_cells_of(blot))


def test_make_blot_masked_start_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.ones([4, 4], dtype=bool)
    permitted[0, 0] = False
    assert make_blot([4, 4], [1, 1], 2, permitted=permitted) is None


def test_make_blot_masked_neighbour_without_squeeze_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    # 3x1 grid: start (0,0) is permitted; every neighbour of it is not
    # (the only in-grid neighbour is masked, the others are off-grid).
    permitted = np.array([[True], [False], [True]])
    assert make_blot([3, 1], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_squeeze_skips_masked_and_stops_early(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.array([[True], [False], [True]])
    blot = make_blot([3, 1], [1, 1], 3, permitted=permitted, squeeze=True)
    assert blot is not None
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_squeeze_never_includes_masked_cells(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    # Column x=5 is fully masked, splitting a 10x10 grid into two halves.
    permitted = np.ones([10, 10], dtype=bool)
    permitted[5, :] = False
    blot = make_blot([10, 10], [1, 1], 60, permitted=permitted, squeeze=True)
    cells = _cells_of(blot)
    assert len(cells) == 50  # only the left half (x in 0..4) is reachable
    assert all(x < 5 for x, _ in cells)


def test_make_blot_request_larger_than_grid_without_squeeze_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    # Once the 2x2 grid is full, off-grid neighbours are drawn and rejected.
    assert make_blot([2, 2], [1, 1], 5, squeeze=False) is None


def test_make_blot_request_larger_than_grid_with_squeeze_returns_whole_grid(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    blot = make_blot([2, 2], [1, 1], 5, squeeze=True)
    assert set(_cells_of(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_make_blot_same_seed_gives_same_blot():
    np.random.seed(42)
    first = _cells_of(make_blot([8, 8], [1, 1], 15, squeeze=True))
    np.random.seed(42)
    second = _cells_of(make_blot([8, 8], [1, 1], 15, squeeze=True))
    assert first == second


def test_make_blot_neighbour_weights_follow_shared_edge_length(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    recorded = {}

    def fake_choice(indices, p=None):
        recorded["indices"] = np.asarray(indices)
        recorded["p"] = np.asarray(p)
        return indices[3]  # pick the neighbour at (0, 1)

    monkeypatch.setattr(np.random, "choice", fake_choice)

    # cell_dims = [30, 10]: left/right neighbours share an edge of height 10,
    # top/bottom neighbours share an edge of width 30.
    blot = make_blot([5, 5], [30, 10], 2, squeeze=False)

    # Neighbour order: (-1,0), (0,-1), (1,0), (0,1)
    assert np.allclose(recorded["p"], [10 / 80, 30 / 80, 10 / 80, 30 / 80])
    assert _cells_of(blot) == [(0, 0), (0, 1)]


def test_make_blot_default_permitted_matches_explicit_all_true(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    np.random.seed(11)
    default_cells = _cells_of(make_blot([4, 4], [1, 1], 10, squeeze=True))
    np.random.seed(11)
    explicit = np.ones([4, 4], dtype=bool)
    explicit_cells = _cells_of(
        make_blot([4, 4], [1, 1], 10, permitted=explicit, squeeze=True)
    )
    assert default_cells == explicit_cells


def test_make_blot_one_by_one_grid_squeeze_returns_single_cell(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    blot = make_blot([1, 1], [1, 1], 3, squeeze=True)
    assert _cells_of(blot) == [(0, 0)]


def test_make_blot_one_by_one_grid_no_squeeze_request_above_one_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None
