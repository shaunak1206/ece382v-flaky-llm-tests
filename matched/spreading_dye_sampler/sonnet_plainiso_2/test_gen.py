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


@pytest.fixture
def fixed_start(monkeypatch):
    """Patch numpy.random.random so the start cell is predictable."""
    def _set(value):
        monkeypatch.setattr(np.random, "random", lambda: value)
    return _set


@pytest.fixture
def first_choice(monkeypatch):
    """Patch numpy.random.choice to always pick the first candidate."""
    monkeypatch.setattr(np.random, "choice", lambda indices, p=None: indices[0])


def test_dyeblot_new_is_empty():
    blot = DyeBlot([10, 10], [1, 1])
    assert blot.num_cells() == 0
    assert collect_cells(blot) == []


def test_dyeblot_stores_dimensions():
    blot = DyeBlot([7, 9], [2, 3])
    assert blot._grid_dims == [7, 9]
    assert blot._cell_dims == [2, 3]


def test_for_each_cell_visits_in_insertion_order():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(2, 3), (0, 1), (4, 4)]
    assert collect_cells(blot) == [(2, 3), (0, 1), (4, 4)]
    assert blot.num_cells() == 3


def test_for_each_cell_passes_x_then_y_as_positional_args():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(1, 2)]
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))

    blot.for_each_cell(record)
    assert calls == [((1, 2), {})]


def test_for_each_cell_on_empty_blot_never_calls_function():
    blot = DyeBlot([5, 5], [1, 1])
    calls = []
    blot.for_each_cell(lambda x, y: calls.append(1))
    assert calls == []


def test_for_each_cell_can_be_called_repeatedly():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0), (1, 0)]
    first = collect_cells(blot)
    second = collect_cells(blot)
    assert first == second == [(0, 0), (1, 0)]


def test_for_each_cell_propagates_exceptions_from_callback():
    blot = DyeBlot([5, 5], [1, 1])
    blot._cells = [(0, 0)]

    def boom(x, y):
        raise ValueError("bad cell")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


def test_make_blot_single_cell_grid_single_cell_request():
    blot = make_blot([1, 1], [1, 1], 1)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert collect_cells(blot) == [(0, 0)]


def test_make_blot_zero_requested_cells_still_has_start_cell():
    blot = make_blot([1, 1], [1, 1], 0)
    assert blot is not None
    assert collect_cells(blot) == [(0, 0)]


def test_make_blot_start_cell_from_random(fixed_start):
    fixed_start(0.999)
    blot = make_blot([4, 3], [1, 1], 1)
    assert collect_cells(blot) == [(3, 2)]


def test_make_blot_records_dims_on_blot(fixed_start):
    fixed_start(0.0)
    blot = make_blot([4, 3], [2, 5], 1)
    assert blot._grid_dims == [4, 3]
    assert blot._cell_dims == [2, 5]


def test_make_blot_non_squeeze_hits_grid_edge_returns_none():
    # A 1x1 grid cannot hold two cells; the first neighbour drawn is outside.
    assert make_blot([1, 1], [1, 1], 2, squeeze=False) is None


def test_make_blot_squeeze_stops_when_grid_exhausted():
    np.random.seed(0)
    blot = make_blot([2, 2], [1, 1], 10, squeeze=True)
    assert blot is not None
    assert blot.num_cells() == 4
    assert set(collect_cells(blot)) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_make_blot_squeeze_first_choice_deterministic_cells(fixed_start, first_choice):
    fixed_start(0.0)
    blot = make_blot([3, 3], [1, 1], 3, squeeze=True)
    assert collect_cells(blot) == [(0, 0), (1, 0), (0, 1)]


def test_make_blot_non_squeeze_first_choice_hits_out_of_bounds(fixed_start, first_choice):
    fixed_start(0.0)
    # First neighbour of (0, 0) is (-1, 0), which is outside the grid.
    assert make_blot([3, 3], [1, 1], 3, squeeze=False) is None


def test_make_blot_start_not_permitted_returns_none(fixed_start):
    fixed_start(0.9)
    permitted = np.ones((2, 2), dtype=bool)
    permitted[1, 1] = False
    assert make_blot([2, 2], [1, 1], 1, permitted=permitted) is None
    assert make_blot([2, 2], [1, 1], 1, permitted=permitted, squeeze=True) is None


def test_make_blot_nothing_permitted_returns_none():
    permitted = np.zeros((3, 3), dtype=bool)
    for seed in range(5):
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True) is None
        np.random.seed(seed)
        assert make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False) is None


def test_make_blot_edge_length_weights_use_cell_dims(monkeypatch, fixed_start):
    fixed_start(0.0)
    seen = {}

    def fake_choice(indices, p=None):
        seen["p"] = list(p)
        seen["n"] = len(indices)
        return indices[-1]

    monkeypatch.setattr(np.random, "choice", fake_choice)
    make_blot([5, 5], [2, 5], 2, squeeze=True)
    # Neighbours of (0, 0) in insertion order: left/right share height (5), up/down share width (2).
    assert seen["n"] == 4
    assert seen["p"] == pytest.approx([5 / 14.0, 2 / 14.0, 5 / 14.0, 2 / 14.0])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_make_blot_squeeze_produces_connected_unique_in_bounds_cells(seed):
    np.random.seed(seed)
    blot = make_blot([5, 4], [1, 1], 7, squeeze=True)
    assert blot is not None
    cells = collect_cells(blot)
    assert len(cells) == 7
    assert blot.num_cells() == 7
    assert len(set(cells)) == len(cells)
    assert all(0 <= x < 5 and 0 <= y < 4 for x, y in cells)
    assert is_connected(cells)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_make_blot_squeeze_respects_mask(fixed_start, seed):
    fixed_start(0.0)
    np.random.seed(seed)
    permitted = np.zeros((4, 4), dtype=bool)
    permitted[0:2, 0:2] = True
    blot = make_blot([4, 4], [1, 1], 10, permitted=permitted, squeeze=True)
    assert blot is not None
    assert set(collect_cells(blot)) == {(0, 0), (1, 0), (0, 1), (1, 1)}
