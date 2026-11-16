import numpy as np
import pytest

from spreading_dye_sampler import dye_blot
from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


def _collect(blot):
    cells = []
    blot.for_each_cell(lambda x, y: cells.append((x, y)))
    return cells


def _fix_start(monkeypatch, value):
    """Force the starting cell choice: x = y = floor(value * dim)."""
    monkeypatch.setattr(np.random, "random", lambda: value)


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
    blot._cells = [(1, 2), (3, 4), (0, 0)]
    assert blot.num_cells() == 3
    assert _collect(blot) == [(1, 2), (3, 4), (0, 0)]


def test_for_each_cell_propagates_exceptions():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 1)]

    def boom(x, y):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        blot.for_each_cell(boom)


def test_dims_are_stored():
    blot = DyeBlot([12, 7], [3, 4])
    assert blot._grid_dims == [12, 7]
    assert blot._cell_dims == [3, 4]


def test_make_blot_returns_requested_number_of_cells():
    _fix_start(pytest.MonkeyPatch() if False else _Dummy(), 0.5) if False else None
    np.random.seed(0)
    blot = make_blot([101, 101], [1, 1], 20, squeeze=True)
    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 20


def test_for_each_cell_on_made_blot_visits_every_cell():
    np.random.seed(1)
    blot = make_blot([50, 50], [2, 3], 15, squeeze=True)
    cells = _collect(blot)
    assert len(cells) == blot.num_cells() == 15


def test_cells_are_unique_and_in_bounds():
    np.random.seed(2)
    blot = make_blot([30, 20], [1, 1], 40, squeeze=True)
    cells = _collect(blot)
    assert len(set(cells)) == len(cells)
    for x, y in cells:
        assert 0 <= x < 30
        assert 0 <= y < 20


def test_blot_is_edge_connected():
    np.random.seed(3)
    blot = make_blot([40, 40], [1, 1], 30, squeeze=True)
    cells = _collect(blot)
    for i in range(1, len(cells)):
        x, y = cells[i]
        assert any(abs(x - px) + abs(y - py) == 1 for px, py in cells[:i])


def test_single_cell_blot(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    blot = make_blot([101, 101], [1, 1], 1)
    assert blot.num_cells() == 1
    assert _collect(blot) == [(50, 50)]


def test_zero_requested_cells_still_places_start_cell(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    blot = make_blot([11, 11], [1, 1], 0)
    assert blot.num_cells() == 1


def test_unsqueezed_blot_in_grid_interior_succeeds(monkeypatch):
    _fix_start(monkeypatch, 0.5)
    np.random.seed(4)
    blot = make_blot([101, 101], [1, 1], 5, squeeze=False)
    assert blot is not None
    assert blot.num_cells() == 5


def test_one_by_one_grid(monkeypatch):
    blot = make_blot([1, 1], [1, 1], 1)
    assert blot.num_cells() == 1
    assert _collect(blot) == [(0, 0)]


def test_squeeze_fills_small_grid_when_request_too_large():
    np.random.seed(5)
    blot = make_blot([2, 2], [1, 1], 10, squeeze=True)
    assert blot is not None
    assert sorted(_collect(blot)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_no_squeeze_returns_none_when_request_exceeds_grid():
    np.random.seed(6)
    assert make_blot([2, 2], [1, 1], 10, squeeze=False) is None


def test_non_square_grid_uses_width_then_height():
    np.random.seed(7)
    blot = make_blot([6, 3], [1, 1], 18, squeeze=True)
    cells = _collect(blot)
    assert sorted(cells) == sorted((x, y) for x in range(6) for y in range(3))


@pytest.mark.parametrize("squeeze", [True, False])
def test_all_forbidden_mask_returns_none(squeeze):
    permitted = np.zeros((10, 10), dtype=bool)
    np.random.seed(8)
    assert make_blot([10, 10], [1, 1], 5, permitted=permitted, squeeze=squeeze) is None


def test_unpermitted_start_returns_none_even_with_squeeze(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.ones((10, 10), dtype=bool)
    permitted[0, 0] = False
    assert make_blot([10, 10], [1, 1], 3, permitted=permitted, squeeze=True) is None


def test_squeeze_with_single_permitted_cell_gives_one_cell(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[0, 0] = True
    blot = make_blot([5, 5], [1, 1], 5, permitted=permitted, squeeze=True)
    assert blot is not None
    assert _collect(blot) == [(0, 0)]


def test_no_squeeze_hitting_masked_cell_returns_none(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((5, 5), dtype=bool)
    permitted[0, 0] = True
    assert make_blot([5, 5], [1, 1], 3, permitted=permitted, squeeze=False) is None


def test_squeeze_respects_mask_cells():
    permitted = np.zeros((20, 20), dtype=bool)
    permitted[:10, :] = True
    found = 0
    for seed in range(20):
        np.random.seed(seed)
        blot = make_blot([20, 20], [1, 1], 25, permitted=permitted, squeeze=True)
        if blot is None:
            continue
        found += 1
        assert blot.num_cells() == 25
        for x, y in _collect(blot):
            assert permitted[x, y]
    assert found > 0


def test_squeeze_cannot_exceed_number_of_permitted_connected_cells(monkeypatch):
    _fix_start(monkeypatch, 0.0)
    permitted = np.zeros((6, 6), dtype=bool)
    permitted[0:2, 0:2] = True
    np.random.seed(9)
    blot = make_blot([6, 6], [1, 1], 20, permitted=permitted, squeeze=True)
    assert sorted(_collect(blot)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
