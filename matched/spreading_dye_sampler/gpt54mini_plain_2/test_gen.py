import numpy as np
import pytest

from spreading_dye_sampler.dye_blot import DyeBlot, make_blot


class DummyRandom:
    def __init__(self, values):
        self._values = list(values)
        self.calls = 0

    def __call__(self):
        value = self._values[self.calls]
        self.calls += 1
        return value


def patch_random(monkeypatch, values):
    dummy = DummyRandom(values)
    monkeypatch.setattr("numpy.random.random", dummy)
    return dummy


def patch_choice(monkeypatch, selector):
    def _choice(indices, p=None):
        return selector(indices, p)

    monkeypatch.setattr("numpy.random.choice", _choice)


def test_dyeblot_initial_state():
    blot = DyeBlot([10, 20], [2, 3])
    assert blot._grid_dims == [10, 20]
    assert blot._cell_dims == [2, 3]
    assert blot.num_cells() == 0


def test_for_each_cell_calls_function_for_all_cells_in_order():
    blot = DyeBlot([10, 10], [1, 1])
    blot._cells = [(1, 2), (3, 4), (5, 6)]
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == [(1, 2), (3, 4), (5, 6)]


def test_for_each_cell_with_no_cells_does_not_call_function():
    blot = DyeBlot([5, 5], [1, 1])
    seen = []

    blot.for_each_cell(lambda x, y: seen.append((x, y)))

    assert seen == []


def test_make_blot_single_cell_without_permitted_mask(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.7])  # start at (0, 0) in a 3x3 grid
    blot = make_blot([3, 3], [2, 4], 1)

    assert isinstance(blot, DyeBlot)
    assert blot.num_cells() == 1
    assert blot._cells == [(0, 2)]


def test_make_blot_returns_none_when_start_cell_is_not_permitted(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.7])  # start at (0, 2)
    permitted = np.ones((3, 3), dtype=bool)
    permitted[0, 2] = False

    blot = make_blot([3, 3], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_constructs_multiple_cells_with_choice(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)

    def selector(indices, p=None):
        # After starting at (0, 0), the available neighbors should be:
        # (1, 0) and (0, 1). Choose the first available every time.
        return indices[0]

    patch_choice(monkeypatch, selector)

    blot = make_blot([3, 3], [2, 3], 2)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (1, 0)]


def test_make_blot_chooses_neighbor_by_edge_length_weights(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)

    seen_p = []

    def selector(indices, p=None):
        seen_p.append(np.array(p))
        return indices[0]

    patch_choice(monkeypatch, selector)

    blot = make_blot([3, 3], [2, 5], 2)

    assert blot is not None
    assert blot._cells == [(0, 0), (1, 0)]
    assert len(seen_p) == 1
    np.testing.assert_allclose(seen_p[0], np.array([2 / 7, 5 / 7]))


def test_make_blot_rejects_permitted_masked_neighbor_when_squeeze_false(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False  # make the first chosen neighbor invalid

    def selector(indices, p=None):
        return indices[0]  # choose (1, 0)

    patch_choice(monkeypatch, selector)

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=False)

    assert blot is None


def test_make_blot_squeeze_true_skips_masked_neighbor_and_continues(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)
    permitted = np.ones((3, 3), dtype=bool)
    permitted[1, 0] = False  # first neighbor invalid

    # First draw invalid (1, 0), second draw valid (0, 1)
    call_count = {"n": 0}

    def selector(indices, p=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return indices[0]  # (1, 0)
        return indices[0]      # after deletion, remaining first is (0, 1)

    patch_choice(monkeypatch, selector)

    blot = make_blot([3, 3], [1, 1], 2, permitted=permitted, squeeze=True)

    assert blot is not None
    assert blot.num_cells() == 2
    assert blot._cells == [(0, 0), (0, 1)]


def test_make_blot_with_req_num_cells_one_never_draws_neighbors(monkeypatch):
    patch_random(monkeypatch, [0.1, 0.1])
    choice_calls = {"n": 0}

    def selector(indices, p=None):
        choice_calls["n"] += 1
        return indices[0]

    patch_choice(monkeypatch, selector)

    blot = make_blot([4, 4], [1, 1], 1)

    assert blot is not None
    assert blot.num_cells() == 1
    assert choice_calls["n"] == 0


def test_make_blot_respects_grid_boundaries(monkeypatch):
    # Start at (0, 0); neighbors outside the grid should never appear.
    patch_random(monkeypatch, [0.01, 0.01])

    def selector(indices, p=None):
        # Only in-bounds neighbors should be present: (1, 0), (0, 1)
        assert all(tuple(idx) not in [(-1, 0), (0, -1)] for idx in indices)
        return indices[0]

    patch_choice(monkeypatch, selector)

    blot = make_blot([2, 2], [1, 1], 2)

    assert blot is not None
    assert blot._cells[0] == (0, 0)
    assert blot._cells[1] in {(1, 0), (0, 1)}


def test_make_blot_returns_none_when_start_outside_permitted_mask_due_to_mask_false(monkeypatch):
    patch_random(monkeypatch, [0.8, 0.8])  # start at (1, 1) in 2x2
    permitted = np.zeros((2, 2), dtype=bool)

    blot = make_blot([2, 2], [1, 1], 1, permitted=permitted)

    assert blot is None


def test_make_blot_can_return_large_blot_without_duplicates(monkeypatch):
    patch_random(monkeypatch, [0.2, 0.2])  # start at (0, 0)
    # Deterministically pick the first available neighbor each time.
    patch_choice(monkeypatch, lambda indices, p=None: indices[0])

    blot = make_blot([3, 3], [1, 1], 4)

    assert blot is not None
    assert blot.num_cells() == 4
    assert len(set(blot._cells)) == 4
    assert blot._cells[0] == (0, 0)
