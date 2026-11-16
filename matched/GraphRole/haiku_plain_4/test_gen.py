import numpy as np
import pandas as pd
import pytest

from graphrole.roles import extract as extract_mod
from graphrole.roles.extract import RoleExtractor


def make_features(n_rows=5, n_cols=3):
    return pd.DataFrame(
        np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols),
        index=[f"node_{i}" for i in range(n_rows)],
        columns=[f"feat_{j}" for j in range(n_cols)],
    )


# ---------------------------------------------------------------------------
# __init__ and properties
# ---------------------------------------------------------------------------

def test_init_defaults():
    extractor = RoleExtractor()
    assert extractor.n_roles is None
    assert (extractor.min_roles, extractor.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (extractor.min_bits, extractor.max_bits) == RoleExtractor.N_BIT_RANGE
    assert extractor.node_role_factor is None
    assert extractor.role_feature_factor is None


def test_init_custom_ranges_and_n_roles():
    extractor = RoleExtractor(n_roles=4, n_role_range=(3, 5), n_bit_range=(2, 6))
    assert extractor.n_roles == 4
    assert (extractor.min_roles, extractor.max_roles) == (3, 5)
    assert (extractor.min_bits, extractor.max_bits) == (2, 6)


def test_roles_is_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_is_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_roles_maps_each_node_to_argmax_role():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [0.8, 0.2]],
        index=["x", "y"],
        columns=["role_0", "role_1"],
    )
    assert extractor.roles == {"x": "role_1", "y": "role_0"}


def test_role_percentage_rows_sum_to_one():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=["x", "y"],
        columns=["role_0", "role_1"],
    )
    pct = extractor.role_percentage
    np.testing.assert_allclose(pct.loc["x"].values, [0.25, 0.75])
    np.testing.assert_allclose(pct.loc["y"].values, [0.5, 0.5])
    np.testing.assert_allclose(pct.sum(axis=1).values, [1.0, 1.0])


# ---------------------------------------------------------------------------
# extract_role_factors
# ---------------------------------------------------------------------------

def test_extract_with_n_roles_uses_fixed_roles_and_log2_bits(monkeypatch):
    features = make_features(n_rows=5, n_cols=3)
    calls = []

    def fake_encoded(feats, n_roles, n_bits):
        calls.append((feats, n_roles, n_bits))
        return np.ones((5, n_roles)), np.ones((n_roles, 3))

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))

    extractor = RoleExtractor(n_roles=2)
    extractor.extract_role_factors(features)

    assert len(calls) == 1
    feats, n_roles, n_bits = calls[0]
    assert feats is features
    assert n_roles == 2
    # int(log2(2 * min(5, 3))) = int(log2(6)) = 2
    assert n_bits == 2


def test_extract_builds_factor_frames_with_labels_and_indices(monkeypatch):
    features = make_features(n_rows=5, n_cols=3)

    def fake_encoded(feats, n_roles, n_bits):
        G = np.arange(5 * n_roles, dtype=float).reshape(5, n_roles)
        F = np.arange(n_roles * 3, dtype=float).reshape(n_roles, 3)
        return G, F

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))

    extractor = RoleExtractor(n_roles=2)
    extractor.extract_role_factors(features)

    pd.testing.assert_index_equal(extractor.node_role_factor.index, features.index)
    assert list(extractor.node_role_factor.columns) == ["role_0", "role_1"]
    assert list(extractor.role_feature_factor.index) == ["role_0", "role_1"]
    pd.testing.assert_index_equal(extractor.role_feature_factor.columns, features.columns)
    assert extractor.node_role_factor.shape == (5, 2)
    assert extractor.role_feature_factor.shape == (2, 3)


def test_extract_without_n_roles_delegates_to_select_model(monkeypatch):
    features = make_features()
    seen = []

    def fake_select(self, feats):
        seen.append(feats)
        return np.ones((5, 3)), np.ones((3, 3))

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select)

    extractor = RoleExtractor()
    extractor.extract_role_factors(features)

    assert len(seen) == 1 and seen[0] is features
    assert extractor.node_role_factor.shape == (5, 3)
    assert list(extractor.node_role_factor.columns) == ["role_0", "role_1", "role_2"]


def test_extract_with_n_roles_does_not_call_select_model(monkeypatch):
    features = make_features()

    def fail_select(self, feats):
        raise AssertionError("_select_model should not be called when n_roles is set")

    def fake_encoded(feats, n_roles, n_bits):
        return np.ones((5, n_roles)), np.ones((n_roles, 3))

    monkeypatch.setattr(RoleExtractor, "_select_model", fail_select)
    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))

    RoleExtractor(n_roles=3).extract_role_factors(features)


def test_explain_is_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        RoleExtractor().explain()


# ---------------------------------------------------------------------------
# _rescale_costs
# ---------------------------------------------------------------------------

def test_rescale_costs_normalizes_each_row_by_its_norm():
    costs = np.array([[3.0, 4.0], [6.0, 8.0]])
    result = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.6, 0.8]])


def test_rescale_costs_ignores_nan_entries():
    costs = np.array([[3.0, np.nan], [6.0, 8.0]])
    result = RoleExtractor._rescale_costs(costs)
    # row 0 norm is sqrt(9) = 3 when nan is skipped
    np.testing.assert_allclose(result, [[1.0, np.nan], [0.6, 0.8]])


# ---------------------------------------------------------------------------
# _get_encoded_role_factors
# ---------------------------------------------------------------------------

def test_get_encoded_role_factors_passes_values_and_roles_to_nmf(monkeypatch):
    features = make_features(n_rows=4, n_cols=2)
    seen = {}

    def fake_nmf(V, n_roles):
        seen["V"] = V
        seen["n_roles"] = n_roles
        return np.ones((4, n_roles)), np.ones((n_roles, 2))

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr(extract_mod, "encode", lambda M, n_bins: M)

    RoleExtractor._get_encoded_role_factors(features, 3, 2)

    np.testing.assert_array_equal(seen["V"], features.values)
    assert seen["n_roles"] == 3


def test_get_encoded_role_factors_uses_two_to_the_bits_bins(monkeypatch):
    features = make_features(n_rows=4, n_cols=2)
    bins_seen = []

    monkeypatch.setattr(
        extract_mod, "get_nmf_decomposition",
        lambda V, n_roles: (np.ones((4, n_roles)), np.ones((n_roles, 2))),
    )

    def fake_encode(M, n_bins):
        bins_seen.append(n_bins)
        return M

    monkeypatch.setattr(extract_mod, "encode", fake_encode)

    RoleExtractor._get_encoded_role_factors(features, 2, 3)

    assert bins_seen == [8, 8]


def test_get_encoded_role_factors_returns_encoded_g_then_f(monkeypatch):
    features = make_features(n_rows=4, n_cols=2)
    G = np.full((4, 2), 1.0)
    F = np.full((2, 2), 2.0)

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", lambda V, n_roles: (G, F))
    monkeypatch.setattr(extract_mod, "encode", lambda M, n_bins: M * 10)

    G_encoded, F_encoded = RoleExtractor._get_encoded_role_factors(features, 2, 2)

    np.testing.assert_array_equal(G_encoded, G * 10)
    np.testing.assert_array_equal(F_encoded, F * 10)


# ---------------------------------------------------------------------------
# _select_model
# ---------------------------------------------------------------------------

def _model_encoder(features, n_roles, n_bits):
    """Model tagged so roles = G.shape[1] and bits = F[0, 0]."""
    G = np.zeros((features.shape[0], n_roles))
    F = np.full((n_roles, features.shape[1]), float(n_bits))
    return G, F


def _cost_by_bits(features, model):
    """Encoding cost grows with bits; error cost is constant."""
    G, F = model
    bits = int(F[0, 0])
    encoding_cost = {1: 1.0, 2: 2.0, 3: 3.0}[bits]
    error_cost = 1.0
    return encoding_cost, error_cost


def test_select_model_picks_minimum_combined_cost(monkeypatch):
    # features with 2 rows -> only roles=2 is in the grid
    features = pd.DataFrame(np.zeros((2, 4)))
    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(_model_encoder))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", _cost_by_bits)

    extractor = RoleExtractor(n_bit_range=(1, 3))
    G, F = extractor._select_model(features)

    assert G.shape[1] == 2
    assert int(F[0, 0]) == 1


def test_select_model_skips_combinations_raising_value_error(monkeypatch):
    features = pd.DataFrame(np.zeros((2, 4)))

    def encoder_raising_for_one_bit(feats, n_roles, n_bits):
        if n_bits == 1:
            raise ValueError("too many bits for number of samples")
        return _model_encoder(feats, n_roles, n_bits)

    monkeypatch.setattr(
        RoleExtractor, "_get_encoded_role_factors",
        staticmethod(encoder_raising_for_one_bit),
    )
    monkeypatch.setattr(extract_mod, "get_description_length_costs", _cost_by_bits)

    extractor = RoleExtractor(n_bit_range=(1, 3))
    G, F = extractor._select_model(features)

    # bits=1 was skipped, so the next cheapest (bits=2) must be chosen
    assert int(F[0, 0]) == 2


def test_select_model_roles_grid_capped_by_feature_count(monkeypatch):
    features = pd.DataFrame(np.zeros((3, 6)))  # min(shape) = 3
    roles_seen = set()

    def encoder(feats, n_roles, n_bits):
        roles_seen.add(n_roles)
        return _model_encoder(feats, n_roles, n_bits)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(encoder))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", lambda f, m: (1.0, 1.0))

    RoleExtractor(n_bit_range=(1, 2))._select_model(features)

    # roles range is 2..min(3, 8) inclusive
    assert roles_seen == {2, 3}


def test_select_model_bits_grid_follows_bit_range(monkeypatch):
    features = pd.DataFrame(np.zeros((2, 4)))
    bits_seen = set()

    def encoder(feats, n_roles, n_bits):
        bits_seen.add(n_bits)
        return _model_encoder(feats, n_roles, n_bits)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(encoder))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", lambda f, m: (1.0, 1.0))

    RoleExtractor(n_bit_range=(2, 4))._select_model(features)

    assert bits_seen == {2, 3, 4}
