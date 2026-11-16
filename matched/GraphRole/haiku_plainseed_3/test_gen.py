import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_module
from graphrole.roles.extract import RoleExtractor


def make_features(n_nodes=4, n_feats=6):
    index = [f"n{i}" for i in range(n_nodes)]
    columns = [f"f{j}" for j in range(n_feats)]
    data = np.arange(n_nodes * n_feats, dtype=float).reshape(n_nodes, n_feats)
    return pd.DataFrame(data, index=index, columns=columns)


def test_init_defaults():
    ext = RoleExtractor()
    assert ext.n_roles is None
    assert (ext.min_roles, ext.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (ext.min_bits, ext.max_bits) == RoleExtractor.N_BIT_RANGE
    assert ext.node_role_factor is None
    assert ext.role_feature_factor is None


def test_init_custom_ranges_and_n_roles():
    ext = RoleExtractor(n_roles=3, n_role_range=(2, 5), n_bit_range=(3, 4))
    assert ext.n_roles == 3
    assert (ext.min_roles, ext.max_roles) == (2, 5)
    assert (ext.min_bits, ext.max_bits) == (3, 4)


def test_roles_is_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_is_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_roles_maps_each_node_to_argmax_role():
    ext = RoleExtractor()
    ext.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [0.8, 0.2]],
        index=["a", "b"],
        columns=["role_0", "role_1"],
    )
    assert ext.roles == {"a": "role_1", "b": "role_0"}


def test_role_percentage_rows_sum_to_one():
    ext = RoleExtractor()
    ext.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=["a", "b"],
        columns=["role_0", "role_1"],
    )
    pct = ext.role_percentage
    np.testing.assert_allclose(pct.loc["a"].values, [0.25, 0.75])
    np.testing.assert_allclose(pct.loc["b"].values, [0.5, 0.5])
    np.testing.assert_allclose(pct.sum(axis=1).values, [1.0, 1.0])


def test_extract_fixed_n_roles_uses_log2_bit_count(monkeypatch):
    calls = []

    def fake_factors(features, n_roles, n_bits):
        calls.append((n_roles, n_bits))
        return np.ones((features.shape[0], n_roles)), np.ones((n_roles, features.shape[1]))

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    # n_bits = int(log2(2 * min(4, 6))) = int(log2(8)) = 3
    RoleExtractor(n_roles=2).extract_role_factors(make_features(4, 6))
    assert calls == [(2, 3)]


def test_extract_sets_node_role_factor_index_and_columns(monkeypatch):
    features = make_features(4, 6)

    def fake_factors(feats, n_roles, n_bits):
        return np.full((feats.shape[0], n_roles), 0.5), np.full((n_roles, feats.shape[1]), 0.1)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    ext = RoleExtractor(n_roles=3)
    ext.extract_role_factors(features)
    assert list(ext.node_role_factor.index) == list(features.index)
    assert list(ext.node_role_factor.columns) == ["role_0", "role_1", "role_2"]


def test_extract_sets_role_feature_factor_index_and_columns(monkeypatch):
    features = make_features(4, 6)

    def fake_factors(feats, n_roles, n_bits):
        return np.full((feats.shape[0], n_roles), 0.5), np.full((n_roles, feats.shape[1]), 0.1)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    ext = RoleExtractor(n_roles=2)
    ext.extract_role_factors(features)
    assert list(ext.role_feature_factor.index) == ["role_0", "role_1"]
    assert list(ext.role_feature_factor.columns) == list(features.columns)


def test_roles_after_extraction_maps_nodes(monkeypatch):
    features = make_features(3, 4)
    G = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    F = np.ones((2, 4))

    monkeypatch.setattr(
        RoleExtractor,
        "_get_encoded_role_factors",
        staticmethod(lambda feats, n_roles, n_bits: (G, F)),
    )
    ext = RoleExtractor(n_roles=2)
    ext.extract_role_factors(features)
    assert ext.roles == {"n0": "role_0", "n1": "role_1", "n2": "role_0"}


def test_extract_without_n_roles_delegates_to_select_model(monkeypatch):
    features = make_features(4, 6)
    G = np.ones((4, 3))
    F = np.ones((3, 6))
    seen = []

    def fake_select(self, feats):
        seen.append(feats)
        return G, F

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select)
    ext = RoleExtractor()
    ext.extract_role_factors(features)
    assert len(seen) == 1
    assert seen[0] is features
    assert list(ext.node_role_factor.columns) == ["role_0", "role_1", "role_2"]


def test_extract_zero_n_roles_falls_back_to_select_model(monkeypatch):
    # n_roles=0 is falsy, so the code takes the model selection branch
    called = []

    def fake_select(self, feats):
        called.append(True)
        return np.ones((feats.shape[0], 2)), np.ones((2, feats.shape[1]))

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select)
    RoleExtractor(n_roles=0).extract_role_factors(make_features())
    assert called == [True]


def test_get_encoded_role_factors_encodes_with_two_to_n_bits(monkeypatch):
    features = make_features(4, 6)
    G = np.full((4, 2), 0.5)
    F = np.full((2, 6), 0.25)
    nmf_calls = []
    encode_calls = []

    def fake_nmf(V, n_roles):
        nmf_calls.append((V, n_roles))
        return G, F

    def fake_encode(X, n_bins):
        encode_calls.append(n_bins)
        return X + 1

    monkeypatch.setattr(extract_module, "get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr(extract_module, "encode", fake_encode)

    G_enc, F_enc = RoleExtractor._get_encoded_role_factors(features, 2, 3)

    assert len(nmf_calls) == 1
    np.testing.assert_array_equal(nmf_calls[0][0], features.values)
    assert nmf_calls[0][1] == 2
    assert encode_calls == [8, 8]
    np.testing.assert_allclose(G_enc, G + 1)
    np.testing.assert_allclose(F_enc, F + 1)


def test_rescale_costs_divides_rows_by_l2_norm():
    costs = np.array([[3.0, 4.0], [6.0, 8.0]])
    result = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.6, 0.8]])


def test_rescale_costs_ignores_nan_entries():
    nan = np.nan
    costs = np.array([[3.0, 4.0, nan], [6.0, 8.0, nan]])
    result = RoleExtractor._rescale_costs(costs)
    expected = np.array([[0.6, 0.8, nan], [0.6, 0.8, nan]])
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_select_model_returns_minimum_cost_model(monkeypatch):
    # features shape (5, 4): roles grid 2..4, bits grid 1..8
    def fake_factors(features, n_roles, n_bits):
        return (n_roles, n_bits)

    def fake_costs(features, model):
        if model == (3, 5):
            return 1.0, 1.0
        return 10.0, 10.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, "get_description_length_costs", fake_costs)

    ext = RoleExtractor()
    model = ext._select_model(make_features(5, 4))
    assert model == (3, 5)


def test_select_model_skips_value_error_combinations(monkeypatch):
    def fake_factors(features, n_roles, n_bits):
        return (n_roles, n_bits)

    def fake_costs(features, model):
        _, bits = model
        if bits >= 7:
            raise ValueError("too many bits for number of samples")
        return 10.0, 10.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, "get_description_length_costs", fake_costs)

    # All valid costs are tied, so the first valid cell (roles=2, bits=1) wins
    model = RoleExtractor()._select_model(make_features(5, 4))
    assert model == (2, 1)
    assert model[1] <= 6


def test_select_model_default_grid_capped_by_feature_shape(monkeypatch):
    seen_roles = set()
    seen_bits = set()

    def fake_factors(features, n_roles, n_bits):
        seen_roles.add(n_roles)
        seen_bits.add(n_bits)
        return (n_roles, n_bits)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    monkeypatch.setattr(
        extract_module, "get_description_length_costs", lambda features, model: (1.0, 1.0)
    )

    # min(features.shape) = 3 caps roles at 3, since default max_roles is 8
    RoleExtractor()._select_model(make_features(3, 10))
    assert seen_roles == {2, 3}
    assert seen_bits == set(range(1, 9))


def test_select_model_custom_grid_ranges(monkeypatch):
    seen_roles = set()
    seen_bits = set()

    def fake_factors(features, n_roles, n_bits):
        seen_roles.add(n_roles)
        seen_bits.add(n_bits)
        return (n_roles, n_bits)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_factors))
    monkeypatch.setattr(
        extract_module, "get_description_length_costs", lambda features, model: (1.0, 1.0)
    )

    ext = RoleExtractor(n_role_range=(2, 5), n_bit_range=(2, 3))
    ext._select_model(make_features(10, 10))
    assert seen_roles == {2, 3, 4, 5}
    assert seen_bits == {2, 3}


def test_explain_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        RoleExtractor().explain()
