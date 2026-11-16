import numpy as np
import pandas as pd
import pytest

from graphrole.roles import extract as extract_module
from graphrole.roles.extract import RoleExtractor


def make_features(n_nodes=20, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    data = rng.rand(n_nodes, n_features) * 10
    return pd.DataFrame(
        data,
        index=[f'n{i}' for i in range(n_nodes)],
        columns=[f'f{j}' for j in range(n_features)],
    )


def test_init_defaults():
    ex = RoleExtractor()
    assert ex.n_roles is None
    assert (ex.min_roles, ex.max_roles) == (2, 8)
    assert (ex.min_bits, ex.max_bits) == (1, 8)
    assert ex.node_role_factor is None
    assert ex.role_feature_factor is None


def test_init_custom_values():
    ex = RoleExtractor(n_roles=3, n_role_range=(3, 5), n_bit_range=(2, 4))
    assert ex.n_roles == 3
    assert (ex.min_roles, ex.max_roles) == (3, 5)
    assert (ex.min_bits, ex.max_bits) == (2, 4)


def test_roles_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_roles_returns_argmax_column():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[1.0, 5.0], [7.0, 2.0]],
        index=['a', 'b'],
        columns=['role_0', 'role_1'],
    )
    assert ex.roles == {'a': 'role_1', 'b': 'role_0'}


def test_role_percentage_values():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=['a', 'b'],
        columns=['role_0', 'role_1'],
    )
    pct = ex.role_percentage
    assert pct.loc['a', 'role_0'] == pytest.approx(0.25)
    assert pct.loc['a', 'role_1'] == pytest.approx(0.75)
    assert pct.loc['b', 'role_0'] == pytest.approx(0.5)
    assert np.allclose(pct.sum(axis=1).values, 1.0)


def test_explain_not_implemented():
    with pytest.raises(NotImplementedError):
        RoleExtractor().explain()


def test_rescale_costs_rows_have_unit_norm():
    costs = np.array([[3.0, 4.0], [6.0, 8.0], [1.0, np.nan]])
    out = RoleExtractor._rescale_costs(costs)
    assert out.shape == costs.shape
    norms = np.sqrt(np.nansum(np.square(out), axis=1))
    assert np.allclose(norms, 1.0)
    assert out[0, 0] == pytest.approx(0.6)
    assert out[0, 1] == pytest.approx(0.8)


def test_rescale_costs_preserves_nan_positions():
    costs = np.array([[1.0, np.nan, 2.0], [np.nan, 3.0, 4.0]])
    out = RoleExtractor._rescale_costs(costs)
    assert np.array_equal(np.isnan(out), np.isnan(costs))


def test_get_encoded_role_factors_uses_nmf_and_encode(monkeypatch):
    features = make_features(6, 4)
    G = np.ones((6, 2))
    F = np.ones((2, 4)) * 2
    calls = {'nmf': [], 'encode': []}

    def fake_nmf(V, n_roles):
        calls['nmf'].append((V.shape, n_roles))
        return G, F

    def fake_encode(M, n_bins):
        calls['encode'].append((M.shape, n_bins))
        return M * 10

    monkeypatch.setattr(extract_module, 'get_nmf_decomposition', fake_nmf)
    monkeypatch.setattr(extract_module, 'encode', fake_encode)

    G_enc, F_enc = RoleExtractor._get_encoded_role_factors(features, 2, 3)

    assert calls['nmf'] == [((6, 4), 2)]
    assert calls['encode'] == [((6, 2), 8), ((2, 4), 8)]
    assert np.array_equal(G_enc, G * 10)
    assert np.array_equal(F_enc, F * 10)


def test_extract_role_factors_with_fixed_n_roles_calls_with_bits(monkeypatch):
    features = make_features(10, 4)
    seen = {}

    def fake(feats, n_roles, n_bits):
        seen['args'] = (n_roles, n_bits)
        return np.ones((10, n_roles)), np.ones((n_roles, 4))

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake))
    ex = RoleExtractor(n_roles=2)
    ex.extract_role_factors(features)

    # int(log2(2 * min(10, 4))) == 3
    assert seen['args'] == (2, 3)


def test_extract_role_factors_labels_and_indices(monkeypatch):
    features = make_features(10, 4)

    def fake(feats, n_roles, n_bits):
        return np.arange(30, dtype=float).reshape(10, 3), np.ones((3, 4))

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake))
    ex = RoleExtractor(n_roles=3)
    ex.extract_role_factors(features)

    assert list(ex.node_role_factor.columns) == ['role_0', 'role_1', 'role_2']
    assert list(ex.node_role_factor.index) == list(features.index)
    assert list(ex.role_feature_factor.index) == ['role_0', 'role_1', 'role_2']
    assert list(ex.role_feature_factor.columns) == list(features.columns)
    assert set(ex.roles.values()) == {'role_2'}


def test_extract_role_factors_without_n_roles_uses_select_model(monkeypatch):
    features = make_features(8, 3)
    called = []

    def fake_select(self, feats):
        called.append(True)
        return np.ones((8, 2)), np.ones((2, 3))

    monkeypatch.setattr(RoleExtractor, '_select_model', fake_select)
    ex = RoleExtractor()
    ex.extract_role_factors(features)

    assert called == [True]
    assert ex.node_role_factor.shape == (8, 2)
    assert ex.role_feature_factor.shape == (2, 3)


def test_select_model_picks_minimum_cost(monkeypatch):
    features = make_features(10, 6)
    enc = {(2, 1): 10.0, (2, 2): 10.0, (3, 1): 10.0, (3, 2): 1.0}

    def fake_factors(feats, roles, bits):
        return ('model', roles, bits)

    def fake_costs(feats, model):
        _, roles, bits = model
        return enc[(roles, bits)], enc[(roles, bits)]

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, 'get_description_length_costs', fake_costs)

    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    assert ex._select_model(features) == ('model', 3, 2)


def test_select_model_skips_value_errors(monkeypatch):
    features = make_features(10, 6)
    enc = {(2, 1): 10.0, (2, 2): 10.0, (3, 1): 5.0, (3, 2): 1.0}

    def fake_factors(feats, roles, bits):
        return ('model', roles, bits)

    def fake_costs(feats, model):
        _, roles, bits = model
        if (roles, bits) == (3, 2):
            raise ValueError('too many bits')
        return enc[(roles, bits)], enc[(roles, bits)]

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, 'get_description_length_costs', fake_costs)

    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    assert ex._select_model(features) == ('model', 3, 1)


def test_select_model_caps_roles_at_smallest_dimension(monkeypatch):
    features = make_features(10, 3)
    tried = []

    def fake_factors(feats, roles, bits):
        tried.append((roles, bits))
        return ('model', roles, bits)

    def fake_costs(feats, model):
        return 1.0 + model[1], 1.0 + model[2]

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, 'get_description_length_costs', fake_costs)

    ex = RoleExtractor(n_role_range=(2, 8), n_bit_range=(1, 1))
    ex._select_model(features)

    assert sorted(r for r, _ in tried) == [2, 3]
    assert all(b == 1 for _, b in tried)


def test_extract_role_factors_end_to_end_fixed_roles():
    features = make_features(20, 5)
    ex = RoleExtractor(n_roles=2)
    ex.extract_role_factors(features)

    assert ex.node_role_factor.shape == (20, 2)
    assert ex.role_feature_factor.shape == (2, 5)
    assert list(ex.node_role_factor.index) == list(features.index)
    assert list(ex.role_feature_factor.columns) == list(features.columns)
    roles = ex.roles
    assert set(roles.keys()) == set(features.index)
    assert set(roles.values()) <= {'role_0', 'role_1'}


def test_extract_role_factors_end_to_end_model_selection():
    features = make_features(20, 5, seed=1)
    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 3))
    ex.extract_role_factors(features)

    n_roles = ex.node_role_factor.shape[1]
    assert 2 <= n_roles <= 3
    assert ex.node_role_factor.shape[0] == 20
    assert ex.role_feature_factor.shape == (n_roles, 5)
    assert set(ex.roles.keys()) == set(features.index)
