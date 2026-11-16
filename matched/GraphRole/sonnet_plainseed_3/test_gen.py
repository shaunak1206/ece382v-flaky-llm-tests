import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_module
from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features():
    rng = np.random.RandomState(0)
    data = rng.rand(20, 6) * 10
    index = [f'n{i}' for i in range(20)]
    columns = [f'f{j}' for j in range(6)]
    return pd.DataFrame(data, index=index, columns=columns)


def test_init_defaults():
    ex = RoleExtractor()
    assert ex.n_roles is None
    assert (ex.min_roles, ex.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (ex.min_bits, ex.max_bits) == RoleExtractor.N_BIT_RANGE
    assert ex.node_role_factor is None
    assert ex.role_feature_factor is None


def test_init_custom_ranges():
    ex = RoleExtractor(n_roles=4, n_role_range=(3, 5), n_bit_range=(2, 6))
    assert ex.n_roles == 4
    assert (ex.min_roles, ex.max_roles) == (3, 5)
    assert (ex.min_bits, ex.max_bits) == (2, 6)


def test_roles_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_explain_not_implemented():
    with pytest.raises(NotImplementedError):
        RoleExtractor().explain()


def test_rescale_costs_known_values():
    costs = np.array([[3.0, 4.0], [0.0, 2.0]])
    result = RoleExtractor._rescale_costs(costs)
    expected = np.array([[0.6, 0.8], [0.0, 1.0]])
    assert np.allclose(result, expected)


def test_rescale_costs_skips_nan():
    costs = np.array([[np.nan, 3.0, 4.0], [np.nan, 1.0, 0.0]])
    result = RoleExtractor._rescale_costs(costs)
    assert np.isnan(result[:, 0]).all()
    assert np.allclose(result[:, 1:], np.array([[0.6, 0.8], [1.0, 0.0]]))


def test_get_encoded_role_factors_shapes(features):
    G, F = RoleExtractor._get_encoded_role_factors(features, 3, 4)
    assert G.shape == (20, 3)
    assert F.shape == (3, 6)
    assert np.all(np.asarray(G) >= 0)
    assert np.all(np.asarray(F) >= 0)


def test_extract_with_fixed_n_roles_structure(features):
    ex = RoleExtractor(n_roles=3)
    ex.extract_role_factors(features)
    assert ex.node_role_factor.shape == (20, 3)
    assert ex.role_feature_factor.shape == (3, 6)
    labels = ['role_0', 'role_1', 'role_2']
    assert list(ex.node_role_factor.columns) == labels
    assert list(ex.role_feature_factor.index) == labels
    assert list(ex.node_role_factor.index) == list(features.index)
    assert list(ex.role_feature_factor.columns) == list(features.columns)


def test_roles_and_percentage_after_fixed_extraction(features):
    ex = RoleExtractor(n_roles=3)
    ex.extract_role_factors(features)
    roles = ex.roles
    assert set(roles.keys()) == set(features.index)
    assert set(roles.values()) <= {'role_0', 'role_1', 'role_2'}
    pct = ex.role_percentage
    assert pct.shape == (20, 3)
    sums = pct.sum(axis=1).values
    valid = ~np.isnan(sums)
    assert np.allclose(sums[valid], 1.0)


def test_n_bits_computation_for_fixed_roles(monkeypatch):
    calls = []

    def fake(features, n_roles, n_bits):
        calls.append((n_roles, n_bits))
        return np.ones((features.shape[0], n_roles)), np.ones((n_roles, features.shape[1]))

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake))
    df = pd.DataFrame(np.ones((10, 8)))
    RoleExtractor(n_roles=4).extract_role_factors(df)
    # int(log2(4 * min(10, 8))) == 5
    assert calls == [(4, 5)]


def test_select_model_called_only_without_n_roles(monkeypatch):
    called = []

    def fake_select(self, features):
        called.append(True)
        return (
            np.ones((features.shape[0], 2)),
            np.ones((2, features.shape[1])),
        )

    monkeypatch.setattr(RoleExtractor, '_select_model', fake_select)
    df = pd.DataFrame(np.ones((5, 4)))
    ex = RoleExtractor()
    ex.extract_role_factors(df)
    assert called == [True]
    assert ex.node_role_factor.shape == (5, 2)

    called.clear()
    monkeypatch.setattr(
        RoleExtractor,
        '_get_encoded_role_factors',
        staticmethod(lambda f, r, b: (np.ones((f.shape[0], r)), np.ones((r, f.shape[1])))),
    )
    RoleExtractor(n_roles=2).extract_role_factors(df)
    assert called == []


def test_roles_and_percentage_with_known_factors(monkeypatch):
    G = np.array([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0]])
    F = np.ones((2, 2))
    monkeypatch.setattr(
        RoleExtractor, '_select_model', lambda self, features: (G, F)
    )
    df = pd.DataFrame(np.ones((3, 2)), index=['a', 'b', 'c'], columns=['x', 'y'])
    ex = RoleExtractor()
    ex.extract_role_factors(df)
    assert ex.roles['a'] == 'role_0'
    assert ex.roles['b'] == 'role_1'
    pct = ex.role_percentage
    assert np.allclose(pct.loc['a'].values, [0.75, 0.25])
    assert np.allclose(pct.loc['b'].values, [0.0, 1.0])
    assert np.allclose(pct.loc['c'].values, [0.5, 0.5])


def test_select_model_picks_minimal_cost(monkeypatch):
    def fake_factors(features, n_roles, n_bits):
        return (
            np.full((features.shape[0], n_roles), float(n_bits)),
            np.full((n_roles, features.shape[1]), float(n_roles)),
        )

    def fake_costs(features, model):
        G, F = model
        roles, bits = G.shape[1], int(G[0, 0])
        if roles == 3 and bits == 2:
            return 1.0, 1.0
        return 10.0, 10.0

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, 'get_description_length_costs', fake_costs)

    df = pd.DataFrame(np.ones((10, 5)))
    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 3))
    G, F = ex._select_model(df)
    assert G.shape == (10, 3)
    assert G[0, 0] == 2.0
    assert F.shape == (3, 5)


def test_select_model_skips_value_errors(monkeypatch):
    def fake_factors(features, n_roles, n_bits):
        if n_bits > 1:
            raise ValueError('too many bits')
        return (
            np.ones((features.shape[0], n_roles)),
            np.ones((n_roles, features.shape[1])),
        )

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(
        extract_module, 'get_description_length_costs',
        lambda features, model: (float(model[0].shape[1]), 1.0),
    )
    df = pd.DataFrame(np.ones((6, 4)))
    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 4))
    G, F = ex._select_model(df)
    assert G.shape[1] in (2, 3)


def test_automatic_model_selection_within_range(features):
    ex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 4))
    ex.extract_role_factors(features)
    n_roles = ex.node_role_factor.shape[1]
    assert 2 <= n_roles <= 4
    assert ex.node_role_factor.shape[0] == 20
    assert ex.role_feature_factor.shape == (n_roles, 6)
    assert set(ex.roles.keys()) == set(features.index)


def test_max_roles_capped_by_feature_shape(monkeypatch):
    seen_roles = set()

    def fake_factors(features, n_roles, n_bits):
        seen_roles.add(n_roles)
        return (
            np.ones((features.shape[0], n_roles)),
            np.ones((n_roles, features.shape[1])),
        )

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(
        extract_module, 'get_description_length_costs',
        lambda features, model: (1.0, 1.0),
    )
    df = pd.DataFrame(np.ones((10, 3)))
    ex = RoleExtractor(n_role_range=(2, 8), n_bit_range=(1, 2))
    ex._select_model(df)
    assert max(seen_roles) == 3
    assert min(seen_roles) == 2
