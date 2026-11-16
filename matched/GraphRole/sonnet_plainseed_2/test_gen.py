from unittest import mock

import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features():
    rng = np.random.RandomState(0)
    data = rng.rand(20, 6) + 0.1
    return pd.DataFrame(
        data,
        index=[f'n{i}' for i in range(20)],
        columns=[f'f{j}' for j in range(6)],
    )


def test_init_defaults():
    ex = RoleExtractor()
    assert ex.n_roles is None
    assert (ex.min_roles, ex.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (ex.min_bits, ex.max_bits) == RoleExtractor.N_BIT_RANGE
    assert ex.node_role_factor is None
    assert ex.role_feature_factor is None


def test_init_custom_ranges():
    ex = RoleExtractor(n_roles=3, n_role_range=(3, 5), n_bit_range=(2, 4))
    assert ex.n_roles == 3
    assert (ex.min_roles, ex.max_roles) == (3, 5)
    assert (ex.min_bits, ex.max_bits) == (2, 4)


def test_roles_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_explain_not_implemented():
    with pytest.raises(NotImplementedError):
        RoleExtractor().explain()


def test_roles_returns_argmax_role_per_node():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
        index=['a', 'b', 'c'],
        columns=['role_0', 'role_1'],
    )
    assert ex.roles == {'a': 'role_1', 'b': 'role_0', 'c': 'role_1'}


def test_role_percentage_rows_sum_to_one():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 5.0]],
        index=['a', 'b', 'c'],
        columns=['role_0', 'role_1'],
    )
    pct = ex.role_percentage
    assert list(pct.index) == ['a', 'b', 'c']
    assert list(pct.columns) == ['role_0', 'role_1']
    np.testing.assert_allclose(pct.sum(axis=1).values, 1.0)
    assert pct.loc['a', 'role_1'] == pytest.approx(0.75)
    assert pct.loc['b', 'role_0'] == pytest.approx(0.5)


def test_rescale_costs_rows_have_unit_norm():
    costs = np.array([[3.0, 4.0], [1.0, 1.0]])
    scaled = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(scaled[0], [0.6, 0.8])
    np.testing.assert_allclose(np.linalg.norm(scaled, axis=1), 1.0)


def test_rescale_costs_ignores_nan_in_norm():
    costs = np.array([[3.0, np.nan, 4.0]])
    scaled = RoleExtractor._rescale_costs(costs)
    assert scaled.shape == costs.shape
    assert scaled[0, 0] == pytest.approx(0.6)
    assert np.isnan(scaled[0, 1])
    assert scaled[0, 2] == pytest.approx(0.8)


def test_get_encoded_role_factors_shapes(features):
    G, F = RoleExtractor._get_encoded_role_factors(features, 2, 3)
    assert G.shape == (features.shape[0], 2)
    assert F.shape == (2, features.shape[1])
    assert np.all(np.asarray(G) >= 0)
    assert np.all(np.asarray(F) >= 0)


def test_extract_role_factors_with_fixed_n_roles(features):
    ex = RoleExtractor(n_roles=3)
    ex.extract_role_factors(features)

    labels = ['role_0', 'role_1', 'role_2']
    assert list(ex.node_role_factor.columns) == labels
    assert list(ex.node_role_factor.index) == list(features.index)
    assert list(ex.role_feature_factor.index) == labels
    assert list(ex.role_feature_factor.columns) == list(features.columns)

    roles = ex.roles
    assert set(roles.keys()) == set(features.index)
    assert set(roles.values()) <= set(labels)

    pct = ex.role_percentage
    assert pct.shape == ex.node_role_factor.shape


def test_extract_role_factors_computes_n_bits_from_shape(features):
    # features shape (20, 6): n_bits = int(log2(2 * 6)) = 3
    G = np.ones((20, 2))
    F = np.ones((2, 6))
    with mock.patch.object(
        RoleExtractor, '_get_encoded_role_factors', return_value=(G, F)
    ) as m:
        ex = RoleExtractor(n_roles=2)
        ex.extract_role_factors(features)
    assert m.call_count == 1
    args = m.call_args[0]
    assert args[1] == 2
    assert args[2] == 3


def test_extract_role_factors_uses_model_selection_without_n_roles(features):
    G = np.ones((20, 4))
    F = np.ones((4, 6))
    ex = RoleExtractor()
    with mock.patch.object(RoleExtractor, '_select_model', return_value=(G, F)) as m:
        ex.extract_role_factors(features)
    m.assert_called_once()
    assert list(ex.node_role_factor.columns) == [f'role_{i}' for i in range(4)]
    assert ex.node_role_factor.shape == (20, 4)
    assert ex.role_feature_factor.shape == (4, 6)


def test_extract_role_factors_with_model_selection_end_to_end(features):
    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 3))
    ex.extract_role_factors(features)
    n_roles = ex.node_role_factor.shape[1]
    assert 2 <= n_roles <= 3
    assert ex.role_feature_factor.shape == (n_roles, features.shape[1])
    assert set(ex.roles.keys()) == set(features.index)


def test_select_model_searches_full_grid(features):
    calls = []

    def fake_encode(feats, roles, bits):
        calls.append((roles, bits))
        return ('model', roles, bits)

    def fake_cost(feats, model):
        _, roles, bits = model
        return float(roles + bits), float(10 - roles + bits)

    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    with mock.patch.object(RoleExtractor, '_get_encoded_role_factors', side_effect=fake_encode), \
            mock.patch('graphrole.roles.extract.get_description_length_costs', side_effect=fake_cost):
        model = ex._select_model(features)

    assert sorted(calls) == [(2, 1), (2, 2), (3, 1), (3, 2)]
    assert model[0] == 'model'
    assert (model[1], model[2]) in calls


def test_select_model_caps_roles_by_feature_shape():
    small = pd.DataFrame(np.random.RandomState(1).rand(10, 2) + 0.1)
    calls = []

    def fake_encode(feats, roles, bits):
        calls.append((roles, bits))
        return ('model', roles, bits)

    ex = RoleExtractor(n_role_range=(2, 8), n_bit_range=(1, 1))
    with mock.patch.object(RoleExtractor, '_get_encoded_role_factors', side_effect=fake_encode), \
            mock.patch('graphrole.roles.extract.get_description_length_costs',
                       return_value=(1.0, 1.0)):
        ex._select_model(small)

    assert {r for r, _ in calls} == {2}


def test_select_model_skips_combinations_raising_value_error(features):
    def fake_encode(feats, roles, bits):
        return ('model', roles, bits)

    def fake_cost(feats, model):
        _, roles, bits = model
        if bits == 2:
            raise ValueError('too many bits')
        return 1.0, 1.0

    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    with mock.patch.object(RoleExtractor, '_get_encoded_role_factors', side_effect=fake_encode), \
            mock.patch('graphrole.roles.extract.get_description_length_costs', side_effect=fake_cost):
        model = ex._select_model(features)

    assert model[2] == 1
    assert model[1] in (2, 3)
