import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_module
from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features():
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        rng.rand(30, 5),
        index=[f'n{i}' for i in range(30)],
        columns=[f'f{j}' for j in range(5)],
    )


def test_init_defaults():
    ex = RoleExtractor()
    assert ex.n_roles is None
    assert (ex.min_roles, ex.max_roles) == RoleExtractor.N_ROLE_RANGE == (2, 8)
    assert (ex.min_bits, ex.max_bits) == RoleExtractor.N_BIT_RANGE == (1, 8)
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


def test_roles_returns_argmax_column_per_node():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]],
        index=['a', 'b', 'c'],
        columns=['role_0', 'role_1'],
    )
    assert ex.roles == {'a': 'role_0', 'b': 'role_1', 'c': 'role_0'}


def test_role_percentage_rows_sum_to_one():
    ex = RoleExtractor()
    ex.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=['a', 'b'],
        columns=['role_0', 'role_1'],
    )
    pct = ex.role_percentage
    assert np.allclose(pct.sum(axis=1), 1.0)
    assert pct.loc['a', 'role_0'] == pytest.approx(0.25)
    assert pct.loc['a', 'role_1'] == pytest.approx(0.75)
    assert pct.loc['b', 'role_0'] == pytest.approx(0.5)


def test_explain_not_implemented():
    with pytest.raises(NotImplementedError):
        RoleExtractor().explain()


def test_rescale_costs_normalizes_rows_and_keeps_nan():
    costs = np.array([[3.0, 4.0], [np.nan, 2.0]])
    result = RoleExtractor._rescale_costs(costs)
    assert result.shape == costs.shape
    assert result[0] == pytest.approx([0.6, 0.8])
    assert np.isnan(result[1, 0])
    assert result[1, 1] == pytest.approx(1.0)


def test_get_encoded_role_factors_uses_nmf_and_encode(monkeypatch, features):
    calls = {}
    G = np.ones((30, 3))
    F = np.ones((3, 5))

    def fake_nmf(V, n_roles):
        calls['nmf'] = (V.shape, n_roles)
        return G, F

    encode_calls = []

    def fake_encode(matrix, n_bins):
        encode_calls.append((matrix, n_bins))
        return matrix * 2

    monkeypatch.setattr(extract_module, 'get_nmf_decomposition', fake_nmf)
    monkeypatch.setattr(extract_module, 'encode', fake_encode)

    G_enc, F_enc = RoleExtractor._get_encoded_role_factors(features, 3, 4)

    assert calls['nmf'] == ((30, 5), 3)
    assert [n for _, n in encode_calls] == [16, 16]
    assert encode_calls[0][0] is G and encode_calls[1][0] is F
    assert np.array_equal(G_enc, G * 2)
    assert np.array_equal(F_enc, F * 2)


def test_extract_role_factors_fixed_roles_computes_bits(monkeypatch):
    feats = pd.DataFrame(np.random.RandomState(1).rand(16, 8))
    seen = {}

    def fake(f, n_roles, n_bits):
        seen['args'] = (n_roles, n_bits)
        return np.ones((16, n_roles)), np.ones((n_roles, 8))

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake))
    ex = RoleExtractor(n_roles=2)
    ex.extract_role_factors(feats)
    # int(log2(2 * min(16, 8))) == 4
    assert seen['args'] == (2, 4)


def test_extract_role_factors_labels_and_indices(monkeypatch, features):
    def fake_select(self, f):
        return np.ones((30, 3)), np.ones((3, 5))

    monkeypatch.setattr(RoleExtractor, '_select_model', fake_select)
    ex = RoleExtractor()
    ex.extract_role_factors(features)

    labels = ['role_0', 'role_1', 'role_2']
    assert list(ex.node_role_factor.columns) == labels
    assert list(ex.node_role_factor.index) == list(features.index)
    assert list(ex.role_feature_factor.index) == labels
    assert list(ex.role_feature_factor.columns) == list(features.columns)


def test_extract_role_factors_without_n_roles_calls_select_model(monkeypatch, features):
    called = []

    def fake_select(self, f):
        called.append(f)
        return np.ones((30, 2)), np.ones((2, 5))

    monkeypatch.setattr(RoleExtractor, '_select_model', fake_select)
    RoleExtractor().extract_role_factors(features)
    assert len(called) == 1
    assert called[0] is features


def test_extract_role_factors_real_pipeline(features):
    ex = RoleExtractor(n_roles=2)
    ex.extract_role_factors(features)

    assert ex.node_role_factor.shape == (30, 2)
    assert ex.role_feature_factor.shape == (2, 5)
    assert not ex.node_role_factor.isnull().values.any()
    roles = ex.roles
    assert set(roles.keys()) == set(features.index)
    assert set(roles.values()) <= {'role_0', 'role_1'}


def _patch_grid(monkeypatch, enc_fn, err_fn, fail_bits=None, record=None):
    def fake_factors(f, n_roles, n_bits):
        if record is not None:
            record.append((n_roles, n_bits))
        return (n_roles, n_bits)

    def fake_costs(f, model):
        roles, bits = model
        if fail_bits is not None and bits == fail_bits:
            raise ValueError('too many bits')
        return enc_fn(roles, bits), err_fn(roles, bits)

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', staticmethod(fake_factors))
    monkeypatch.setattr(extract_module, 'get_description_length_costs', fake_costs)


def test_select_model_picks_minimum_cost(monkeypatch):
    feats = pd.DataFrame(np.ones((5, 4)))
    errors = {1: 10.0, 2: 1.0, 3: 10.0}

    def err(roles, bits):
        return errors[bits] if roles == 3 else 10.0

    _patch_grid(monkeypatch, lambda r, b: 1.0, err)
    ex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 3))
    with np.errstate(all='ignore'):
        model = ex._select_model(feats)
    assert model == (3, 2)


def test_select_model_skips_value_errors(monkeypatch):
    feats = pd.DataFrame(np.ones((5, 4)))
    # Bits=3 would otherwise be cheapest, but it raises ValueError
    errors = {1: 10.0, 2: 5.0, 3: 0.1}

    _patch_grid(
        monkeypatch,
        lambda r, b: 1.0,
        lambda r, b: errors[b] if r == 2 else 10.0,
        fail_bits=3,
    )
    ex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 3))
    with np.errstate(all='ignore'):
        roles, bits = ex._select_model(feats)
    assert bits != 3
    assert (roles, bits) == (2, 2)


def test_select_model_grid_limited_by_feature_shape(monkeypatch):
    feats = pd.DataFrame(np.ones((3, 10)))
    record = []
    _patch_grid(monkeypatch, lambda r, b: float(r + b), lambda r, b: float(r * b), record=record)
    ex = RoleExtractor(n_bit_range=(1, 2))
    with np.errstate(all='ignore'):
        ex._select_model(feats)
    roles_tried = {r for r, _ in record}
    bits_tried = {b for _, b in record}
    # max roles default 8 is clamped to min(features.shape) == 3
    assert roles_tried == {2, 3}
    assert bits_tried == {1, 2}
    assert len(record) == 4


def test_select_model_real_pipeline_returns_factor_tuple(features):
    ex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 3))
    G, F = ex._select_model(features)
    assert G.shape[0] == 30
    assert F.shape[1] == 5
    assert G.shape[1] == F.shape[0]
    assert 2 <= G.shape[1] <= 3
