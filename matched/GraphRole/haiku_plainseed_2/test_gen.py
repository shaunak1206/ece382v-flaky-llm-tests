import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_mod
from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features():
    """6 nodes x 4 features, indexed by node name."""
    return pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4),
        index=[f"n{i}" for i in range(6)],
        columns=[f"f{j}" for j in range(4)],
    )


def _stub_factors(n_nodes, n_roles, n_features):
    """Deterministic factor pair with the expected shapes."""
    G = np.full((n_nodes, n_roles), 1.0)
    F = np.full((n_roles, n_features), 2.0)
    return G, F


# ---------------------------------------------------------------------------
# Initialization and simple properties
# ---------------------------------------------------------------------------

def test_init_defaults():
    ext = RoleExtractor()
    assert ext.n_roles is None
    assert (ext.min_roles, ext.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (ext.min_bits, ext.max_bits) == RoleExtractor.N_BIT_RANGE
    assert ext.node_role_factor is None
    assert ext.role_feature_factor is None


def test_init_stores_n_roles_and_custom_ranges():
    ext = RoleExtractor(n_roles=3, n_role_range=(2, 5), n_bit_range=(4, 6))
    assert ext.n_roles == 3
    assert (ext.min_roles, ext.max_roles) == (2, 5)
    assert (ext.min_bits, ext.max_bits) == (4, 6)


def test_roles_is_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_is_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_roles_returns_argmax_mapping():
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


def test_explain_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        RoleExtractor().explain()


# ---------------------------------------------------------------------------
# extract_role_factors with a fixed number of roles
# ---------------------------------------------------------------------------

def test_extract_fixed_roles_passes_log2_bits(monkeypatch, features):
    calls = []

    def fake_encoded(feats, n_roles, n_bits):
        calls.append((n_roles, n_bits))
        return _stub_factors(feats.shape[0], n_roles, feats.shape[1])

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))

    # 2 roles * min(6, 4) = 8 -> log2(8) = 3 bits
    RoleExtractor(n_roles=2).extract_role_factors(features)
    assert calls == [(2, 3)]


def test_extract_bits_are_truncated_to_int(monkeypatch):
    calls = []

    def fake_encoded(feats, n_roles, n_bits):
        calls.append(n_bits)
        return _stub_factors(feats.shape[0], n_roles, feats.shape[1])

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))

    features = pd.DataFrame(np.zeros((10, 6)))
    # 4 roles * min(10, 6) = 24 -> log2(24) ~ 4.58 -> int 4
    RoleExtractor(n_roles=4).extract_role_factors(features)
    assert calls == [4]


def test_extract_fixed_roles_bypasses_model_selection(monkeypatch, features):
    def fake_encoded(feats, n_roles, n_bits):
        return _stub_factors(feats.shape[0], n_roles, feats.shape[1])

    def fail_select(self, feats):
        raise AssertionError("_select_model should not be called")

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))
    monkeypatch.setattr(RoleExtractor, "_select_model", fail_select)

    RoleExtractor(n_roles=2).extract_role_factors(features)


def test_extract_zero_roles_triggers_model_selection(monkeypatch, features):
    seen = []

    def fake_select(self, feats):
        seen.append(feats)
        return _stub_factors(feats.shape[0], 2, feats.shape[1])

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select)

    # n_roles=0 is falsy, so model selection should be used
    RoleExtractor(n_roles=0).extract_role_factors(features)
    assert len(seen) == 1
    assert seen[0] is features


def test_extract_uses_selected_model_factors(monkeypatch, features):
    G = np.arange(12, dtype=float).reshape(6, 2)
    F = np.arange(8, dtype=float).reshape(2, 4)

    monkeypatch.setattr(RoleExtractor, "_select_model", lambda self, feats: (G, F))

    ext = RoleExtractor()
    ext.extract_role_factors(features)
    np.testing.assert_array_equal(ext.node_role_factor.values, G)
    np.testing.assert_array_equal(ext.role_feature_factor.values, F)


def test_extract_output_frames_have_expected_labels_and_indices(monkeypatch, features):
    G = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    F = np.ones((2, 4))

    monkeypatch.setattr(RoleExtractor, "_select_model", lambda self, feats: (G, F))

    ext = RoleExtractor()
    ext.extract_role_factors(features)

    assert list(ext.node_role_factor.index) == list(features.index)
    assert list(ext.node_role_factor.columns) == ["role_0", "role_1"]
    assert list(ext.role_feature_factor.index) == ["role_0", "role_1"]
    assert list(ext.role_feature_factor.columns) == list(features.columns)


def test_roles_and_percentage_available_after_extraction(monkeypatch):
    features = pd.DataFrame(np.ones((2, 3)), index=["a", "b"], columns=["x", "y", "z"])
    G = np.array([[0.1, 0.9], [0.8, 0.2]])
    F = np.ones((2, 3))

    monkeypatch.setattr(RoleExtractor, "_select_model", lambda self, feats: (G, F))

    ext = RoleExtractor()
    ext.extract_role_factors(features)

    assert ext.roles == {"a": "role_1", "b": "role_0"}
    np.testing.assert_allclose(ext.role_percentage.sum(axis=1).values, [1.0, 1.0])


# ---------------------------------------------------------------------------
# _select_model grid search
# ---------------------------------------------------------------------------

def test_select_model_skips_value_error_and_returns_remaining(monkeypatch):
    def fake_encoded(feats, n_roles, n_bits):
        if (n_roles, n_bits) != (3, 2):
            raise ValueError("too many bits")
        return ("model_3_2", "factor")

    def fake_costs(feats, model):
        return 1.0, 1.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_costs)

    features = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    with np.errstate(all="ignore"):
        model = RoleExtractor()._select_model(features)
    assert model == ("model_3_2", "factor")


def test_select_model_respects_custom_ranges(monkeypatch):
    calls = []

    def fake_encoded(feats, n_roles, n_bits):
        calls.append((n_roles, n_bits))
        return ("m", "f")

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", lambda feats, model: (1.0, 1.0))

    features = pd.DataFrame(np.zeros((10, 10)))
    ext = RoleExtractor(n_role_range=(3, 4), n_bit_range=(2, 3))
    with np.errstate(all="ignore"):
        ext._select_model(features)

    assert set(calls) == {(3, 2), (3, 3), (4, 2), (4, 3)}


def test_select_model_caps_roles_at_smallest_feature_dimension(monkeypatch):
    calls = []

    def fake_encoded(feats, n_roles, n_bits):
        calls.append((n_roles, n_bits))
        return ("m", "f")

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_encoded))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", lambda feats, model: (1.0, 1.0))

    # min(shape) == 3, so roles are capped at 3 even though max_roles is 8
    features = pd.DataFrame(np.zeros((3, 10)))
    with np.errstate(all="ignore"):
        RoleExtractor()._select_model(features)

    assert {r for r, _ in calls} == {2, 3}


# ---------------------------------------------------------------------------
# _get_encoded_role_factors
# ---------------------------------------------------------------------------

def test_get_encoded_role_factors_wires_nmf_and_encode(monkeypatch):
    nmf_calls = []
    encode_calls = []

    def fake_nmf(V, n_roles):
        nmf_calls.append((V, n_roles))
        return np.full((V.shape[0], n_roles), 1.0), np.full((n_roles, V.shape[1]), 2.0)

    def fake_encode(M, n_bins):
        encode_calls.append(n_bins)
        return M + n_bins

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr(extract_mod, "encode", fake_encode)

    features = pd.DataFrame(np.arange(6, dtype=float).reshape(2, 3))
    G_enc, F_enc = RoleExtractor._get_encoded_role_factors(features, 2, 3)

    assert len(nmf_calls) == 1
    V, n_roles = nmf_calls[0]
    np.testing.assert_array_equal(V, features.values)
    assert n_roles == 2

    assert encode_calls == [8, 8]  # 2 ** 3 bins for both factors
    np.testing.assert_array_equal(G_enc, np.full((2, 2), 9.0))  # 1 + 8
    np.testing.assert_array_equal(F_enc, np.full((2, 3), 10.0))  # 2 + 8


# ---------------------------------------------------------------------------
# _rescale_costs
# ---------------------------------------------------------------------------

def test_rescale_costs_normalizes_rows_and_skips_nan():
    costs = np.array([[3.0, np.nan], [6.0, 8.0]])
    out = RoleExtractor._rescale_costs(costs)
    expected = np.array([[1.0, np.nan], [0.6, 0.8]])
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_rescale_costs_preserves_shape():
    costs = np.ones((3, 4))
    out = RoleExtractor._rescale_costs(costs)
    assert out.shape == (3, 4)
    # each row has L2 norm sqrt(4) = 2
    np.testing.assert_allclose(out, np.full((3, 4), 0.5))
