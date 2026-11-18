import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_mod
from graphrole.roles.extract import RoleExtractor


def test_init_uses_defaults_when_ranges_not_provided():
    rex = RoleExtractor()
    assert rex.n_roles is None
    assert rex.min_roles == RoleExtractor.N_ROLE_RANGE[0]
    assert rex.max_roles == RoleExtractor.N_ROLE_RANGE[1]
    assert rex.min_bits == RoleExtractor.N_BIT_RANGE[0]
    assert rex.max_bits == RoleExtractor.N_BIT_RANGE[1]
    assert rex.node_role_factor is None
    assert rex.role_feature_factor is None


def test_init_uses_custom_ranges():
    rex = RoleExtractor(n_roles=3, n_role_range=(4, 6), n_bit_range=(2, 5))
    assert rex.n_roles == 3
    assert rex.min_roles == 4
    assert rex.max_roles == 6
    assert rex.min_bits == 2
    assert rex.max_bits == 5


def test_roles_property_returns_none_before_factors_are_set():
    rex = RoleExtractor()
    assert rex.roles is None


def test_role_percentage_property_returns_none_before_factors_are_set():
    rex = RoleExtractor()
    assert rex.role_percentage is None


def test_roles_property_returns_argmax_role_per_node():
    rex = RoleExtractor()
    rex.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [2.0, 1.0], [0.0, 0.0]],
        index=["n1", "n2", "n3"],
        columns=["role_0", "role_1"],
    )
    assert rex.roles == {"n1": "role_1", "n2": "role_0", "n3": "role_0"}


def test_role_percentage_property_normalizes_each_row():
    rex = RoleExtractor()
    rex.node_role_factor = pd.DataFrame(
        [[1.0, 1.0], [2.0, 6.0]],
        index=["a", "b"],
        columns=["role_0", "role_1"],
    )

    result = rex.role_percentage
    assert isinstance(result, pd.DataFrame)
    assert result.loc["a", "role_0"] == pytest.approx(0.5)
    assert result.loc["a", "role_1"] == pytest.approx(0.5)
    assert result.loc["b", "role_0"] == pytest.approx(0.25)
    assert result.loc["b", "role_1"] == pytest.approx(0.75)


def test_extract_role_factors_with_fixed_n_roles_sets_dataframes_and_labels(monkeypatch):
    rex = RoleExtractor(n_roles=2)
    features = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        index=["n1", "n2"],
        columns=["f1", "f2", "f3"],
    )

    called = {}

    def fake_get_encoded_role_factors(features_arg, n_roles_arg, n_bits_arg):
        called["features"] = features_arg
        called["n_roles"] = n_roles_arg
        called["n_bits"] = n_bits_arg
        return np.array([[1, 0], [0, 1]]), np.array([[1, 2, 3], [4, 5, 6]])

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))

    rex.extract_role_factors(features)

    assert called["features"] is features
    assert called["n_roles"] == 2
    assert called["n_bits"] == int(np.log2(2 * min(features.shape)))
    assert list(rex.node_role_factor.index) == ["n1", "n2"]
    assert list(rex.node_role_factor.columns) == ["role_0", "role_1"]
    assert list(rex.role_feature_factor.index) == ["role_0", "role_1"]
    assert list(rex.role_feature_factor.columns) == ["f1", "f2", "f3"]


def test_extract_role_factors_without_n_roles_uses_model_selection(monkeypatch):
    rex = RoleExtractor()
    features = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=["a", "b", "c"],
        columns=["x", "y"],
    )

    sentinel = (np.array([[1, 0], [0, 1], [1, 1]]), np.array([[0.5, 0.5], [0.2, 0.8]]))
    called = {}

    def fake_select_model(features_arg):
        called["features"] = features_arg
        return sentinel

    monkeypatch.setattr(rex, "_select_model", fake_select_model)

    rex.extract_role_factors(features)

    assert called["features"] is features
    assert rex.node_role_factor.shape == (3, 2)
    assert rex.role_feature_factor.shape == (2, 2)
    assert list(rex.node_role_factor.columns) == ["role_0", "role_1"]


def test_get_encoded_role_factors_calls_nmf_and_encode(monkeypatch):
    features = pd.DataFrame([[1, 2], [3, 4]], columns=["f1", "f2"])

    calls = {"encode": []}

    def fake_get_nmf_decomposition(V, n_roles):
        assert n_roles == 3
        np.testing.assert_array_equal(V, features.values)
        return np.array([[10, 11], [12, 13]]), np.array([[20, 21], [22, 23]])

    def fake_encode(arr, n_bins):
        calls["encode"].append((arr.copy(), n_bins))
        return arr + n_bins

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", fake_get_nmf_decomposition)
    monkeypatch.setattr(extract_mod, "encode", fake_encode)

    g_enc, f_enc = RoleExtractor._get_encoded_role_factors(features, n_roles=3, n_bits=2)

    assert len(calls["encode"]) == 2
    assert calls["encode"][0][1] == 4
    assert calls["encode"][1][1] == 4
    np.testing.assert_array_equal(g_enc, np.array([[14, 15], [16, 17]]))
    np.testing.assert_array_equal(f_enc, np.array([[24, 25], [26, 27]]))


def test_rescale_costs_scales_each_row_by_l2_norm():
    costs = np.array([[3.0, 4.0], [0.0, 5.0]])
    result = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(result[0], np.array([0.6, 0.8]))
    np.testing.assert_allclose(result[1], np.array([0.0, 1.0]))


def test_rescale_costs_preserves_nan_entries_and_row_scaling():
    costs = np.array([[np.nan, 3.0], [4.0, np.nan]])
    result = RoleExtractor._rescale_costs(costs)
    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 1])
    assert result[0, 1] == pytest.approx(1.0)
    assert result[1, 0] == pytest.approx(1.0)


def test_select_model_picks_minimum_cost_model(monkeypatch):
    rex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 3))
    features = pd.DataFrame(np.ones((3, 3)))

    def fake_get_encoded_role_factors(features_arg, roles, bits):
        return np.array([[roles, bits]]), np.array([[roles + bits]])

    def fake_get_description_length_costs(features_arg, model):
        g, f = model
        roles = int(g[0, 0])
        bits = int(g[0, 1])
        return float(roles * 10 + bits), float(100 - (roles * 10 + bits))

    monkeypatch.setattr(rex, "_get_encoded_role_factors", fake_get_encoded_role_factors)
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_get_description_length_costs)

    model = rex._select_model(features)

    np.testing.assert_array_equal(model[0], np.array([[2, 1]]))
    np.testing.assert_array_equal(model[1], np.array([[3]]))


def test_select_model_ignores_value_error_from_oversized_bits(monkeypatch):
    rex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 3))
    features = pd.DataFrame(np.ones((2, 2)))

    calls = []

    def fake_get_encoded_role_factors(features_arg, roles, bits):
        calls.append((roles, bits))
        if bits == 3:
            raise ValueError("too many bits")
        return np.array([[roles, bits]]), np.array([[0]])

    def fake_get_description_length_costs(features_arg, model):
        return float(model[0][0, 0]), 0.0

    monkeypatch.setattr(rex, "_get_encoded_role_factors", fake_get_encoded_role_factors)
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_get_description_length_costs)

    model = rex._select_model(features)

    assert (2, 3) in calls or (3, 3) in calls
    np.testing.assert_array_equal(model[0], np.array([[2, 1]]))


def test_select_model_tie_breaks_by_first_smallest_index(monkeypatch):
    rex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 3))
    features = pd.DataFrame(np.ones((3, 3)))

    def fake_get_encoded_role_factors(features_arg, roles, bits):
        return np.array([[roles, bits]]), np.array([[roles + bits]])

    def fake_get_description_length_costs(features_arg, model):
        return 1.0, 1.0

    monkeypatch.setattr(rex, "_get_encoded_role_factors", fake_get_encoded_role_factors)
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_get_description_length_costs)

    model = rex._select_model(features)

    np.testing.assert_array_equal(model[0], np.array([[2, 1]]))


def test_explain_raises_not_implemented_error():
    rex = RoleExtractor()
    with pytest.raises(NotImplementedError, match="Role explanation"):
        rex.explain()
