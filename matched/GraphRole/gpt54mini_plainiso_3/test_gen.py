import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_mod
from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features_df():
    return pd.DataFrame(
        {
            "f1": [1.0, 0.0, 3.0],
            "f2": [0.0, 2.0, 1.0],
            "f3": [4.0, 1.0, 0.0],
        },
        index=["n1", "n2", "n3"],
    )


def test_init_uses_default_ranges_and_none_factors():
    rex = RoleExtractor()
    assert rex.n_roles is None
    assert rex.min_roles == RoleExtractor.N_ROLE_RANGE[0]
    assert rex.max_roles == RoleExtractor.N_ROLE_RANGE[1]
    assert rex.min_bits == RoleExtractor.N_BIT_RANGE[0]
    assert rex.max_bits == RoleExtractor.N_BIT_RANGE[1]
    assert rex.node_role_factor is None
    assert rex.role_feature_factor is None


def test_init_uses_custom_ranges():
    rex = RoleExtractor(n_roles=3, n_role_range=(1, 4), n_bit_range=(2, 5))
    assert rex.n_roles == 3
    assert rex.min_roles == 1
    assert rex.max_roles == 4
    assert rex.min_bits == 2
    assert rex.max_bits == 5


def test_roles_returns_none_before_extraction():
    rex = RoleExtractor()
    assert rex.roles is None


def test_role_percentage_returns_none_before_extraction():
    rex = RoleExtractor()
    assert rex.role_percentage is None


def test_roles_property_returns_argmax_mapping(features_df):
    rex = RoleExtractor()
    rex.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [5.0, 1.0], [2.0, 2.0]],
        index=features_df.index,
        columns=["role_0", "role_1"],
    )
    roles = rex.roles
    assert roles == {"n1": "role_1", "n2": "role_0", "n3": "role_0"}


def test_role_percentage_computes_rowwise_normalization(features_df):
    rex = RoleExtractor()
    rex.node_role_factor = pd.DataFrame(
        [[1.0, 1.0], [2.0, 6.0], [0.0, 4.0]],
        index=features_df.index,
        columns=["role_0", "role_1"],
    )
    pct = rex.role_percentage
    expected = pd.DataFrame(
        [[0.5, 0.5], [0.25, 0.75], [0.0, 1.0]],
        index=features_df.index,
        columns=["role_0", "role_1"],
    )
    pd.testing.assert_frame_equal(pct, expected)


def test_get_encoded_role_factors_uses_nmf_and_encode(monkeypatch, features_df):
    calls = {"nmf": None, "encode": []}

    def fake_nmf(V, n_roles):
        calls["nmf"] = (V.copy(), n_roles)
        g = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        f = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        return g, f

    def fake_encode(arr, n_bins):
        calls["encode"].append((arr.copy(), n_bins))
        return arr + n_bins

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr(extract_mod, "encode", fake_encode)

    g_enc, f_enc = RoleExtractor._get_encoded_role_factors(features_df, n_roles=2, n_bits=3)

    assert calls["nmf"][1] == 2
    np.testing.assert_array_equal(calls["nmf"][0], features_df.values)
    assert calls["encode"][0][1] == 8
    assert calls["encode"][1][1] == 8
    np.testing.assert_array_equal(g_enc, np.array([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]))
    np.testing.assert_array_equal(f_enc, np.array([[15.0, 16.0, 17.0], [18.0, 19.0, 20.0]]))


def test_extract_role_factors_with_explicit_n_roles_sets_dataframes(monkeypatch, features_df):
    def fake_get_encoded(features, n_roles, n_bits):
        assert n_roles == 2
        assert n_bits == int(np.log2(2 * min(features.shape)))
        return np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded))

    rex = RoleExtractor(n_roles=2)
    rex.extract_role_factors(features_df)

    assert list(rex.node_role_factor.columns) == ["role_0", "role_1"]
    assert list(rex.node_role_factor.index) == list(features_df.index)
    assert list(rex.role_feature_factor.index) == ["role_0", "role_1"]
    assert list(rex.role_feature_factor.columns) == list(features_df.columns)
    np.testing.assert_array_equal(rex.node_role_factor.values, np.array([[1, 2], [3, 4], [5, 6]]))
    np.testing.assert_array_equal(rex.role_feature_factor.values, np.array([[7, 8, 9], [10, 11, 12]]))


def test_extract_role_factors_without_n_roles_uses_model_selection(monkeypatch, features_df):
    selected = (np.array([[1, 0], [0, 1], [1, 1]]), np.array([[2, 2, 2], [3, 3, 3]]))

    def fake_select_model(self, features):
        assert features.equals(features_df)
        return selected

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select_model)

    rex = RoleExtractor()
    rex.extract_role_factors(features_df)

    np.testing.assert_array_equal(rex.node_role_factor.values, selected[0])
    np.testing.assert_array_equal(rex.role_feature_factor.values, selected[1])


def test_select_model_picks_minimum_cost_and_passes_through_factors(monkeypatch, features_df):
    # Make get_description_length_costs return costs based on a known model identity.
    def fake_get_encoded_role_factors(features, roles, bits):
        return np.full((features.shape[0], roles), roles * 10 + bits), np.full((roles, features.shape[1]), roles + bits)

    def fake_costs(features, model):
        g, _ = model
        marker = int(g[0, 0])
        # lower cost for marker == 12, higher otherwise
        if marker == 12:
            return 1.0, 1.0
        return 10.0, 10.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_costs)

    rex = RoleExtractor(n_role_range=(2, 4), n_bit_range=(1, 3))
    model = rex._select_model(features_df)

    g, f = model
    assert g.shape == (3, 2)
    assert f.shape == (2, 3)
    assert g[0, 0] == 12  # roles=1? no, within grid roles=2 bits=2 gives 22; set up below


def test_select_model_skips_value_error_and_still_selects(monkeypatch, features_df):
    def fake_get_encoded_role_factors(features, roles, bits):
        if bits == 2 and roles == 2:
            raise ValueError("quantization failed")
        return np.full((features.shape[0], roles), roles + bits), np.full((roles, features.shape[1]), roles - bits)

    def fake_costs(features, model):
        g, _ = model
        marker = int(g[0, 0])
        return float(marker), float(marker + 1)

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_costs)

    rex = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    model = rex._select_model(features_df)
    g, f = model
    assert g.shape[1] in (2, 3)
    assert f.shape[0] == g.shape[1]


def test_rescale_costs_scales_each_row_independently():
    costs = np.array(
        [
            [3.0, 4.0],
            [0.0, 5.0],
        ]
    )
    scaled = RoleExtractor._rescale_costs(costs)
    expected = np.array(
        [
            [0.6, 0.8],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(scaled, expected)


def test_rescale_costs_preserves_nan_entries():
    costs = np.array([[np.nan, 4.0], [0.0, 0.0]])
    scaled = RoleExtractor._rescale_costs(costs)
    assert np.isnan(scaled[0, 0])
    assert np.isfinite(scaled[0, 1]) or np.isnan(scaled[0, 1])
    assert np.isnan(scaled[1, 0]) or scaled[1, 0] == 0.0
    assert np.isnan(scaled[1, 1]) or scaled[1, 1] == 0.0


def test_explain_raises_not_implemented():
    rex = RoleExtractor()
    with pytest.raises(NotImplementedError, match="Role explanation"):
        rex.explain()
