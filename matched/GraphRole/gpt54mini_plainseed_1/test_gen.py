import numpy as np
import pandas as pd
import pytest

import graphrole.roles.extract as extract_mod
from graphrole.roles.extract import RoleExtractor


def make_features():
    return pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=["n1", "n2", "n3"],
        columns=["f1", "f2"],
    )


def test_init_uses_default_ranges_and_none_factors():
    re = RoleExtractor()
    assert re.n_roles is None
    assert re.min_roles == RoleExtractor.N_ROLE_RANGE[0]
    assert re.max_roles == RoleExtractor.N_ROLE_RANGE[1]
    assert re.min_bits == RoleExtractor.N_BIT_RANGE[0]
    assert re.max_bits == RoleExtractor.N_BIT_RANGE[1]
    assert re.node_role_factor is None
    assert re.role_feature_factor is None


def test_init_overrides_ranges():
    re = RoleExtractor(n_roles=3, n_role_range=(4, 6), n_bit_range=(2, 5))
    assert re.n_roles == 3
    assert re.min_roles == 4
    assert re.max_roles == 6
    assert re.min_bits == 2
    assert re.max_bits == 5


def test_roles_returns_none_when_no_factor_dataframe():
    re = RoleExtractor()
    assert re.roles is None


def test_role_percentage_returns_none_when_no_factor_dataframe():
    re = RoleExtractor()
    assert re.role_percentage is None


def test_roles_property_returns_column_with_max_value_per_row():
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame(
        [[1.0, 5.0, 2.0], [7.0, 0.1, 7.1], [0.0, 0.0, 3.0]],
        index=["n1", "n2", "n3"],
        columns=["role_0", "role_1", "role_2"],
    )
    assert re.roles == {"n1": "role_1", "n2": "role_2", "n3": "role_2"}


def test_role_percentage_computes_row_normalized_dataframe():
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=["n1", "n2"],
        columns=["role_0", "role_1"],
    )
    result = re.role_percentage
    expected = pd.DataFrame(
        [[0.25, 0.75], [0.5, 0.5]],
        index=["n1", "n2"],
        columns=["role_0", "role_1"],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_get_encoded_role_factors_delegates_to_nmf_and_encode(monkeypatch):
    calls = {"nmf": None, "encode": []}

    def fake_nmf(V, n_roles):
        calls["nmf"] = (V.copy(), n_roles)
        return np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]])

    def fake_encode(arr, n_bins):
        calls["encode"].append((arr.copy(), n_bins))
        return arr + n_bins

    monkeypatch.setattr(extract_mod, "get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr(extract_mod, "encode", fake_encode)

    features = make_features()
    g_enc, f_enc = RoleExtractor._get_encoded_role_factors(features, n_roles=2, n_bits=3)

    assert calls["nmf"][1] == 2
    np.testing.assert_array_equal(calls["nmf"][0], features.values)
    assert len(calls["encode"]) == 2
    np.testing.assert_array_equal(g_enc, np.array([[9, 10], [11, 12]]))
    np.testing.assert_array_equal(f_enc, np.array([[13, 14, 15], [16, 17, 18]]))


def test_extract_role_factors_with_fixed_n_roles_sets_dataframes(monkeypatch):
    def fake_get_encoded_role_factors(features, n_roles, n_bits):
        assert n_roles == 3
        assert isinstance(n_bits, int)
        return np.array([[1, 0, 0], [0, 1, 0]]), np.array([[1, 2], [3, 4], [5, 6]])

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))

    re = RoleExtractor(n_roles=3)
    features = make_features().iloc[:2]
    re.extract_role_factors(features)

    assert list(re.node_role_factor.index) == ["n1", "n2"]
    assert list(re.node_role_factor.columns) == ["role_0", "role_1", "role_2"]
    assert list(re.role_feature_factor.index) == ["role_0", "role_1", "role_2"]
    assert list(re.role_feature_factor.columns) == ["f1", "f2"]
    np.testing.assert_array_equal(re.node_role_factor.values, np.array([[1, 0, 0], [0, 1, 0]]))


def test_extract_role_factors_uses_select_model_when_n_roles_is_none(monkeypatch):
    selected = (np.array([[1, 0], [0, 1]]), np.array([[2, 3], [4, 5]]))

    def fake_select_model(features):
        assert features.equals(make_features())
        return selected

    monkeypatch.setattr(RoleExtractor, "_select_model", fake_select_model)

    re = RoleExtractor(n_roles=None)
    features = make_features()
    re.extract_role_factors(features)

    np.testing.assert_array_equal(re.node_role_factor.values, selected[0])
    np.testing.assert_array_equal(re.role_feature_factor.values, selected[1])
    assert list(re.node_role_factor.columns) == ["role_0", "role_1"]


def test_extract_role_factors_role_labels_follow_number_of_columns(monkeypatch):
    monkeypatch.setattr(
        RoleExtractor,
        "_get_encoded_role_factors",
        staticmethod(lambda features, n_roles, n_bits: (np.ones((3, 4)), np.ones((4, 2)))),
    )
    re = RoleExtractor(n_roles=4)
    features = make_features()
    re.extract_role_factors(features)
    assert list(re.node_role_factor.columns) == ["role_0", "role_1", "role_2", "role_3"]
    assert list(re.role_feature_factor.index) == ["role_0", "role_1", "role_2", "role_3"]


def test_select_model_picks_min_cost_model(monkeypatch):
    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))

    def fake_get_encoded_role_factors(features, roles, bits):
        return np.full((2, roles), roles + bits), np.full((roles, 2), roles - bits)

    def fake_costs(features, model):
        g, f = model
        roles = g.shape[1]
        bits = int(np.max(g[0]) - roles)
        if roles == 2 and bits == 1:
            return 1.0, 1.0
        if roles == 2 and bits == 2:
            return 5.0, 5.0
        if roles == 3 and bits == 1:
            return 9.0, 9.0
        return 10.0, 10.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_costs)

    features = make_features()
    model = re._select_model(features)

    np.testing.assert_array_equal(model[0], np.full((2, 2), 3))
    np.testing.assert_array_equal(model[1], np.full((2, 2), 1))


def test_select_model_skips_value_error_and_still_selects_available_model(monkeypatch):
    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))

    def fake_get_encoded_role_factors(features, roles, bits):
        if roles == 2 and bits == 2:
            raise ValueError("too many bits")
        return np.full((2, roles), roles * 10 + bits), np.full((roles, 2), bits)

    def fake_costs(features, model):
        g, _ = model
        roles = g.shape[1]
        bits = int(g[0, 0] - roles * 10)
        if roles == 2 and bits == 1:
            return 1.0, 1.0
        return 5.0, 5.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", fake_costs)

    model = re._select_model(make_features())
    np.testing.assert_array_equal(model[0], np.full((2, 2), 21))


def test_select_model_uses_only_feasible_search_space(monkeypatch):
    re = RoleExtractor(n_role_range=(2, 8), n_bit_range=(1, 5))

    seen = []

    def fake_get_encoded_role_factors(features, roles, bits):
        seen.append((roles, bits))
        return np.ones((2, roles)), np.ones((roles, 2))

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr(extract_mod, "get_description_length_costs", lambda features, model: (1.0, 1.0))

    re._select_model(make_features().iloc[:, :3])

    assert all(2 <= r <= 3 for r, b in seen)
    assert all(1 <= b <= 5 for r, b in seen)
    assert set(seen) == {(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)}


def test_rescale_costs_normalizes_each_row_independently():
    costs = np.array([[3.0, 4.0], [0.0, 5.0]])
    rescaled = RoleExtractor._rescale_costs(costs)
    expected = np.array([[0.6, 0.8], [0.0, 1.0]])
    np.testing.assert_allclose(rescaled, expected)


def test_rescale_costs_preserves_nan_entries():
    costs = np.array([[1.0, np.nan], [2.0, 2.0]])
    rescaled = RoleExtractor._rescale_costs(costs)
    assert np.isnan(rescaled[0, 1])
    assert np.isfinite(rescaled[1, 0])
    assert np.isfinite(rescaled[1, 1])


def test_explain_raises_not_implemented_error():
    re = RoleExtractor()
    with pytest.raises(NotImplementedError, match="Role explanation"):
        re.explain()


def test_extract_role_factors_with_missing_n_roles_uses_model_selection_and_indexes_are_preserved(monkeypatch):
    selected = (np.array([[9, 8], [7, 6], [5, 4]]), np.array([[1, 2], [3, 4]]))

    monkeypatch.setattr(RoleExtractor, "_select_model", lambda self, features: selected)

    re = RoleExtractor()
    features = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=[10, 20, 30], columns=["a", "b"])
    re.extract_role_factors(features)

    assert list(re.node_role_factor.index) == [10, 20, 30]
    assert list(re.role_feature_factor.columns) == ["a", "b"]
    np.testing.assert_array_equal(re.node_role_factor.values, selected[0])
    np.testing.assert_array_equal(re.role_feature_factor.values, selected[1])
