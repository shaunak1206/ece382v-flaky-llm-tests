import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


def test_init_uses_default_ranges_and_none_factors():
    extractor = RoleExtractor()

    assert extractor.n_roles is None
    assert extractor.min_roles == 2
    assert extractor.max_roles == 8
    assert extractor.min_bits == 1
    assert extractor.max_bits == 8
    assert extractor.node_role_factor is None
    assert extractor.role_feature_factor is None


def test_init_uses_custom_ranges():
    extractor = RoleExtractor(
        n_roles=3,
        n_role_range=(4, 6),
        n_bit_range=(2, 5),
    )

    assert extractor.n_roles == 3
    assert extractor.min_roles == 4
    assert extractor.max_roles == 6
    assert extractor.min_bits == 2
    assert extractor.max_bits == 5


def test_roles_returns_none_when_factor_is_missing():
    extractor = RoleExtractor()

    assert extractor.roles is None


def test_role_percentage_returns_none_when_factor_is_missing():
    extractor = RoleExtractor()

    assert extractor.role_percentage is None


def test_roles_returns_argmax_mapping():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [5.0, 2.0], [0.0, 0.0]],
        index=["n1", "n2", "n3"],
        columns=["role_0", "role_1"],
    )

    assert extractor.roles == {"n1": "role_1", "n2": "role_0", "n3": "role_0"}


def test_role_percentage_normalizes_each_row():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=["n1", "n2"],
        columns=["role_0", "role_1"],
    )

    result = extractor.role_percentage

    assert np.allclose(result.loc["n1"].values, [0.25, 0.75])
    assert np.allclose(result.loc["n2"].values, [0.5, 0.5])


def test_get_encoded_role_factors_delegates_to_factor_functions(monkeypatch):
    calls = {}

    def fake_nmf(v, n_roles):
        calls["nmf"] = (v.copy(), n_roles)
        return np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])

    def fake_encode(x, n_bins):
        calls.setdefault("encode", []).append((x.copy(), n_bins))
        return x + n_bins

    monkeypatch.setattr("graphrole.roles.extract.get_nmf_decomposition", fake_nmf)
    monkeypatch.setattr("graphrole.roles.extract.encode", fake_encode)

    features = pd.DataFrame([[10, 11], [12, 13]], columns=["a", "b"])
    g_enc, f_enc = RoleExtractor._get_encoded_role_factors(features, n_roles=2, n_bits=3)

    assert np.array_equal(calls["nmf"][0], features.values)
    assert calls["nmf"][1] == 2
    assert len(calls["encode"]) == 2
    assert np.array_equal(g_enc, np.array([[9.0, 10.0], [11.0, 12.0]]))
    assert np.array_equal(f_enc, np.array([[13.0, 14.0], [15.0, 16.0]]))


def test_extract_role_factors_with_fixed_n_roles_uses_bit_heuristic(monkeypatch):
    extractor = RoleExtractor(n_roles=3)
    features = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["x", "y"])

    seen = {}

    def fake_get_encoded_role_factors(features_arg, n_roles, n_bits):
        seen["args"] = (features_arg.copy(), n_roles, n_bits)
        return np.ones((features_arg.shape[0], n_roles)), np.zeros((n_roles, features_arg.shape[1]))

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))

    extractor.extract_role_factors(features)

    assert seen["args"][1] == 3
    assert seen["args"][2] == int(np.log2(3 * min(features.shape)))
    assert list(extractor.node_role_factor.columns) == ["role_0", "role_1", "role_2"]
    assert list(extractor.node_role_factor.index) == list(features.index)
    assert list(extractor.role_feature_factor.index) == ["role_0", "role_1", "role_2"]
    assert list(extractor.role_feature_factor.columns) == list(features.columns)


def test_extract_role_factors_with_model_selection_uses_selected_model(monkeypatch):
    extractor = RoleExtractor(n_roles=None)
    features = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["f1", "f2"])

    expected = (np.array([[1.0], [2.0]]), np.array([[3.0, 4.0]]))
    monkeypatch.setattr(RoleExtractor, "_select_model", lambda self, feats: expected)

    extractor.extract_role_factors(features)

    assert extractor.node_role_factor.shape == (2, 1)
    assert extractor.role_feature_factor.shape == (1, 2)
    assert extractor.node_role_factor.loc["a", "role_0"] == 1.0
    assert extractor.role_feature_factor.loc["role_0", "f1"] == 3.0


def test_select_model_picks_lowest_total_cost(monkeypatch):
    extractor = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    features = pd.DataFrame(np.zeros((4, 4)), columns=list("abcd"))

    def fake_get_encoded_role_factors(features_arg, roles, bits):
        return np.full((features_arg.shape[0], roles), roles), np.full((roles, features_arg.shape[1]), bits)

    def fake_costs(features_arg, model):
        g, f = model
        roles = g.shape[1]
        bits = int(np.max(f))
        cost_map = {
            (2, 1): (10.0, 10.0),
            (2, 2): (1.0, 1.0),
            (3, 1): (5.0, 5.0),
            (3, 2): (9.0, 9.0),
        }
        return cost_map[(roles, bits)]

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr("graphrole.roles.extract.get_description_length_costs", fake_costs)

    model = extractor._select_model(features)

    assert model[0].shape == (4, 2)
    assert model[1].shape == (2, 4)
    assert np.all(model[0] == 2)
    assert np.all(model[1] == 2)


def test_select_model_skips_value_error_from_overlarge_bits(monkeypatch):
    extractor = RoleExtractor(n_role_range=(2, 2), n_bit_range=(1, 2))
    features = pd.DataFrame(np.zeros((3, 3)), columns=list("abc"))

    def fake_get_encoded_role_factors(features_arg, roles, bits):
        if bits == 2:
            raise ValueError("too many bits")
        return np.ones((features_arg.shape[0], roles)), np.ones((roles, features_arg.shape[1]))

    def fake_costs(features_arg, model):
        return 2.0, 2.0

    monkeypatch.setattr(RoleExtractor, "_get_encoded_role_factors", staticmethod(fake_get_encoded_role_factors))
    monkeypatch.setattr("graphrole.roles.extract.get_description_length_costs", fake_costs)

    model = extractor._select_model(features)

    assert model[0].shape == (3, 2)


def test_rescale_costs_normalizes_rowwise():
    costs = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])

    result = RoleExtractor._rescale_costs(costs)

    assert np.allclose(result[0], [0.6, 0.8])
    assert np.isnan(result[1]).all()
    assert np.allclose(result[2], [0.6, 0.8])


def test_rescale_costs_preserves_nan_entries():
    costs = np.array([[1.0, np.nan], [np.nan, 2.0]])

    result = RoleExtractor._rescale_costs(costs)

    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 0])


def test_explain_raises_not_implemented_error():
    extractor = RoleExtractor()

    with pytest.raises(NotImplementedError, match="Role explanation"):
        extractor.explain()


def test_extract_role_factors_overwrites_previous_factors(monkeypatch):
    extractor = RoleExtractor(n_roles=2)
    extractor.node_role_factor = pd.DataFrame([[9]], columns=["old"])
    extractor.role_feature_factor = pd.DataFrame([[9]], index=["old"])

    features = pd.DataFrame([[1, 2], [3, 4]], columns=["f1", "f2"])

    monkeypatch.setattr(
        RoleExtractor,
        "_get_encoded_role_factors",
        staticmethod(lambda features, n_roles, n_bits: (np.zeros((2, 2)), np.ones((2, 2)))),
    )

    extractor.extract_role_factors(features)

    assert list(extractor.node_role_factor.columns) == ["role_0", "role_1"]
    assert list(extractor.role_feature_factor.index) == ["role_0", "role_1"]
    assert extractor.node_role_factor.iloc[0].sum() == 0.0
    assert extractor.role_feature_factor.iloc[0].sum() == 2.0
