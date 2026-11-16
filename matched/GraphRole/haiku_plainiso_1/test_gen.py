from unittest import mock

import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def features():
    return pd.DataFrame(
        np.arange(12, dtype=float).reshape(4, 3) + 1.0,
        index=["a", "b", "c", "d"],
        columns=["f1", "f2", "f3"],
    )


# ---------------------------------------------------------------------------
# __init__ and simple properties
# ---------------------------------------------------------------------------

def test_init_defaults():
    extractor = RoleExtractor()
    assert extractor.n_roles is None
    assert (extractor.min_roles, extractor.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (extractor.min_bits, extractor.max_bits) == RoleExtractor.N_BIT_RANGE
    assert extractor.node_role_factor is None
    assert extractor.role_feature_factor is None


def test_init_custom_ranges_and_n_roles():
    extractor = RoleExtractor(n_roles=3, n_role_range=(2, 5), n_bit_range=(3, 6))
    assert extractor.n_roles == 3
    assert (extractor.min_roles, extractor.max_roles) == (2, 5)
    assert (extractor.min_bits, extractor.max_bits) == (3, 6)


def test_init_none_ranges_fall_back_to_defaults():
    extractor = RoleExtractor(n_role_range=None, n_bit_range=None)
    assert (extractor.min_roles, extractor.max_roles) == RoleExtractor.N_ROLE_RANGE
    assert (extractor.min_bits, extractor.max_bits) == RoleExtractor.N_BIT_RANGE


def test_roles_is_none_before_extraction():
    assert RoleExtractor().roles is None


def test_role_percentage_is_none_before_extraction():
    assert RoleExtractor().role_percentage is None


def test_roles_returns_argmax_role_per_node():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]],
        index=["x", "y", "z"],
        columns=["role_0", "role_1"],
    )
    assert extractor.roles == {"x": "role_1", "y": "role_0", "z": "role_1"}


def test_role_percentage_rows_sum_to_one():
    extractor = RoleExtractor()
    extractor.node_role_factor = pd.DataFrame(
        [[1.0, 3.0], [2.0, 2.0]],
        index=["x", "y"],
        columns=["role_0", "role_1"],
    )
    pct = extractor.role_percentage
    np.testing.assert_allclose(pct.loc["x"].values, [0.25, 0.75])
    np.testing.assert_allclose(pct.loc["y"].values, [0.5, 0.5])
    np.testing.assert_allclose(pct.sum(axis=1).values, [1.0, 1.0])


def test_explain_not_implemented():
    with pytest.raises(NotImplementedError):
        RoleExtractor().explain()


# ---------------------------------------------------------------------------
# _rescale_costs
# ---------------------------------------------------------------------------

def test_rescale_costs_unit_l2_rows():
    costs = np.array([[3.0, 4.0], [1.0, 0.0]])
    result = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(result, [[0.6, 0.8], [1.0, 0.0]])


def test_rescale_costs_ignores_nan_and_preserves_it():
    costs = np.array([[3.0, 4.0, np.nan], [1.0, 0.0, 0.0]])
    result = RoleExtractor._rescale_costs(costs)
    np.testing.assert_allclose(result[0, :2], [0.6, 0.8])
    assert np.isnan(result[0, 2])
    np.testing.assert_allclose(result[1], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# _get_encoded_role_factors
# ---------------------------------------------------------------------------

def test_get_encoded_role_factors_uses_nmf_and_encodes_with_2_pow_bits(features):
    G = np.ones((4, 2))
    F = np.ones((2, 3))
    encoded_G = np.full((4, 2), 7.0)
    encoded_F = np.full((2, 3), 9.0)

    with mock.patch(
        "graphrole.roles.extract.get_nmf_decomposition", return_value=(G, F)
    ) as nmf, mock.patch(
        "graphrole.roles.extract.encode", side_effect=[encoded_G, encoded_F]
    ) as enc:
        result = RoleExtractor._get_encoded_role_factors(features, 2, 3)

    nmf_args = nmf.call_args[0]
    np.testing.assert_array_equal(nmf_args[0], features.values)
    assert nmf_args[1] == 2

    assert enc.call_count == 2
    np.testing.assert_array_equal(enc.call_args_list[0][0][0], G)
    np.testing.assert_array_equal(enc.call_args_list[1][0][0], F)
    assert enc.call_args_list[0][0][1] == 8
    assert enc.call_args_list[1][0][1] == 8

    np.testing.assert_array_equal(result[0], encoded_G)
    np.testing.assert_array_equal(result[1], encoded_F)


# ---------------------------------------------------------------------------
# _select_model
# ---------------------------------------------------------------------------

def _fake_encoded_factors(features, n_roles, n_bits):
    # Fail for bits that cannot be quantized, mimicking the ValueError path
    if n_bits > 2:
        raise ValueError("too many bits")
    G = np.ones((features.shape[0], n_roles))
    F = np.full((n_roles, features.shape[1]), float(n_bits))
    return G, F


def _fake_description_length_costs(features, model):
    G, F = model
    n_roles = G.shape[1]
    n_bits = int(F[0, 0])
    encoding_cost = 1.0
    error_cost = 0.0 if (n_roles == 3 and n_bits == 2) else 1.0
    return encoding_cost, error_cost


def test_select_model_picks_minimum_cost_model(features):
    extractor = RoleExtractor()
    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", side_effect=_fake_encoded_factors
    ), mock.patch(
        "graphrole.roles.extract.get_description_length_costs",
        side_effect=_fake_description_length_costs,
    ):
        G, F = extractor._select_model(features)

    assert G.shape == (features.shape[0], 3)
    assert F[0, 0] == 2


def test_select_model_skips_value_error_combinations(features):
    extractor = RoleExtractor()
    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", side_effect=_fake_encoded_factors
    ), mock.patch(
        "graphrole.roles.extract.get_description_length_costs",
        side_effect=_fake_description_length_costs,
    ) as costs:
        G, F = extractor._select_model(features)

    # Roles 2..3 and bits 1..2 are valid: 4 cost evaluations, the rest skipped
    assert costs.call_count == 4
    assert int(F[0, 0]) in (1, 2)


def test_select_model_respects_custom_role_and_bit_ranges(features):
    extractor = RoleExtractor(n_role_range=(2, 2), n_bit_range=(1, 2))
    seen = []

    def record(features_arg, n_roles, n_bits):
        seen.append((n_roles, n_bits))
        return _fake_encoded_factors(features_arg, n_roles, n_bits)

    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", side_effect=record
    ), mock.patch(
        "graphrole.roles.extract.get_description_length_costs",
        side_effect=_fake_description_length_costs,
    ):
        extractor._select_model(features)

    assert sorted(set(seen)) == [(2, 1), (2, 2)]


# ---------------------------------------------------------------------------
# extract_role_factors
# ---------------------------------------------------------------------------

def test_extract_role_factors_fixed_n_roles_computes_bits_and_calls_factors(features):
    G = np.ones((4, 2))
    F = np.ones((2, 3))
    extractor = RoleExtractor(n_roles=2)

    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", return_value=(G, F)
    ) as factors, mock.patch.object(RoleExtractor, "_select_model") as select:
        extractor.extract_role_factors(features)

    select.assert_not_called()
    args = factors.call_args[0]
    assert args[0] is features
    assert args[1] == 2
    # int(log2(2 * min(4, 3))) == int(log2(6)) == 2
    assert args[2] == 2


def test_extract_role_factors_uses_model_selection_when_n_roles_unset(features):
    G = np.ones((4, 3))
    F = np.ones((3, 3))
    extractor = RoleExtractor()

    with mock.patch.object(
        RoleExtractor, "_select_model", return_value=(G, F)
    ) as select:
        extractor.extract_role_factors(features)

    select.assert_called_once()
    assert select.call_args[0][0] is features


def test_extract_role_factors_builds_labelled_dataframes(features):
    G = np.arange(8, dtype=float).reshape(4, 2)
    F = np.arange(6, dtype=float).reshape(2, 3)
    extractor = RoleExtractor(n_roles=2)

    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", return_value=(G, F)
    ):
        extractor.extract_role_factors(features)

    assert list(extractor.node_role_factor.index) == list(features.index)
    assert list(extractor.node_role_factor.columns) == ["role_0", "role_1"]
    np.testing.assert_array_equal(extractor.node_role_factor.values, G)

    assert list(extractor.role_feature_factor.index) == ["role_0", "role_1"]
    assert list(extractor.role_feature_factor.columns) == list(features.columns)
    np.testing.assert_array_equal(extractor.role_feature_factor.values, F)


def test_extract_role_factors_then_roles_property(features):
    G = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.9, 0.1]])
    F = np.ones((2, 3))
    extractor = RoleExtractor(n_roles=2)

    with mock.patch.object(
        RoleExtractor, "_get_encoded_role_factors", return_value=(G, F)
    ):
        extractor.extract_role_factors(features)

    assert extractor.roles == {
        "a": "role_1",
        "b": "role_0",
        "c": "role_1",
        "d": "role_0",
    }


def test_extract_role_factors_end_to_end_small_real_nmf():
    rng = np.random.RandomState(0)
    data = np.abs(rng.normal(size=(6, 4))) + 0.1
    features = pd.DataFrame(
        data,
        index=["n{}".format(i) for i in range(6)],
        columns=["f{}".format(j) for j in range(4)],
    )
    extractor = RoleExtractor(n_roles=2)
    extractor.extract_role_factors(features)

    assert extractor.node_role_factor.shape == (6, 2)
    assert extractor.role_feature_factor.shape == (2, 4)
    assert set(extractor.roles.keys()) == set(features.index)
    assert set(extractor.roles.values()) <= {"role_0", "role_1"}
