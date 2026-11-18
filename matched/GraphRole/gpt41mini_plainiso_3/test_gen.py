import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def sample_features():
    return pd.DataFrame({
        "feat1": [0.1, 0.2, 0.3, 0.4],
        "feat2": [1.0, 0.9, 0.8, 0.7],
        "feat3": [5, 6, 7, 8],
    }, index=["node1", "node2", "node3", "node4"])


def test_init_defaults():
    re = RoleExtractor()
    assert re.n_roles is None
    assert re.min_roles == 2
    assert re.max_roles == 8
    assert re.min_bits == 1
    assert re.max_bits == 8
    assert re.node_role_factor is None
    assert re.role_feature_factor is None


def test_init_with_custom_ranges():
    re = RoleExtractor(n_roles=4, n_role_range=(3, 5), n_bit_range=(2, 4))
    assert re.n_roles == 4
    assert re.min_roles == 3
    assert re.max_roles == 5
    assert re.min_bits == 2
    assert re.max_bits == 4


def test_roles_property_none_when_no_factor():
    re = RoleExtractor()
    assert re.roles is None


def test_role_percentage_property_none_when_no_factor():
    re = RoleExtractor()
    assert re.role_percentage is None


def test_extract_role_factors_sets_factors_and_labels(sample_features):
    re = RoleExtractor(n_roles=3)
    re.extract_role_factors(sample_features)
    # Factors must be set and must have shape consistent with roles/features
    assert isinstance(re.node_role_factor, pd.DataFrame)
    assert isinstance(re.role_feature_factor, pd.DataFrame)
    assert re.node_role_factor.shape[0] == sample_features.shape[0]
    assert re.node_role_factor.shape[1] == 3
    assert re.role_feature_factor.shape[0] == 3
    assert re.role_feature_factor.shape[1] == sample_features.shape[1]

    # Index and columns correctness
    assert all(re.node_role_factor.index == sample_features.index)
    assert all(col.startswith("role_") for col in re.node_role_factor.columns)
    assert all(col.startswith("role_") for col in re.role_feature_factor.index)
    assert all(re.role_feature_factor.columns == sample_features.columns)


def test_roles_property_returns_idxmax_mapping(sample_features):
    re = RoleExtractor(n_roles=3)
    re.extract_role_factors(sample_features)
    roles = re.roles
    assert isinstance(roles, dict)
    assert set(roles.keys()) == set(sample_features.index)
    for val in roles.values():
        # idxmax returns column label like "role_x"
        assert val in re.node_role_factor.columns or isinstance(val, (np.generic, float))


def test_role_percentage_property_returns_correct_percentages(sample_features):
    re = RoleExtractor(n_roles=3)
    re.extract_role_factors(sample_features)
    percentages = re.role_percentage
    assert isinstance(percentages, pd.DataFrame)
    # All rows should sum to 1 (within tolerance)
    row_sums = percentages.sum(axis=1)
    assert np.allclose(row_sums.values, np.ones_like(row_sums.values))


def test_explain_raises_not_implemented():
    re = RoleExtractor()
    with pytest.raises(NotImplementedError):
        re.explain()


def test__get_encoded_role_factors_returns_ndarrays(sample_features):
    n_roles = 2
    n_bits = 3
    G, F = RoleExtractor._get_encoded_role_factors(sample_features, n_roles, n_bits)
    # Should be numpy arrays
    assert isinstance(G, np.ndarray)
    assert isinstance(F, np.ndarray)
    # Shapes should be consistent with NMF factors: G (nodes x roles), F (roles x features)
    assert G.shape[0] == sample_features.shape[0]
    assert G.shape[1] == n_roles
    assert F.shape[0] == n_roles
    assert F.shape[1] == sample_features.shape[1]


def test__rescale_costs_basic_behavior():
    # Create a cost matrix with some values and NaNs
    costs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [np.nan, np.nan, np.nan],
    ])
    scaled = RoleExtractor._rescale_costs(costs)
    # The shape is unchanged
    assert scaled.shape == costs.shape
    # NaNs should remain NaN
    assert np.isnan(scaled[2, 0])
    # Rows with values are scaled such that norm is 1 (or close due to nan)
    norms = np.sqrt(np.nansum(np.square(costs), axis=1))
    # For rows with valid norms, scaling returns costs / norm
    expected = costs / norms.reshape(-1, 1)
    np.testing.assert_allclose(scaled[:2], expected[:2], equal_nan=True)


def test__select_model_returns_factors_within_range(sample_features):
    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    # forcibly calling private method; it should return two ndarrays of matching shape
    node_roles, role_features = re._select_model(sample_features)
    assert isinstance(node_roles, np.ndarray)
    assert isinstance(role_features, np.ndarray)
    # Shape checks — number of roles matches columns of node_roles and rows of role_features
    n_roles = node_roles.shape[1]
    assert n_roles == role_features.shape[0]
    # Number of nodes and features check
    assert node_roles.shape[0] == sample_features.shape[0]
    assert role_features.shape[1] == sample_features.shape[1]


def test_extract_role_factors_uses_select_model_when_n_roles_none(sample_features):
    re = RoleExtractor()
    re.extract_role_factors(sample_features)
    # Factors should be set
    assert isinstance(re.node_role_factor, pd.DataFrame)
    assert isinstance(re.role_feature_factor, pd.DataFrame)


def test_extract_role_factors_uses_encoded_role_factors_when_n_roles_set(sample_features):
    n_roles = 4
    re = RoleExtractor(n_roles=n_roles)
    re.extract_role_factors(sample_features)
    # The number of roles should equal n_roles (or close, log2 rounding issues)
    assert re.node_role_factor.shape[1] == n_roles


def test__select_model_skips_invalid_bits(sample_features):
    # Set max_bits high so that bits > log2 roles*n_nodes triggers ValueError internally
    re = RoleExtractor(n_role_range=(2, 2), n_bit_range=(1, 20))
    # The grid search should handle ValueError internally and not fail
    node_roles, role_features = re._select_model(sample_features)
    assert node_roles.shape[0] == sample_features.shape[0]
    assert role_features.shape[1] == sample_features.shape[1]


def test_roles_property_returns_none_if_factor_columns_empty():
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame()
    assert re.roles is None


def test_role_percentage_property_returns_none_if_factor_empty():
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame()
    assert re.role_percentage is None


def test_extract_role_factors_preserves_index_and_columns(sample_features):
    re = RoleExtractor(n_roles=3)
    re.extract_role_factors(sample_features)
    # Check index unchanged and columns labeled role_0..role_{n-1}
    assert all(re.node_role_factor.index == sample_features.index)
    assert all(col == f"role_{i}" for i, col in enumerate(re.node_role_factor.columns))
    assert all(col == f"role_{i}" for i, col in enumerate(re.role_feature_factor.index))
    assert all(re.role_feature_factor.columns == sample_features.columns)
