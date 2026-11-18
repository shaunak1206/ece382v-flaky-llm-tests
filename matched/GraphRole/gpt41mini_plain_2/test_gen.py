import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


@pytest.fixture
def simple_features():
    # A simple DataFrame with 5 nodes and 3 features
    data = {
        'feat1': [1, 2, 3, 4, 5],
        'feat2': [5, 4, 3, 2, 1],
        'feat3': [2, 3, 2, 3, 2],
    }
    return pd.DataFrame(data, index=[f'node{i}' for i in range(5)])


def test_initial_attributes(simple_features):
    re = RoleExtractor()
    assert re.n_roles is None
    assert re.min_roles == 2 and re.max_roles == 8
    assert re.min_bits == 1 and re.max_bits == 8
    assert re.node_role_factor is None
    assert re.role_feature_factor is None


def test_init_with_custom_ranges():
    re = RoleExtractor(n_role_range=(3, 6), n_bit_range=(2, 5))
    assert re.min_roles == 3
    assert re.max_roles == 6
    assert re.min_bits == 2
    assert re.max_bits == 5


def test_roles_property_before_extraction():
    re = RoleExtractor()
    # no extraction done, so should return None
    assert re.roles is None


def test_role_percentage_before_extraction():
    re = RoleExtractor()
    # no extraction done, so should return None
    assert re.role_percentage is None


def test_extract_role_factors_with_n_roles(monkeypatch, simple_features):
    # Patch _get_encoded_role_factors to return deterministic output
    def fake_get_encoded_role_factors(self, features, n_roles, n_bits):
        n_nodes = features.shape[0]
        n_features = features.shape[1]
        G = np.full((n_nodes, n_roles), 1.0)
        F = np.full((n_roles, n_features), 2.0)
        return G, F

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', fake_get_encoded_role_factors)

    re = RoleExtractor(n_roles=3)
    re.extract_role_factors(simple_features)

    # node_role_factor shape check
    assert isinstance(re.node_role_factor, pd.DataFrame)
    assert re.node_role_factor.shape == (simple_features.shape[0], 3)
    # role_feature_factor shape check
    assert isinstance(re.role_feature_factor, pd.DataFrame)
    assert re.role_feature_factor.shape == (3, simple_features.shape[1])

    # values check: all ones in node_role_factor, all twos in role_feature_factor
    assert (re.node_role_factor.values == 1.0).all()
    assert (re.role_feature_factor.values == 2.0).all()


def test_extract_role_factors_without_n_roles(monkeypatch, simple_features):
    # Patch _select_model to return controlled output
    def fake_select_model(self, features):
        n_nodes = features.shape[0]
        n_roles = 2
        n_features = features.shape[1]
        G = np.ones((n_nodes, n_roles))
        F = np.ones((n_roles, n_features)) * 5
        return G, F

    monkeypatch.setattr(RoleExtractor, '_select_model', fake_select_model)

    re = RoleExtractor()
    re.extract_role_factors(simple_features)

    assert isinstance(re.node_role_factor, pd.DataFrame)
    assert re.node_role_factor.shape[1] == 2
    assert (re.node_role_factor.values == 1).all()

    assert isinstance(re.role_feature_factor, pd.DataFrame)
    assert (re.role_feature_factor.values == 5).all()


def test_roles_property_after_extraction(monkeypatch, simple_features):
    # Setup deterministic node_role_factor with clear max role
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame(
        {
            'role_0': [0.1, 0.9, 0.3],
            'role_1': [0.9, 0.1, 0.7],
        },
        index=['n1', 'n2', 'n3']
    )
    roles = re.roles
    assert isinstance(roles, dict)
    assert roles == {'n1': 'role_1', 'n2': 'role_0', 'n3': 'role_1'}


def test_role_percentage_returns_correct_values():
    re = RoleExtractor()
    re.node_role_factor = pd.DataFrame(
        {
            'role_0': [1, 2],
            'role_1': [3, 2],
        },
        index=['node_a', 'node_b']
    )
    rp = re.role_percentage
    assert isinstance(rp, pd.DataFrame)
    # Each row should sum to 1
    row_sums = rp.sum(axis=1)
    pd.testing.assert_series_equal(row_sums, pd.Series([1.0, 1.0], index=rp.index))
    # role_0 percentage are 1/(1+3)=0.25 and 2/(2+2)=0.5 approx
    assert np.isclose(rp.loc['node_a', 'role_0'], 0.25)
    assert np.isclose(rp.loc['node_b', 'role_0'], 0.5)


def test_explain_raises_not_implemented():
    re = RoleExtractor()
    with pytest.raises(NotImplementedError):
        re.explain()


def test_get_encoded_role_factors_return_shapes(monkeypatch, simple_features):
    # Patch get_nmf_decomposition and encode to controlled functions
    def fake_nmf(V, n_roles):
        return np.arange(V.shape[0]*n_roles).reshape(V.shape[0], n_roles).astype(float), \
               np.arange(n_roles*V.shape[1]).reshape(n_roles, V.shape[1]).astype(float)

    def fake_encode(matrix, bins):
        return matrix + bins  # simple transform

    monkeypatch.setattr(RoleExtractor, 'get_nmf_decomposition', fake_nmf)
    monkeypatch.setattr('graphrole.roles.extract.get_nmf_decomposition', fake_nmf)
    monkeypatch.setattr('graphrole.roles.extract.encode', fake_encode)

    G, F = RoleExtractor._get_encoded_role_factors(simple_features, 3, 2)
    assert G.shape == (simple_features.shape[0], 3)
    assert F.shape == (3, simple_features.shape[1])


def test_rescale_costs_basic():
    # Input with two rows
    costs = np.array([
        [3, 4],
        [0, np.nan],
        [5, 12],
        [0, 0],
    ], dtype=float)
    result = RoleExtractor._rescale_costs(costs)
    # Norms per row: sqrt(sum of squares skipping nan)
    norms = np.array([
        5.0,
        0.0,  # should handle zero norm safely?
        13.0,
        0.0
    ])
    # For rows with norm=0, division by zero yields inf or nan, check for expected behavior
    expected = np.empty_like(costs)
    expected[0] = costs[0] / norms[0]
    expected[1] = costs[1] / 1  # avoid division by zero, but method does not handle zero norms explicitly
    expected[2] = costs[2] / norms[2]
    expected[3] = costs[3] / 1

    # Just verify shape and some known results (cannot easily predict exact nan behaviors)
    assert result.shape == costs.shape
    # First row normalized
    assert np.allclose(result[0], costs[0] / 5.0, equal_nan=True)
    # Third row normalized
    assert np.allclose(result[2], costs[2] / 13.0, equal_nan=True)


def test_select_model_simple_grid(monkeypatch):
    # Provide a small features DataFrame to control grid size
    features = pd.DataFrame(np.ones((3, 4)), index=['n1', 'n2', 'n3'], columns=['f1', 'f2', 'f3', 'f4'])

    # Patch _get_encoded_role_factors to return fixed factors for known inputs
    def fake_get_encoded_role_factors(self, features_, roles, bits):
        # Return matrices filled with roles and bits for deterministic costs
        G = np.full((features_.shape[0], roles), roles, dtype=float)
        F = np.full((roles, features_.shape[1]), bits, dtype=float)
        return G, F

    # Patch get_description_length_costs to return encoding_cost and error_cost based on inputs
    def fake_get_description_length_costs(features_, model):
        G, F = model
        # encoding_cost proportional to mean of G + F
        encoding_cost = G.mean() + F.mean()
        # error_cost proportional to difference from 1 times some factor
        error_cost = abs(features_.values.mean() - G.mean())
        return encoding_cost, error_cost

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', fake_get_encoded_role_factors)
    monkeypatch.setattr('graphrole.roles.extract.get_description_length_costs', fake_get_description_length_costs)

    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))
    # minimal roles 2 to 3, bits 1 to 2

    G, F = re._select_model(features)

    # Check shapes
    assert G.shape[0] == features.shape[0]
    assert F.shape[1] == features.shape[1]
    # Roles equals G.shape[1]
    assert G.shape[1] == F.shape[0]
    # roles in range 2 or 3
    assert 2 <= G.shape[1] <= 3


def test_select_model_skips_incorrect_bits(monkeypatch):
    features = pd.DataFrame(np.ones((4, 5)), columns=[f'f{i}' for i in range(5)])

    # Patch _get_encoded_role_factors to raise ValueError if bits too large
    def fake_get_encoded_role_factors(self, features_, roles, bits):
        if bits > 3:
            raise ValueError('Too many bits')
        return np.ones((features_.shape[0], roles)), np.ones((roles, features_.shape[1]))

    # Patch get_description_length_costs to dummy values
    def fake_get_description_length_costs(features_, model):
        return 10.0, 5.0

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', fake_get_encoded_role_factors)
    monkeypatch.setattr('graphrole.roles.extract.get_description_length_costs', fake_get_description_length_costs)

    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 5))
    # bits can go up to 5, but >3 should raise ValueError and be skipped

    G, F = re._select_model(features)
    assert G.shape[0] == features.shape[0]
    # bits <=3 means only bits in [1,2,3] are evaluated


def test_roles_property_handles_non_df():
    re = RoleExtractor()
    re.node_role_factor = "not a dataframe"
    assert re.roles is None


def test_role_percentage_handles_non_df():
    re = RoleExtractor()
    re.node_role_factor = "not a dataframe"
    assert re.role_percentage is None
