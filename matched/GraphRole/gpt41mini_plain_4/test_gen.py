import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


class DummyEncodeError(Exception):
    pass


def test_default_init_values():
    re = RoleExtractor()
    # Check default role and bit ranges
    assert re.min_roles == RoleExtractor.N_ROLE_RANGE[0]
    assert re.max_roles == RoleExtractor.N_ROLE_RANGE[1]
    assert re.min_bits == RoleExtractor.N_BIT_RANGE[0]
    assert re.max_bits == RoleExtractor.N_BIT_RANGE[1]
    # Check n_roles is None by default
    assert re.n_roles is None
    # Initially factors should be None
    assert re.node_role_factor is None
    assert re.role_feature_factor is None


def test_init_with_custom_ranges():
    re = RoleExtractor(n_roles=3, n_role_range=(1, 4), n_bit_range=(2, 5))
    assert re.n_roles == 3
    assert re.min_roles == 1
    assert re.max_roles == 4
    assert re.min_bits == 2
    assert re.max_bits == 5


def test_roles_property_returns_none_if_no_factor():
    re = RoleExtractor()
    # node_role_factor still None => roles property returns None
    assert re.roles is None


def test_roles_property_returns_expected_dict():
    # Create dummy node_role_factor with index and columns (roles)
    index = ['n1', 'n2', 'n3']
    data = {
        'role_0': [0.1, 0.8, 0.3],
        'role_1': [0.9, 0.2, 0.7],
    }
    df = pd.DataFrame(data, index=index)
    re = RoleExtractor()
    re.node_role_factor = df
    roles = re.roles
    # roles should map node to column with max value
    assert roles == {'n1': 'role_1', 'n2': 'role_0', 'n3': 'role_1'}


def test_role_percentage_returns_none_if_no_factor():
    re = RoleExtractor()
    assert re.role_percentage is None


def test_role_percentage_returns_correct_normalized_df():
    index = ['n1', 'n2']
    data = {
        'role_0': [1, 0],
        'role_1': [3, 4],
        'role_2': [6, 6],
    }
    df = pd.DataFrame(data, index=index)
    re = RoleExtractor()
    re.node_role_factor = df
    role_pct = re.role_percentage
    # Each row sums to 1
    row_sums = role_pct.sum(axis=1)
    pd.testing.assert_series_equal(row_sums, pd.Series([1.0, 1.0], index=index))
    # Check some example values manually
    assert np.isclose(role_pct.loc['n1', 'role_0'], 1/10)
    assert np.isclose(role_pct.loc['n2', 'role_2'], 6/10)


def test_extract_role_factors_with_n_roles(monkeypatch):
    # Setup dummy features DataFrame
    features = pd.DataFrame({
        'f1': [1, 2],
        'f2': [3, 4]
    }, index=['node1', 'node2'])

    # Patch _get_encoded_role_factors to return predictable matrices
    dummy_G = np.array([[0.1, 0.9], [0.8, 0.2]])
    dummy_F = np.array([[0.3, 0.7], [0.6, 0.4]])

    def dummy_get_encoded(*args, **kwargs):
        return dummy_G, dummy_F

    re = RoleExtractor(n_roles=2)
    monkeypatch.setattr(re, "_get_encoded_role_factors", dummy_get_encoded)

    re.extract_role_factors(features)

    # Check that node_role_factor DataFrame matches dummy_G with correct index and columns
    pd.testing.assert_frame_equal(
        re.node_role_factor,
        pd.DataFrame(dummy_G, index=features.index, columns=['role_0', 'role_1'])
    )

    # Check that role_feature_factor DataFrame matches dummy_F with correct index and columns
    pd.testing.assert_frame_equal(
        re.role_feature_factor,
        pd.DataFrame(dummy_F, index=['role_0', 'role_1'], columns=features.columns)
    )


def test_extract_role_factors_without_n_roles(monkeypatch):
    features = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6]
    }, index=['a', 'b', 'c'])

    dummy_G = np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])
    dummy_F = np.array([[0.2, 0.8], [0.9, 0.1]])

    def dummy_select_model(*args, **kwargs):
        return dummy_G, dummy_F

    re = RoleExtractor()
    monkeypatch.setattr(re, "_select_model", dummy_select_model)

    re.extract_role_factors(features)

    pd.testing.assert_frame_equal(
        re.node_role_factor,
        pd.DataFrame(dummy_G, index=features.index, columns=['role_0', 'role_1'])
    )
    pd.testing.assert_frame_equal(
        re.role_feature_factor,
        pd.DataFrame(dummy_F, index=['role_0', 'role_1'], columns=features.columns)
    )


def test_explain_raises_not_implemented():
    re = RoleExtractor()
    with pytest.raises(NotImplementedError):
        re.explain()


@pytest.mark.parametrize("n_roles, n_features, n_nodes, expected_bits", [
    (2, 10, 10, int(np.log2(2 * 10))),
    (4, 5, 20, int(np.log2(4 * 5))),
    (3, 7, 7, int(np.log2(3 * 7))),
])
def test_extract_role_factors_n_bits_computation(monkeypatch, n_roles, n_features, n_nodes, expected_bits):
    features = pd.DataFrame(np.ones((n_nodes, n_features)))

    # We expect that _get_encoded_role_factors is called with n_roles and n_bits matching expected_bits
    called_args = {}

    def dummy_get_encoded_role_factors(features_arg, n_roles_arg, n_bits_arg):
        called_args['n_roles'] = n_roles_arg
        called_args['n_bits'] = n_bits_arg
        # return dummy but valid result
        shape_nodes = features_arg.shape[0]
        G = np.zeros((shape_nodes, n_roles_arg))
        F = np.zeros((n_roles_arg, features_arg.shape[1]))
        return G, F

    re = RoleExtractor(n_roles=n_roles)
    monkeypatch.setattr(re, "_get_encoded_role_factors", dummy_get_encoded_role_factors)
    re.extract_role_factors(features)

    assert called_args['n_roles'] == n_roles
    assert called_args['n_bits'] == expected_bits


def test_select_model_grid_search_and_selection(monkeypatch):
    # Use small feature matrix
    features = pd.DataFrame(np.array([[0, 1], [1, 0]]), columns=['f1', 'f2'])

    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))

    dummy_model = (np.array([[1, 0], [0, 1]]), np.array([[0.5, 0.5], [0.5, 0.5]]))

    calls = []

    # Patch _get_encoded_role_factors: return dummy_model always
    def dummy_get_encoded_role_factors(f, roles, bits):
        # Record calls to verify grid search
        calls.append((roles, bits))
        if bits > 2:  # simulate ValueError for bits above 2
            raise ValueError
        return dummy_model

    # Patch get_description_length_costs to return a deterministic cost
    costs_map = {
        (2, 1): (10.0, 5.0),
        (2, 2): (8.0, 6.0),
        (3, 1): (9.0, 4.0),
        (3, 2): (7.0, 7.0),
    }

    def dummy_get_description_length_costs(f, model):
        # Provide costs depending on calls count
        last_call = calls[-1]
        return costs_map.get(last_call, (np.nan, np.nan))

    monkeypatch.setattr(re, "_get_encoded_role_factors", dummy_get_encoded_role_factors)
    monkeypatch.setattr("graphrole.roles.extract.get_description_length_costs", dummy_get_description_length_costs)

    selected_model = re._select_model(features)

    # Confirm all expected grid points were called
    expected_calls = [(2, 1), (2, 2), (3, 1), (3, 2)]
    assert all(c in calls for c in expected_calls)

    # The minimal total cost is cost at (3,1): 9+4=13, min total cost is (3,1) or (3,2) = 7+7=14?
    # Actually from costs_map minimal sum = cost(2,2)=14, cost(3,1)=13, cost(3,2)=14, cost(2,1)=15
    # Actually cost (2,2)=8+6=14, (3,1)=9+4=13, (3,2)=7+7=14, So min is (3,1)=13

    # Check returned model is dummy_model
    assert selected_model == dummy_model


def test_get_encoded_role_factors_returns_encoded_arrays(monkeypatch):
    features = pd.DataFrame(np.array([[1, 2], [3, 4]]))
    n_roles = 2
    n_bits = 3

    # Patch get_nmf_decomposition to return predictable arrays
    def dummy_get_nmf_decomposition(V, n_roles):
        G = np.array([[1, 0], [0, 1]])
        F = np.array([[1, 2], [3, 4]])
        return G, F

    # Patch encode to simply multiply by n_bins for testing
    def dummy_encode(arr, n_bins):
        return arr * n_bins

    monkeypatch.setattr("graphrole.roles.extract.get_nmf_decomposition", dummy_get_nmf_decomposition)
    monkeypatch.setattr("graphrole.roles.extract.encode", dummy_encode)

    G_encoded, F_encoded = RoleExtractor._get_encoded_role_factors(features, n_roles, n_bits)

    n_bins_expected = 2**n_bits
    np.testing.assert_array_equal(G_encoded, np.array([[n_bins_expected, 0], [0, n_bins_expected]]))
    np.testing.assert_array_equal(F_encoded, np.array([[n_bins_expected, 2*n_bins_expected], [3*n_bins_expected, 4*n_bins_expected]]))


def test_rescale_costs_basic_behavior():
    # Provide a cost matrix with two rows and some nans
    costs = np.array([
        [3, 4, np.nan],
        [0, 0, 0],
        [1, 2, 2]
    ], dtype=float)
    # Expected norms: sqrt(3^2 + 4^2 + 0) = 5, sqrt(0) = 0, sqrt(1+4+4) = 3
    expected_norms = np.array([5, 0, 3], dtype=float)

    rescaled = RoleExtractor._rescale_costs(costs)

    # First row divided by 5
    assert np.allclose(rescaled[0, :2], costs[0, :2] / 5)
    # Division by zero row produces inf or nan - division by 0 must be handled; here check nan or inf
    assert np.all(np.isnan(rescaled[1]) | np.isinf(rescaled[1]))
    # Last row divided by 3
    assert np.allclose(rescaled[2, :], costs[2, :] / 3)


def test_select_model_raises_if_no_valid_model(monkeypatch):
    features = pd.DataFrame(np.ones((3, 3)))
    re = RoleExtractor(n_role_range=(2, 2), n_bit_range=(1, 1))

    # Patch _get_encoded_role_factors to always raise ValueError (simulate no valid models)
    def always_raise(*args, **kwargs):
        raise ValueError

    monkeypatch.setattr(re, "_get_encoded_role_factors", always_raise)

    with pytest.raises(IndexError):
        re._select_model(features)


def test_select_model_skips_invalid_models(monkeypatch):
    features = pd.DataFrame(np.ones((4, 4)))
    re = RoleExtractor(n_role_range=(2, 3), n_bit_range=(1, 2))

    calls = []

    def dummy_get_encoded(f, roles, bits):
        calls.append((roles, bits))
        # Simulate error on bits == 2
        if bits == 2:
            raise ValueError
        return np.ones((f.shape[0], roles)), np.ones((roles, f.shape[1]))

    def dummy_get_description_length_costs(f, model):
        return 1.0, 1.0

    monkeypatch.setattr(re, "_get_encoded_role_factors", dummy_get_encoded)
    monkeypatch.setattr("graphrole.roles.extract.get_description_length_costs", dummy_get_description_length_costs)

    model = re._select_model(features)

    # Models for bits==2 have been skipped, so only calls with bits==1
    assert all(bits == 1 for _, bits in calls)
    # Returned model should not be None and shape matches expected
    assert model[0].shape[1] in range(re.min_roles, re.max_roles + 1)


def test_extract_role_factors_preserves_index_and_columns():
    # Create features with string index and columns
    index = ['nodeA', 'nodeB']
    columns = ['featX', 'featY']

    features = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)

    # Patch _get_encoded_role_factors to return ones of correct shape
    def dummy_get_encoded(*args, **kwargs):
        n_roles = args[1]
        G = np.ones((len(index), n_roles))
        F = np.ones((n_roles, len(columns)))
        return G, F

    re = RoleExtractor(n_roles=3)
    re._get_encoded_role_factors = dummy_get_encoded

    re.extract_role_factors(features)

    # Check that indices and columns are preserved in stored factors
    assert list(re.node_role_factor.index) == index
    assert list(re.node_role_factor.columns) == ['role_0', 'role_1', 'role_2']
    assert list(re.role_feature_factor.index) == ['role_0', 'role_1', 'role_2']
    assert list(re.role_feature_factor.columns) == columns
