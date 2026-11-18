import numpy as np
import pandas as pd
import pytest

from graphrole.roles.extract import RoleExtractor


def test_init_defaults():
    extractor = RoleExtractor()
    assert extractor.n_roles is None
    assert extractor.min_roles == 2
    assert extractor.max_roles == 8
    assert extractor.min_bits == 1
    assert extractor.max_bits == 8
    assert extractor.node_role_factor is None
    assert extractor.role_feature_factor is None


def test_init_custom_ranges():
    extractor = RoleExtractor(n_roles=3, n_role_range=(1, 5), n_bit_range=(2, 4))
    assert extractor.n_roles == 3
    assert extractor.min_roles == 1
    assert extractor.max_roles == 5
    assert extractor.min_bits == 2
    assert extractor.max_bits == 4


def test_roles_property_no_factor():
    extractor = RoleExtractor()
    assert extractor.roles is None


def test_roles_property_returns_dict():
    df = pd.DataFrame({
        'role_0': [0.1, 0.4],
        'role_1': [0.9, 0.6]
    }, index=['node1', 'node2'])
    extractor = RoleExtractor()
    extractor.node_role_factor = df
    roles = extractor.roles
    assert roles == {'node1': 'role_1', 'node2': 'role_1'}


def test_role_percentage_property_no_factor():
    extractor = RoleExtractor()
    assert extractor.role_percentage is None


def test_role_percentage_property_returns_normalized_df():
    df = pd.DataFrame({
        'role_0': [2, 1],
        'role_1': [2, 3]
    }, index=['node1', 'node2'])
    extractor = RoleExtractor()
    extractor.node_role_factor = df
    percentages = extractor.role_percentage
    expected = pd.DataFrame({
        'role_0': [0.5, 0.25],
        'role_1': [0.5, 0.75]
    }, index=['node1', 'node2'])
    pd.testing.assert_frame_equal(percentages, expected)


def test_extract_role_factors_with_n_roles(monkeypatch):
    # monkeypatch _get_encoded_role_factors to return predictable arrays
    monkeypatch.setattr(
        RoleExtractor,
        '_get_encoded_role_factors',
        lambda self, features, n_roles, n_bits: (
            np.ones((features.shape[0], n_roles)),
            np.ones((n_roles, features.shape[1])),
        )
    )
    # create features with 3 nodes and 4 features
    features = pd.DataFrame(np.random.rand(3, 4), index=['a', 'b', 'c'], columns=['f1', 'f2', 'f3', 'f4'])
    extractor = RoleExtractor(n_roles=3)
    extractor.extract_role_factors(features)
    assert extractor.node_role_factor.shape == (3, 3)
    assert extractor.role_feature_factor.shape == (3, 4)
    # Check index preservation
    assert list(extractor.node_role_factor.index) == ['a', 'b', 'c']
    assert list(extractor.role_feature_factor.columns) == ['f1', 'f2', 'f3', 'f4']
    assert list(extractor.role_feature_factor.index) == ['role_0', 'role_1', 'role_2']


def test_extract_role_factors_without_n_roles(monkeypatch):
    features = pd.DataFrame(np.random.rand(2, 2), index=['x', 'y'], columns=['A', 'B'])
    # patch _select_model to return fixed arrays
    def mock_select_model(self, features_arg):
        return (np.array([[1, 0], [0, 1]]), np.array([[0.5, 0.5], [0.5, 0.5]]))
    monkeypatch.setattr(RoleExtractor, '_select_model', mock_select_model)

    extractor = RoleExtractor()
    extractor.extract_role_factors(features)
    assert extractor.node_role_factor.shape == (2, 2)
    assert extractor.role_feature_factor.shape == (2, 2)
    assert list(extractor.node_role_factor.index) == ['x', 'y']
    assert list(extractor.role_feature_factor.columns) == ['A', 'B']
    assert list(extractor.role_feature_factor.index) == ['role_0', 'role_1']


def test_explain_raises_not_implemented():
    extractor = RoleExtractor()
    with pytest.raises(NotImplementedError):
        extractor.explain()


def test_select_model_grid_search(monkeypatch):
    features = pd.DataFrame(np.ones((4, 3)))

    # Setup dummy _get_encoded_role_factors which fails for bits > 3
    def dummy_get_encoded_role_factors(self, features_arg, roles_arg, bits_arg):
        if bits_arg > 3:
            raise ValueError("Bits too large")
        shape = (features_arg.shape[0], roles_arg)
        return (np.ones(shape), np.ones((roles_arg, features_arg.shape[1])))

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', dummy_get_encoded_role_factors)

    # patch get_description_length_costs to return (encoding_cost, error_cost) = (roles*bits, 10 - bits)
    def dummy_costs(features_arg, model):
        node_factor, feature_factor = model
        n_roles = node_factor.shape[1]
        # simple cost function to guarantee minimum at roles=2, bits=3 for example
        encoding_cost = n_roles * 2
        error_cost = 10 - 1
        return encoding_cost, error_cost

    monkeypatch.setattr('graphrole.roles.extract.get_description_length_costs', dummy_costs)

    extractor = RoleExtractor(
        n_role_range=(2, 3),
        n_bit_range=(1, 3),
    )
    model = extractor._select_model(features)

    # model returned should have first dimension = nodes, second = roles
    node_factor, feature_factor = model
    assert node_factor.shape[0] == features.shape[0]
    assert feature_factor.shape[1] == features.shape[1]
    assert node_factor.shape[1] == feature_factor.shape[0]
    # roles should respect the range min_roles <= roles <= max_roles
    assert 2 <= node_factor.shape[1] <= 3


def test_get_encoded_role_factors_shapes(monkeypatch):
    # Patch get_nmf_decomposition and encode to return predictable arrays
    def dummy_get_nmf_decomposition(V, n_roles):
        # Return arrays of shape (n_samples, n_roles) and (n_roles, n_features)
        n_samples, n_features = V.shape
        G = np.ones((n_samples, n_roles))
        F = np.ones((n_roles, n_features))
        return G, F

    def dummy_encode(array, n_bins):
        # Simulate encoding by returning array multiplied by n_bins
        return array * n_bins

    monkeypatch.setattr('graphrole.roles.extract.get_nmf_decomposition', dummy_get_nmf_decomposition)
    monkeypatch.setattr('graphrole.roles.extract.encode', dummy_encode)

    features = pd.DataFrame(np.random.rand(5, 6))
    n_roles = 4
    n_bits = 3
    G_encoded, F_encoded = RoleExtractor._get_encoded_role_factors(features, n_roles, n_bits)
    assert G_encoded.shape == (5, n_roles)
    assert F_encoded.shape == (n_roles, 6)
    expected_value = 2**n_bits
    assert np.allclose(G_encoded, expected_value)
    assert np.allclose(F_encoded, expected_value)


def test_rescale_costs_basic():
    costs = np.array([
        [3.0, 4.0],
        [6.0, 8.0],
    ])
    scaled = RoleExtractor._rescale_costs(costs)
    # norms:
    # row 0: sqrt(9+16)=5.0
    # row 1: sqrt(36+64)=10.0
    expected = np.array([
        [3.0/5.0, 4.0/5.0],
        [6.0/10.0, 8.0/10.0],
    ])
    np.testing.assert_allclose(scaled, expected)


def test_rescale_costs_with_nan():
    costs = np.array([
        [3.0, np.nan],
        [np.nan, 8.0],
    ])
    scaled = RoleExtractor._rescale_costs(costs)
    # norms:
    # row 0: sqrt(9) = 3.0
    # row 1: sqrt(64) = 8.0
    expected = np.array([
        [3.0/3.0, np.nan],
        [np.nan, 8.0/8.0],
    ])
    np.testing.assert_allclose(scaled, expected, equal_nan=True)


def test_select_model_skips_invalid_models(monkeypatch):
    # Setup features
    features = pd.DataFrame(np.ones((3, 3)))

    call_args = []

    def dummy_get_encoded_role_factors(self, features_arg, roles_arg, bits_arg):
        call_args.append((roles_arg, bits_arg))
        if bits_arg == 2:
            raise ValueError("Invalid bits")
        shape = (features_arg.shape[0], roles_arg)
        return (np.ones(shape), np.ones((roles_arg, features_arg.shape[1])))

    def dummy_costs(features_arg, model):
        return 1.0, 1.0

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', dummy_get_encoded_role_factors)
    monkeypatch.setattr('graphrole.roles.extract.get_description_length_costs', dummy_costs)

    extractor = RoleExtractor(
        n_role_range=(2, 3),
        n_bit_range=(1, 3),
    )
    model = extractor._select_model(features)
    # Ensure that bits=2 calls raised ValueError and model was computed for others only
    assert all(bits != 2 for _, bits in call_args if _ in range(2, 4))
    assert model is not None


def test_extract_role_factors_sets_factors(monkeypatch):
    features = pd.DataFrame(np.ones((3, 2)), index=list('abc'), columns=['f1', 'f2'])

    def dummy_get_encoded_role_factors(self, features_arg, roles_arg, bits_arg):
        G = np.array([[0.1, 0.9]] * features_arg.shape[0])
        F = np.array([[0.9, 0.1], [0.5, 0.5]])
        return G, F

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', dummy_get_encoded_role_factors)
    extractor = RoleExtractor(n_roles=2)
    extractor.extract_role_factors(features)

    # Check assigned DataFrames
    assert isinstance(extractor.node_role_factor, pd.DataFrame)
    assert isinstance(extractor.role_feature_factor, pd.DataFrame)
    assert list(extractor.node_role_factor.columns) == ['role_0', 'role_1']
    assert list(extractor.role_feature_factor.index) == ['role_0', 'role_1']


def test_roles_returns_correct_roles_after_extract(monkeypatch):
    features = pd.DataFrame(np.array([[0.2, 0.8], [0.7, 0.3]]), index=['n1', 'n2'], columns=['f1', 'f2'])

    def dummy_get_encoded_role_factors(self, features_arg, roles_arg, bits_arg):
        # Here role_0 is stronger for n1, role_1 stronger for n2
        G = np.array([[0.6, 0.4], [0.3, 0.7]])
        F = np.array([[0.5, 0.5], [0.5, 0.5]])
        return G, F

    monkeypatch.setattr(RoleExtractor, '_get_encoded_role_factors', dummy_get_encoded_role_factors)
    extractor = RoleExtractor(n_roles=2)
    extractor.extract_role_factors(features)
    roles = extractor.roles
    assert roles['n1'] == 'role_0'
    assert roles['n2'] == 'role_1'
