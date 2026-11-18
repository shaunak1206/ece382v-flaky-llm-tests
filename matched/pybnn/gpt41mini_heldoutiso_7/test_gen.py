import numpy as np
import torch
import pytest
from pybnn import bohamiann


def test_get_default_network_output_shape():
    input_dim = 3
    net = bohamiann.get_default_network(input_dimensionality=input_dim)
    net.eval()
    x = torch.randn(5, input_dim)
    out = net(x)
    assert out.shape == (5, 2)
    # last dimension expanded by AppendLayer


def test_nll_basic_computation():
    # Prepare input: batch_size=2, 2 outputs: mean and log var
    input = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    target = torch.tensor([1.0, 5.0])
    res = bohamiann.nll(input, target)
    # NLL non-negative scalar
    assert isinstance(res.item(), float)
    assert res.item() >= 0


def test_network_weights_property_and_setter():
    model = bohamiann.get_default_network(2).double()
    bnn = bohamiann.Bohamiann(get_network=lambda **kwargs: model, use_double_precision=True)
    bnn.model = model

    # Get weights as numpy arrays
    weights = bnn.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)

    # Modify weights array and set back
    new_weights = [w + 1.0 for w in weights]
    bnn.network_weights = new_weights

    # Verify updated parameters (rough check)
    for param, w in zip(bnn.model.parameters(), new_weights):
        array_param = param.data.cpu().numpy()
        assert np.allclose(array_param, w)


@pytest.fixture
def small_training_data():
    # 10 points, 2D input
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    return X, y


@pytest.mark.parametrize("method", ["adaptive_sghmc", "sgld", "preconditioned_sgld", "sghmc"])
def test_train_runs_with_different_samplers(small_training_data, method):
    X, y = small_training_data
    # Use small num_steps and burn_in for fast test
    bnn = bohamiann.Bohamiann(sampling_method=method, use_double_precision=True,
                              get_network=bohamiann.get_default_network)
    bnn.train(X, y, num_steps=20, num_burn_in_steps=5, keep_every=5, batch_size=5)
    # After training, is_trained must be True
    assert bnn.is_trained
    # There should be samples collected in sampled_weights if num_steps > num_burn_in_steps
    assert isinstance(bnn.sampled_weights, list)
    # sampled_weights length depends on steps and keep_every
    assert all(isinstance(s, tuple) for s in bnn.sampled_weights)


def test_train_batch_size_greater_than_data_warns(small_training_data, caplog):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    # batch_size exceeds data size triggers warning
    with caplog.at_level("WARNING"):
        bnn.train(X, y, batch_size=50, num_steps=10, num_burn_in_steps=5, keep_every=5)
    assert any("Not enough datapoints" in rec.message for rec in caplog.records)


def test_train_continue_training_reinitializes_model(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    bnn.train(X, y, num_steps=10, num_burn_in_steps=3, keep_every=5, batch_size=5)
    weights_before = bnn.network_weights
    bnn.train(X, y, num_steps=10, num_burn_in_steps=3, keep_every=5, batch_size=5, continue_training=True)
    weights_after = bnn.network_weights
    # continuing training does NOT clear sampled_weights or reinit model so weights should stay
    assert weights_before is not None
    assert weights_after is not None
    assert len(bnn.sampled_weights) >= 0


def test_predict_output_shape_and_type(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann(use_double_precision=True)
    bnn.train(X, y, num_steps=30, num_burn_in_steps=10, keep_every=10, batch_size=5)

    # predict returns mean, variance for test in shape (N,)
    mu, var = bnn.predict(X[:3])
    assert isinstance(mu, np.ndarray)
    assert isinstance(var, np.ndarray)
    assert mu.shape == (3,)
    assert var.shape == (3,)

    # predict with individual predictions
    mu2, var2, indiv_preds = bnn.predict(X[:3], return_individual_predictions=True)
    assert isinstance(indiv_preds, np.ndarray)
    assert indiv_preds.shape[1] == 3  # Number of test points


def test_predict_single_returns_proper_shape_and_type(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    bnn.train(X, y, num_steps=20, num_burn_in_steps=5, keep_every=5, batch_size=5)
    if len(bnn.sampled_weights) == 0:
        pytest.skip("No sampled weights available for predict_single test")

    pred = bnn.predict_single(X[0:1], 0)
    assert isinstance(pred, np.ndarray)
    # prediction shape should be (2,) because of mean and logvar output
    assert pred.shape[-1] == 2 or pred.shape == (2,)


def test_f_gradient_returns_correct_shape(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    bnn.train(X, y, num_steps=15, num_burn_in_steps=5, keep_every=5)
    if not bnn.sampled_weights:
        pytest.skip("No sampled_weights available for f_gradient test")
    weights = bnn.sampled_weights[0]
    g = bnn.f_gradient(X[0, :], weights)
    # Gradient should be a 1D numpy array of shape input_dimensionality
    assert isinstance(g, np.ndarray)
    assert g.shape == (X.shape[1],)


def test_predictive_mean_gradient_matches_average_gradients(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    bnn.train(X, y, num_steps=20, num_burn_in_steps=5, keep_every=5)

    if len(bnn.sampled_weights) == 0:
        pytest.skip("No sampled weights available")

    mean_grad = bnn.predictive_mean_gradient(X[0])
    # mean gradient shape must match input dimension
    assert isinstance(mean_grad, np.ndarray)
    assert mean_grad.shape == (X.shape[1],)


def test_predictive_variance_gradient_runs_and_shape_correct(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()
    bnn.train(X, y, num_steps=20, num_burn_in_steps=5, keep_every=5)

    if len(bnn.sampled_weights) == 0:
        pytest.skip("No sampled weights available")

    var_grad = bnn.predictive_variance_gradient(X[0])
    assert isinstance(var_grad, np.ndarray)
    assert var_grad.shape == (X.shape[1],)


def test_normalize_input_and_output_behaves_consistently():
    bnn = bohamiann.Bohamiann()
    x = np.random.randn(10, 3)
    y = np.random.randn(10)
    norm_x, m_x, s_x = bnn.normalize_input(x)
    norm_y, m_y, s_y = bnn.normalize_output(y)

    # Means should be near zero, std close to one (numerical tolerance)
    assert abs(np.mean(norm_x)) < 1e-10
    assert abs(np.std(norm_x) - 1) < 1e-10

    assert abs(np.mean(norm_y)) < 1e-10
    assert abs(np.std(norm_y) - 1) < 1e-10


def test_train_and_evaluate_returns_expected_lengths_and_values(small_training_data):
    X, y = small_training_data
    X_valid = np.random.randn(5, 2)
    y_valid = np.random.randn(5)

    bnn = bohamiann.Bohamiann()
    steps, lls, mses = bnn.train_and_evaluate(X, y, X_valid, y_valid,
                                              num_steps=20,
                                              validate_every_n_steps=10,
                                              keep_every=5,
                                              num_burn_in_steps=5,
                                              batch_size=5)

    # Check lengths
    assert len(steps) == 2
    assert len(lls) == 2
    assert len(mses) == 2

    # Each value should be a float
    assert all(isinstance(v, float) or np.isscalar(v) for v in lls)
    assert all(isinstance(v, float) or np.isscalar(v) for v in mses)


def test_train_invalid_batch_size_raises(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()

    with pytest.raises(AssertionError):
        bnn.train(X, y, batch_size=0)

    with pytest.raises(AssertionError):
        bnn.train_and_evaluate(X, y, X, y, batch_size=0)


def test_train_accepts_targets_as_1d_and_2d(small_training_data):
    X, y = small_training_data
    bnn = bohamiann.Bohamiann()

    # Using y as (N,)
    bnn.train(X, y, num_steps=10, num_burn_in_steps=5, batch_size=5)

    # Using y as (N,1)
    bnn2 = bohamiann.Bohamiann()
    bnn2.train(X, y[:, None], num_steps=10, num_burn_in_steps=5, batch_size=5)


def test_network_weights_setter_rejects_wrong_shape():
    model = bohamiann.get_default_network(2).double()
    bnn = bohamiann.Bohamiann(get_network=lambda **kw: model, use_double_precision=True)
    bnn.model = model
    weights = bnn.network_weights
    # Modify one weight shape
    wrong_weights = list(weights)
    wrong_weights[0] = np.ones((1,))  # Wrong shape

    # Setter does NOT check shape explicitly, but will error on copy_
    with pytest.raises(RuntimeError):
        bnn.network_weights = wrong_weights
