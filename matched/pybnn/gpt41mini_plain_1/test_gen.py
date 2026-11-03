import numpy as np
import pytest
import torch
from pybnn.bohamiann import Bohamiann, nll, get_default_network


def test_get_default_network_output_shape():
    net = get_default_network(3)
    x = torch.randn(5, 3)
    out = net(x)
    assert out.shape == (5, 2)  # Output includes mean and log variance (last layer + AppendLayer)
    assert isinstance(out, torch.Tensor)


def test_nll_basic():
    mean = torch.tensor([[0.0], [0.0]])
    log_var = torch.tensor([[0.0], [0.0]])
    output = torch.cat([mean, log_var], dim=1)
    target = torch.tensor([0.0, 1.0])
    loss = nll(output, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is False
    # Loss should be positive
    assert loss.item() >= 0


def test_network_weights_property_and_setter():
    model = get_default_network(2)
    model.double()
    boh = Bohamiann(get_network=lambda **kwargs: model, use_double_precision=True)
    boh.model = model
    weights = boh.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)
    # Modify weights and set back
    new_weights = [w + 0.1 for w in weights]
    boh.network_weights = new_weights
    updated_weights = boh.network_weights
    for nw, ow in zip(updated_weights, new_weights):
        np.testing.assert_allclose(nw, ow, rtol=1e-5)


def test_train_with_minimal_data_and_default_settings():
    np.random.seed(0)
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=5, keep_every=2, num_burn_in_steps=2, batch_size=5, verbose=False)
    assert boh.is_trained
    # Weights should have been sampled at least once after burn-in (num_steps=5, burn-in=2, keep_every=2)
    assert len(boh.sampled_weights) >= 1


def test_train_with_batch_size_larger_than_data_adjusts_batch():
    x_train = np.random.randn(3, 2)
    y_train = np.random.randn(3)
    boh = Bohamiann()
    with pytest.warns(UserWarning):
        boh.train(x_train, y_train, batch_size=5, num_steps=3, num_burn_in_steps=1)
    # batch_size adjusted internally to 3
    # After train, check is_trained flag
    assert boh.is_trained


def test_predict_return_shapes_and_types():
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(4, 2)
    mu, var = boh.predict(x_test)
    assert isinstance(mu, np.ndarray)
    assert isinstance(var, np.ndarray)
    assert mu.shape == (4,)
    assert var.shape == (4,)
    # Check return individual predictions
    mu2, var2, indiv = boh.predict(x_test, return_individual_predictions=True)
    assert isinstance(indiv, np.ndarray)
    assert indiv.shape[1] == 4


def test_predict_single_consistent_with_predict():
    x_train = np.random.randn(15, 3)
    y_train = np.random.randn(15)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=15, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(2, 3)
    sample_index = 0
    preds_single = boh.predict_single(x_test, sample_index)
    mu, var, indiv = boh.predict(x_test, return_individual_predictions=True)
    np.testing.assert_allclose(preds_single[:, 0], indiv[sample_index, :], rtol=1e-5)


def test_f_gradient_shape_and_type():
    x_train = np.random.randn(20, 2)
    y_train = np.random.randn(20)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=10, keep_every=5, num_burn_in_steps=5, batch_size=5, verbose=False)
    x_test = np.random.randn(2)
    weights = boh.sampled_weights[0]
    grad = boh.f_gradient(x_test, weights)
    assert grad.shape == x_test.shape
    assert isinstance(grad, np.ndarray)


def test_predictive_mean_gradient_returns_correct_shape():
    x_train = np.random.randn(30, 4)
    y_train = np.random.randn(30)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=30, keep_every=10, num_burn_in_steps=10, batch_size=10, verbose=False)
    x_test = np.random.randn(4)
    grad = boh.predictive_mean_gradient(x_test)
    assert grad.shape == x_test.shape
    assert isinstance(grad, np.ndarray)


def test_predictive_variance_gradient_returns_correct_shape():
    x_train = np.random.randn(30, 3)
    y_train = np.random.randn(30)
    boh = Bohamiann()
    boh.train(x_train, y_train, num_steps=30, keep_every=10, num_burn_in_steps=10, batch_size=10, verbose=False)
    x_test = np.random.randn(3)
    grad = boh.predictive_variance_gradient(x_test)
    assert grad.shape == x_test.shape
    assert isinstance(grad, np.ndarray)


def test_normalize_input_output_consistency():
    boh = Bohamiann()
    x = np.random.randn(10, 3)
    x_norm, mean, std = boh.normalize_input(x)
    assert np.allclose(np.mean(x_norm, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(x_norm, axis=0), 1, atol=1e-7)

    y = np.random.randn(10)
    y_norm, mean_y, std_y = boh.normalize_output(y)
    assert np.allclose(np.mean(y_norm), 0, atol=1e-7)
    assert np.allclose(np.std(y_norm), 1, atol=1e-7)


def test_train_and_evaluate_runs_and_returns_lists():
    np.random.seed(2)
    x_train = np.random.randn(40, 2)
    y_train = np.random.randn(40)
    x_valid = np.random.randn(10, 2)
    y_valid = np.random.randn(10)
    boh = Bohamiann()
    n_steps, lls, mses = boh.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=200, validate_every_n_steps=100, keep_every=10,
        num_burn_in_steps=50, batch_size=10, verbose=False
    )
    assert isinstance(n_steps, list)
    assert isinstance(lls, list)
    assert isinstance(mses, list)
    assert len(n_steps) == len(lls) == len(mses)
    assert all(isinstance(x, (float, int, np.integer)) for x in n_steps)
    assert all(isinstance(x, float) for x in lls)
    assert all(isinstance(x, float) for x in mses)


def test_train_raises_assert_on_invalid_batch_size():
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    boh = Bohamiann()
    with pytest.raises(AssertionError):
        boh.train(x_train, y_train, batch_size=0)


def test_train_and_evaluate_raises_on_invalid_batch_size():
    x_train = np.random.randn(5, 2)
    y_train = np.random.randn(5)
    x_valid = np.random.randn(2, 2)
    y_valid = np.random.randn(2)
    boh = Bohamiann()
    with pytest.raises(AssertionError):
        boh.train_and_evaluate(x_train, y_train, x_valid, y_valid, batch_size=0)


def test_sampling_methods_initialization():
    x_train = np.random.randn(10, 2)
    y_train = np.random.randn(10)
    for method in ["adaptive_sghmc", "sgld", "preconditioned_sgld", "sghmc"]:
        boh = Bohamiann(sampling_method=method)
        boh.train(x_train, y_train, num_steps=5, num_burn_in_steps=1, batch_size=5)
        assert boh.sampler is not None


def test_predict_with_no_sampled_weights_raises_indexerror():
    boh = Bohamiann()
    x_test = np.random.randn(2, 1)
    # No sampled_weights exist yet
    with pytest.raises(IndexError):
        _ = boh.predict_single(x_test, 0)
