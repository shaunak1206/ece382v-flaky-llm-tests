import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.uniform(-1, 1, size=(24, 2))
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1] + 0.01 * rng.randn(24)
    return x, y


@pytest.fixture
def trained_model(data):
    x, y = data
    model = Bohamiann(print_every_n_steps=1)
    # steps 0..11, samples kept at steps 4, 6, 8, 10 -> 4 samples
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=2, batch_size=8)
    return model


def test_nll_matches_closed_form():
    inp = torch.tensor([[0.5, 0.2], [-1.0, -0.3], [2.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0, 1.5], dtype=torch.float64)

    mean = inp[:, 0].numpy()
    log_var = inp[:, 1].numpy()
    t = target.numpy()
    per_sample = 0.5 * (t - mean) ** 2 / np.exp(log_var) + 0.5 * log_var
    expected = per_sample.mean()

    result = nll(inp, target)
    assert float(result) == pytest.approx(expected, rel=1e-8)


def test_nll_is_lower_for_better_prediction():
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    zeros = torch.zeros(3, dtype=torch.float64)
    good = torch.stack([t, zeros], dim=1)
    bad = torch.stack([t + 1.0, zeros], dim=1)

    assert torch.isfinite(nll(good, t))
    assert float(nll(good, t)) < float(nll(bad, t))


def test_nll_accepts_column_or_vector_targets():
    inp = torch.tensor([[0.1, -0.2], [0.7, 0.4], [-0.5, 0.0]], dtype=torch.float64)
    t = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)

    vector_result = nll(inp, t)
    column_result = nll(inp, t.view(-1, 1))
    assert float(vector_result) == pytest.approx(float(column_result))


def test_default_network_output_shape_and_input_dim():
    net = get_default_network(3)
    out = net(torch.randn(7, 3))
    assert out.shape == (7, 2)
    assert net.fc1.in_features == 3


def test_constructor_defaults_and_custom_arguments():
    default = Bohamiann()
    assert default.get_network is get_default_network
    assert default.do_normalize_input is True
    assert default.do_normalize_output is True
    assert default.sampling_method == "adaptive_sghmc"
    assert default.use_double_precision is True
    assert default.likelihood_function is nll
    assert default.print_every_n_steps == 100
    assert default.metrics == (nn.MSELoss,)

    def my_net(input_dimensionality):
        return get_default_network(input_dimensionality)

    def my_nll(input, target):
        return nll(input, target)

    custom = Bohamiann(get_network=my_net, normalize_input=False, normalize_output=False,
                       sampling_method="sgld", use_double_precision=False,
                       metrics=(), likelihood_function=my_nll, print_every_n_steps=7)
    assert custom.get_network is my_net
    assert custom.do_normalize_input is False
    assert custom.do_normalize_output is False
    assert custom.sampling_method == "sgld"
    assert custom.use_double_precision is False
    assert custom.metrics == ()
    assert custom.likelihood_function is my_nll
    assert custom.print_every_n_steps == 7


def test_untrained_model_state():
    model = Bohamiann()
    assert model.is_trained is False
    assert model.sampler is None
    assert model.sampled_weights == []


def test_network_weights_getter_returns_arrays(trained_model):
    weights = trained_model.network_weights
    assert isinstance(weights, tuple)
    assert len(weights) == len(list(trained_model.model.parameters()))
    for w in weights:
        assert isinstance(w, np.ndarray)


def test_network_weights_setter_roundtrip(trained_model):
    zeros = [np.zeros_like(w) for w in trained_model.network_weights]
    trained_model.network_weights = zeros
    for w in trained_model.network_weights:
        assert np.all(w == 0)


def test_train_sets_trained_flag_and_collects_samples(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=2, batch_size=8)

    assert model.is_trained is True
    assert len(model.sampled_weights) == 4
    n_params = len(list(model.model.parameters()))
    for sample in model.sampled_weights:
        assert len(sample) == n_params


def test_train_rejects_zero_batch_size(data):
    x, y = data
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_invalid_target_shape(data):
    x, y = data
    y_two_columns = np.stack([y, y], axis=1)
    model = Bohamiann()
    with pytest.raises(AssertionError):
        model.train(x, y_two_columns, num_steps=2)


def test_train_accepts_column_vector_targets(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y[:, None], num_steps=3, num_burn_in_steps=1, keep_every=1, batch_size=8)
    assert model.is_trained is True
    assert model.y.shape == (x.shape[0],)


def test_train_large_batch_logs_warning(data, caplog):
    x, y = data
    model = Bohamiann()
    with caplog.at_level(logging.WARNING):
        model.train(x[:4], y[:4], num_steps=3, num_burn_in_steps=1, keep_every=1, batch_size=10)
    assert "Not enough datapoints" in caplog.text
    assert model.is_trained is True


def test_train_without_continue_resets_samples(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=2, batch_size=8)
    first_model = model.model
    assert len(model.sampled_weights) == 4

    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=2, batch_size=8)
    assert model.model is not first_model
    assert len(model.sampled_weights) == 4


def test_continue_training_keeps_sampler_and_appends(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=2, batch_size=8)
    sampler_before = model.sampler
    network_before = model.model
    assert len(model.sampled_weights) == 4

    # with burn-in 0 and keep_every 2, steps 2,4,6,8,10 are kept -> 5 more samples
    model.train(x, y, num_steps=12, keep_every=2, num_burn_in_steps=0, batch_size=8,
                continue_training=True)
    assert model.sampler is sampler_before
    assert model.model is network_before
    assert len(model.sampled_weights) == 9


@pytest.mark.parametrize("use_double", [True, False])
def test_double_precision_parameter_dtype(data, use_double):
    x, y = data
    model = Bohamiann(use_double_precision=use_double)
    model.train(x, y, num_steps=2, batch_size=8)
    expected = torch.float64 if use_double else torch.float32
    assert next(model.model.parameters()).dtype == expected


@pytest.mark.parametrize("method, sampler_cls", [
    ("adaptive_sghmc", AdaptiveSGHMC),
    ("sgld", SGLD),
    ("sghmc", SGHMC),
    ("preconditioned_sgld", PreconditionedSGLD),
])
def test_sampling_method_selects_sampler(data, method, sampler_cls):
    x, y = data
    model = Bohamiann(sampling_method=method)
    model.train(x, y, num_steps=3, num_burn_in_steps=1, keep_every=1, batch_size=8)
    assert isinstance(model.sampler, sampler_cls)


def test_get_network_receives_input_dimensionality(data):
    x, y = data
    received = {}

    def factory(input_dimensionality):
        received["d"] = input_dimensionality
        return get_default_network(input_dimensionality)

    model = Bohamiann(get_network=factory)
    model.train(x, y, num_steps=2, batch_size=8)
    assert received["d"] == x.shape[1]


def test_verbose_training_prints_progress(data, capsys):
    x, y = data
    model = Bohamiann(print_every_n_steps=1)
    model.train(x, y, num_steps=4, keep_every=1, num_burn_in_steps=1, batch_size=8, verbose=True)
    out = capsys.readouterr().out
    assert "NLL" in out
    assert "MSE" in out


def test_predict_shapes_and_positive_variance(trained_model):
    x_test = np.random.RandomState(1).uniform(-1, 1, size=(5, 2))
    mu, var = trained_model.predict(x_test)
    assert mu.shape == (5,)
    assert var.shape == (5,)
    assert np.all(var > 0)


def test_predict_individual_predictions(trained_model):
    x_test = np.random.RandomState(2).uniform(-1, 1, size=(5, 2))
    mu, var, individual = trained_model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (len(trained_model.sampled_weights), 5)
    assert np.allclose(individual.mean(axis=0), mu)


def test_predict_is_deterministic(trained_model):
    x_test = np.random.RandomState(3).uniform(-1, 1, size=(5, 2))
    mu1, var1 = trained_model.predict(x_test)
    mu2, var2 = trained_model.predict(x_test)
    assert np.allclose(mu1, mu2)
    assert np.allclose(var1, var2)


def test_predict_single_shape(trained_model):
    x_test = np.random.RandomState(4).uniform(-1, 1, size=(5, 2))
    out = trained_model.predict_single(x_test, sample_index=0)
    assert out.shape[0] == 5


def test_train_and_evaluate_learning_curve(data):
    x, y = data
    x_train, y_train = x[:18], y[:18]
    x_valid, y_valid = x[18:], y[18:]

    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=4, validate_every_n_steps=2, keep_every=1,
        num_burn_in_steps=2, batch_size=8,
    )
    assert n_steps == [4, 6]
    assert len(ll) == 2
    assert len(mse) == 2
    assert np.all(np.isfinite(ll))
    assert np.all(np.asarray(mse) >= 0)


def test_f_gradient_matches_finite_differences(trained_model):
    x0 = trained_model.X[0].astype(np.float64)
    weights = trained_model.sampled_weights[0]

    grad = trained_model.f_gradient(x0, weights=weights)
    assert grad.shape == (x0.shape[0],)

    eps = 1e-6
    fd = np.zeros_like(x0)
    for j in range(x0.shape[0]):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = trained_model.predict_single(x_plus[None, :], sample_index=0)[0, 0]
        f_minus = trained_model.predict_single(x_minus[None, :], sample_index=0)[0, 0]
        fd[j] = (f_plus - f_minus) / (2 * eps)

    assert np.allclose(grad, fd, rtol=1e-3, atol=1e-4)
