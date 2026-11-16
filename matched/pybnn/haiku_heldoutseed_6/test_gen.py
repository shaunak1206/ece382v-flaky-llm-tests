import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


def make_data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1, 1, size=(n, d))
    y = np.sin(3 * x[:, 0]) + 0.1 * x[:, 1] + 0.01 * rng.randn(n)
    return x, y


@pytest.fixture
def data():
    return make_data()


@pytest.fixture
def trained_model(data):
    x, y = data
    model = Bohamiann()
    # burn-in 2, keep every 2 steps over steps 3..9 -> samples at steps 4, 6, 8
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    return model


def test_init_stores_configuration():
    model = Bohamiann()
    assert model.get_network is get_default_network
    assert model.do_normalize_input is True
    assert model.do_normalize_output is True
    assert model.sampling_method == "adaptive_sghmc"
    assert model.use_double_precision is True
    assert model.metrics == (nn.MSELoss,)
    assert model.likelihood_function is nll
    assert model.print_every_n_steps == 100
    assert model.is_trained is False
    assert model.sampler is None
    assert model.sampled_weights == []

    custom = Bohamiann(normalize_input=False, normalize_output=False, sampling_method="sgld",
                       use_double_precision=False, print_every_n_steps=7)
    assert custom.do_normalize_input is False
    assert custom.do_normalize_output is False
    assert custom.sampling_method == "sgld"
    assert custom.use_double_precision is False
    assert custom.print_every_n_steps == 7


def test_get_default_network_output_shape_and_first_layer():
    net = get_default_network(input_dimensionality=3).double()
    out = net(torch.zeros(5, 3, dtype=torch.float64))
    assert out.shape == (5, 2)
    assert net.fc1.weight.shape == (50, 3)


def test_nll_known_value():
    # mean 0, log variance 0 (variance 1), target 1 -> per-sample NLL is 0.5
    pred = torch.zeros(4, 2)
    target = torch.ones(4)
    value = nll(input=pred, target=target)
    assert value.dim() == 0
    assert float(value) == pytest.approx(0.5, abs=1e-6)


def test_nll_decreases_when_mean_improves():
    target = torch.tensor([1.0])
    close = nll(input=torch.tensor([[1.0, 0.0]]), target=target)
    far = nll(input=torch.tensor([[2.0, 0.0]]), target=target)
    assert float(close) < float(far)


def test_network_weights_getter_returns_numpy_arrays():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=3).double()
    expected_shapes = [p.shape for p in model.model.parameters()]

    weights = model.network_weights

    assert isinstance(weights, tuple)
    assert [w.shape for w in weights] == expected_shapes
    assert all(isinstance(w, np.ndarray) for w in weights)


def test_network_weights_setter_roundtrip():
    model = Bohamiann()
    model.model = get_default_network(input_dimensionality=3).double()
    new_weights = [np.full(p.shape, 0.5) for p in model.model.parameters()]

    model.network_weights = new_weights

    for got, expected in zip(model.network_weights, new_weights):
        np.testing.assert_allclose(got, expected)


def test_train_populates_samples_and_flags(data, trained_model):
    x, _ = data
    assert trained_model.is_trained is True
    assert len(trained_model.sampled_weights) == 3
    n_params = len(list(trained_model.model.parameters()))
    for sample in trained_model.sampled_weights:
        assert len(sample) == n_params
    assert trained_model.X.shape == x.shape


def test_train_without_continue_clears_previous_samples(data):
    x, y = data
    model = Bohamiann()
    kwargs = dict(num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    model.train(x, y, **kwargs)
    model.train(x, y, continue_training=False, **kwargs)
    assert len(model.sampled_weights) == 3


def test_train_continue_training_keeps_samples(data):
    x, y = data
    model = Bohamiann()
    kwargs = dict(num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    model.train(x, y, **kwargs)
    model.train(x, y, continue_training=True, **kwargs)
    assert len(model.sampled_weights) == 6


def test_train_rejects_zero_batch_size(data):
    x, y = data
    with pytest.raises(AssertionError):
        Bohamiann().train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_non_vector_targets(data):
    x, y = data
    bad_targets = np.stack([y, y], axis=1)  # shape (N, 2)
    with pytest.raises(AssertionError):
        Bohamiann().train(x, bad_targets, num_steps=2, batch_size=5)


def test_train_accepts_column_vector_targets(data):
    x, y = data
    model = Bohamiann()
    model.train(x, y.reshape(-1, 1), num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=5)
    assert model.is_trained is True
    assert model.y.ndim == 1


def test_train_small_dataset_warns_and_uses_all_points(caplog):
    x, y = make_data(n=3)
    model = Bohamiann()
    with caplog.at_level(logging.WARNING):
        model.train(x, y, num_steps=4, keep_every=1, num_burn_in_steps=1, batch_size=20)
    assert "Not enough datapoints" in caplog.text
    assert model.is_trained is True


@pytest.mark.parametrize("method, sampler_cls", [
    ("adaptive_sghmc", AdaptiveSGHMC),
    ("sgld", SGLD),
    ("sghmc", SGHMC),
    ("preconditioned_sgld", PreconditionedSGLD),
])
def test_train_selects_sampler(data, method, sampler_cls):
    x, y = data
    model = Bohamiann(sampling_method=method)
    model.train(x, y, num_steps=2, keep_every=1, num_burn_in_steps=0, batch_size=5)
    assert isinstance(model.sampler, sampler_cls)


def test_train_uses_custom_likelihood_and_network_factory(data):
    x, y = data
    likelihood_calls = []
    factory_calls = []

    def likelihood(input, target):
        likelihood_calls.append(1)
        return nll(input=input, target=target)

    def factory(input_dimensionality):
        factory_calls.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    model = Bohamiann(get_network=factory, likelihood_function=likelihood)
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)

    assert len(likelihood_calls) == 10
    assert factory_calls == [x.shape[1]]


def test_train_verbose_prints_progress(data, capsys):
    x, y = data
    model = Bohamiann(print_every_n_steps=1)
    model.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=5, verbose=True)
    out = capsys.readouterr().out
    assert "Step" in out
    assert "NLL" in out


def test_predict_shapes_and_positive_variance(trained_model):
    x_test = np.random.uniform(-1, 1, size=(7, 2))
    mean, var = trained_model.predict(x_test)
    assert mean.shape == (7,)
    assert var.shape == (7,)
    assert np.all(var > 0)


def test_predict_individual_predictions_shape_and_consistency(trained_model):
    x_test = np.random.uniform(-1, 1, size=(7, 2))
    mean, _, individual = trained_model.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (3, 7)
    np.testing.assert_allclose(mean, individual.mean(axis=0), rtol=1e-6, atol=1e-8)


def test_predict_single_matches_individual_prediction(trained_model):
    x_test = np.random.uniform(-1, 1, size=(4, 2))
    single = trained_model.predict_single(x_test, sample_index=1)
    _, _, individual = trained_model.predict(x_test, return_individual_predictions=True)
    assert single.shape == (4, 2)
    np.testing.assert_allclose(single[:, 0], individual[1], rtol=1e-6, atol=1e-8)


def test_predict_without_normalization_works(data):
    x, y = data
    model = Bohamiann(normalize_input=False, normalize_output=False)
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    mean, var = model.predict(x[:6])
    assert mean.shape == (6,)
    assert var.shape == (6,)
    assert np.all(np.isfinite(mean))


def test_float32_mode_trains_and_predicts(data):
    x, y = data
    model = Bohamiann(use_double_precision=False)
    model.train(x, y, num_steps=10, keep_every=2, num_burn_in_steps=2, batch_size=5)
    assert all(w.dtype == np.float32 for w in model.sampled_weights[0])
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert np.all(np.isfinite(var))


def test_normalize_input_zero_mean_unit_var():
    model = Bohamiann()
    x = np.random.RandomState(1).normal(loc=5.0, scale=3.0, size=(200, 2))
    x_norm, x_mean, x_std = model.normalize_input(x)
    np.testing.assert_allclose(x_norm.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(x_norm.std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(x_mean, x.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(x_std, x.std(axis=0), rtol=1e-6)


def test_f_gradient_matches_finite_difference(trained_model):
    x = np.array([0.3, -0.2])
    weights = trained_model.sampled_weights[0]
    grad = trained_model.f_gradient(x, weights)

    eps = 1e-6
    fd = np.zeros_like(x)
    for j in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = trained_model.predict_single(x_plus[None, :], sample_index=0)[0, 0]
        f_minus = trained_model.predict_single(x_minus[None, :], sample_index=0)[0, 0]
        fd[j] = (f_plus - f_minus) / (2 * eps)

    assert grad.shape == (2,)
    np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-5)


def test_predictive_gradients_shapes(trained_model):
    x = np.array([0.1, 0.4])
    mean_grad = trained_model.predictive_mean_gradient(x)
    var_grad = trained_model.predictive_variance_gradient(x)
    assert mean_grad.shape == (2,)
    assert var_grad.shape == (2,)
    assert np.all(np.isfinite(mean_grad))
    assert np.all(np.isfinite(var_grad))


def test_train_and_evaluate_returns_learning_curve(data):
    x, y = data
    x_valid, y_valid = make_data(n=6, seed=1)[0], make_data(n=6, seed=1)[1]
    model = Bohamiann()
    n_steps, ll, mse = model.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=6, validate_every_n_steps=2, keep_every=1,
        num_burn_in_steps=2, batch_size=5,
    )
    assert n_steps == [4, 6, 8]
    assert len(ll) == 3
    assert len(mse) == 3
    assert all(np.isfinite(v) for v in ll)
    assert all(v >= 0 for v in mse)
