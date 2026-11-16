import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from pybnn.bohamiann import Bohamiann, get_default_network, nll
from pybnn.sampler import AdaptiveSGHMC, PreconditionedSGLD, SGHMC, SGLD


@pytest.fixture(autouse=True)
def seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)


def make_data(n=20, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    y = np.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1]
    return x, y


def make_bohamiann_with_model(input_dim=2, **kwargs):
    b = Bohamiann(**kwargs)
    b.model = get_default_network(input_dimensionality=input_dim).double()
    return b


def make_trained_model(n=20, d=2, num_steps=50, keep_every=10, burn_in=10):
    x, y = make_data(n=n, d=d)
    b = Bohamiann()
    b.train(x, y, num_steps=num_steps, keep_every=keep_every,
            num_burn_in_steps=burn_in, batch_size=5)
    return b


def test_init_default_attributes():
    b = Bohamiann()
    assert b.get_network is get_default_network
    assert b.likelihood_function is nll
    assert b.do_normalize_input is True
    assert b.do_normalize_output is True
    assert b.sampling_method == "adaptive_sghmc"
    assert b.use_double_precision is True
    assert b.metrics == (nn.MSELoss,)
    assert b.print_every_n_steps == 100
    assert b.sampled_weights == []
    assert b.is_trained is False
    assert b.sampler is None


def test_init_stores_custom_arguments():
    def net(input_dimensionality):
        return get_default_network(input_dimensionality)

    def lik(input, target):
        return torch.mean(input[:, 0] ** 2)

    b = Bohamiann(get_network=net, normalize_input=False, normalize_output=False,
                  sampling_method="sgld", use_double_precision=False,
                  metrics=(nn.L1Loss,), likelihood_function=lik, print_every_n_steps=7)
    assert b.get_network is net
    assert b.likelihood_function is lik
    assert b.do_normalize_input is False
    assert b.do_normalize_output is False
    assert b.sampling_method == "sgld"
    assert b.use_double_precision is False
    assert b.metrics == (nn.L1Loss,)
    assert b.print_every_n_steps == 7


def test_nll_matches_manual_gaussian_formula():
    mean = np.array([0.5, -1.0, 2.0])
    log_var = np.array([0.0, np.log(4.0), -1.0])
    target = np.array([1.0, -0.5, 1.5])
    inp = torch.from_numpy(np.stack([mean, log_var], axis=1))
    expected = np.mean(0.5 * (target - mean) ** 2 / np.exp(log_var) + 0.5 * log_var)
    result = nll(inp, torch.from_numpy(target))
    assert float(result) == pytest.approx(expected, rel=1e-9)


def test_nll_is_lower_when_mean_is_closer_and_accepts_column_targets():
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    log_var = torch.zeros(3, dtype=torch.float64)
    good = torch.stack([target, log_var], dim=1)
    bad = torch.stack([target + 1.0, log_var], dim=1)
    assert float(nll(good, target)) < float(nll(bad, target))
    assert float(nll(good, target.view(-1, 1))) == pytest.approx(float(nll(good, target)))


def test_default_network_maps_inputs_to_mean_and_log_variance():
    net = get_default_network(input_dimensionality=3)
    assert isinstance(net, nn.Module)
    out = net(torch.zeros(7, 3, dtype=torch.float32))
    assert tuple(out.shape) == (7, 2)


def test_network_weights_getter_returns_one_array_per_parameter():
    b = make_bohamiann_with_model()
    weights = b.network_weights
    params = list(b.model.parameters())
    assert isinstance(weights, tuple)
    assert len(weights) == len(params)
    for w, p in zip(weights, params):
        assert isinstance(w, np.ndarray)
        assert w.shape == tuple(p.shape)
        assert w.dtype == np.float64


def test_network_weights_getter_returns_detached_copies():
    b = make_bohamiann_with_model()
    before = [w.copy() for w in b.network_weights]
    returned = b.network_weights
    for w in returned:
        w[...] = 123.0
    after = b.network_weights
    for a, c in zip(after, before):
        np.testing.assert_allclose(a, c)


def test_network_weights_setter_roundtrip():
    b = make_bohamiann_with_model()
    new = [np.full(tuple(p.shape), 0.25) for p in b.model.parameters()]
    b.network_weights = new
    for w, n in zip(b.network_weights, new):
        np.testing.assert_allclose(w, n)


def test_normalize_input_centers_and_scales_data():
    b = Bohamiann()
    x = np.random.RandomState(1).normal(5.0, 3.0, size=(200, 2))
    xn, m, s = b.normalize_input(x)
    np.testing.assert_allclose(xn.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(xn.std(axis=0), 1.0, rtol=1e-2)
    np.testing.assert_allclose(np.asarray(m), x.mean(axis=0), rtol=1e-10)


def test_train_populates_samples_and_marks_model_trained():
    x, y = make_data(n=20)
    b = Bohamiann()
    b.train(x, y, num_steps=50, keep_every=10, num_burn_in_steps=10, batch_size=5)
    assert b.is_trained is True
    # steps 20, 30, 40 are kept
    assert len(b.sampled_weights) == 3
    assert len(b.sampled_weights[0]) == len(list(b.model.parameters()))
    assert np.shape(b.x_mean) == (2,)


@pytest.mark.parametrize("method, sampler_cls", [
    ("adaptive_sghmc", AdaptiveSGHMC),
    ("sghmc", SGHMC),
    ("sgld", SGLD),
    ("preconditioned_sgld", PreconditionedSGLD),
])
def test_train_creates_requested_sampler(method, sampler_cls):
    x, y = make_data(n=20)
    b = Bohamiann(sampling_method=method)
    b.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=5, batch_size=5)
    assert isinstance(b.sampler, sampler_cls)
    # steps 10 and 15 are kept
    assert len(b.sampled_weights) == 2


def test_train_builds_network_with_input_dimensionality_keyword():
    calls = []

    def get_network(input_dimensionality):
        calls.append(input_dimensionality)
        return get_default_network(input_dimensionality)

    x, y = make_data(n=12, d=3)
    b = Bohamiann(get_network=get_network)
    b.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=4)
    assert calls == [3]


def test_train_uses_custom_likelihood_function():
    calls = []

    def likelihood(input, target):
        calls.append(tuple(target.shape))
        return torch.mean(input[:, 0] ** 2)

    x, y = make_data(n=20)
    b = Bohamiann(likelihood_function=likelihood)
    b.train(x, y, num_steps=6, keep_every=100, num_burn_in_steps=100, batch_size=5)
    assert len(calls) == 6


def test_train_rejects_non_positive_batch_size():
    x, y = make_data(n=10)
    b = Bohamiann()
    with pytest.raises(AssertionError):
        b.train(x, y, num_steps=2, batch_size=0)


def test_train_rejects_multi_output_targets():
    x, _ = make_data(n=10)
    y = np.zeros((10, 2))
    b = Bohamiann()
    with pytest.raises(AssertionError):
        b.train(x, y, num_steps=2, batch_size=5)


def test_train_warns_and_uses_all_points_when_batch_larger_than_data(caplog):
    x, y = make_data(n=3)
    b = Bohamiann()
    with caplog.at_level(logging.WARNING):
        b.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=10)
    assert "Not enough datapoints" in caplog.text
    assert b.is_trained is True


def test_train_continue_accumulates_samples_and_restart_clears_them():
    x, y = make_data(n=20)
    b = Bohamiann()
    b.train(x, y, num_steps=50, keep_every=10, num_burn_in_steps=10, batch_size=5)
    assert len(b.sampled_weights) == 3
    sampler = b.sampler

    b.train(x, y, num_steps=30, keep_every=10, num_burn_in_steps=0,
            batch_size=5, continue_training=True)
    assert len(b.sampled_weights) == 5
    assert b.sampler is sampler

    b.train(x, y, num_steps=50, keep_every=10, num_burn_in_steps=10, batch_size=5)
    assert len(b.sampled_weights) == 3


def test_train_verbose_prints_progress(capsys):
    x, y = make_data(n=20)
    b = Bohamiann(print_every_n_steps=5)
    b.train(x, y, num_steps=20, keep_every=5, num_burn_in_steps=10,
            batch_size=5, verbose=True)
    out = capsys.readouterr().out
    assert "NLL" in out
    assert "MSE" in out


def test_predict_returns_mean_variance_and_individual_predictions():
    b = make_trained_model()
    x_test = make_data(n=7, seed=3)[0]

    mu, var = b.predict(x_test)
    assert mu.shape == (7,)
    assert var.shape == (7,)
    assert np.all(np.isfinite(mu))
    assert np.all(var > 0)

    mu2, var2, individual = b.predict(x_test, return_individual_predictions=True)
    assert individual.shape == (len(b.sampled_weights), 7)
    np.testing.assert_allclose(mu2, mu)
    np.testing.assert_allclose(var2, var)


def test_predict_single_matches_corresponding_individual_prediction():
    b = make_trained_model()
    x_test = make_data(n=5, seed=4)[0]
    _, _, individual = b.predict(x_test, return_individual_predictions=True)
    for i in range(len(b.sampled_weights)):
        single = b.predict_single(x_test, sample_index=i)
        np.testing.assert_allclose(single[:, 0], individual[i], rtol=1e-6)


def test_train_and_evaluate_returns_learning_curve():
    x, y = make_data(n=20)
    x_valid, y_valid = make_data(n=8, seed=5)
    b = Bohamiann()
    n_steps, ll, mse = b.train_and_evaluate(
        x, y, x_valid, y_valid,
        num_steps=40, validate_every_n_steps=10, keep_every=5,
        num_burn_in_steps=10, lr=1e-2, batch_size=5)
    assert n_steps == [20, 30, 40, 50]
    assert len(ll) == 4
    assert len(mse) == 4
    assert np.all(np.isfinite(ll))
    assert np.all(np.asarray(mse) >= 0)


def make_gradient_model():
    b = Bohamiann()
    b.model = get_default_network(input_dimensionality=2).double()
    b.do_normalize_input = True
    b.do_normalize_output = True
    b.use_double_precision = True
    b.x_mean = np.array([0.3, -0.1])
    b.x_std = np.array([2.0, 0.5])
    b.y_mean = 1.5
    b.y_std = 3.0
    return b


def test_f_gradient_matches_finite_differences():
    b = make_gradient_model()
    weights = b.network_weights
    b.sampled_weights = [weights]
    x0 = np.array([0.2, 0.4])

    g = b.f_gradient(x0, weights)

    eps = 1e-6
    fd = np.zeros(2)
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        f_plus = b.predict_single((x0 + e)[None, :], 0)[0, 0]
        f_minus = b.predict_single((x0 - e)[None, :], 0)[0, 0]
        fd[j] = (f_plus - f_minus) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)


def test_predictive_mean_gradient_is_average_of_sample_gradients():
    b = make_gradient_model()
    w1 = b.network_weights
    w2 = [w * 1.5 + 0.05 for w in w1]
    b.sampled_weights = [w1, w2]
    x0 = np.array([0.2, 0.4])

    expected = (b.f_gradient(x0, weights=w1) + b.f_gradient(x0, weights=w2)) / 2.0
    result = b.predictive_mean_gradient(x0)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_predictive_variance_gradient_is_zero_with_single_sample():
    b = make_gradient_model()
    b.sampled_weights = [b.network_weights]
    x0 = np.array([0.2, 0.4])
    g = b.predictive_variance_gradient(x0)
    assert g.shape == (2,)
    np.testing.assert_allclose(g, 0.0, atol=1e-12)
