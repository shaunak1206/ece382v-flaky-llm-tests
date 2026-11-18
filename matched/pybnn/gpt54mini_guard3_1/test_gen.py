import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as bohamiann_mod
from pybnn.bohamiann import Bohamiann, nll


class DummySampler:
    def __init__(self, params, **kwargs):
        self.params = list(params)
        self.kwargs = kwargs
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_calls += 1


class SimpleBohamiannNet(nn.Module):
    def __init__(self, input_dimensionality: int):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 1, bias=True)
        self.log_var = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        mu = self.linear(x)
        log_var = self.log_var.expand(x.shape[0], 1)
        return torch.cat([mu, log_var], dim=1)


def make_model(monkeypatch, **kwargs):
    monkeypatch.setattr(bohamiann_mod, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(bohamiann_mod, "SGLD", DummySampler)
    monkeypatch.setattr(bohamiann_mod, "SGHMC", DummySampler)
    monkeypatch.setattr(bohamiann_mod, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(bohamiann_mod, "weight_prior", lambda params, dtype=None: torch.tensor(0.0))
    monkeypatch.setattr(bohamiann_mod, "log_variance_prior", lambda x: torch.tensor(0.0))
    return Bohamiann(get_network=lambda input_dimensionality: SimpleBohamiannNet(input_dimensionality),
                     **kwargs)


def test_nll_matches_closed_form():
    input_ = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.0, 1.0], dtype=torch.float64)
    got = nll(input_, target).item()
    expected = 0.5 * np.log(2 * np.pi) + 0.25 * np.log(2.0) + 0.5 * 0.0
    # First sample has zero error and variance 1. Second has error 1 and variance 2.
    expected = (0.5 * np.log(2 * np.pi) + 0.0) + (0.5 * np.log(2 * np.pi) + 0.25 * np.log(2.0) + 0.25)
    expected /= 2.0
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_default_network_output_shape():
    torch.manual_seed(0)
    model = bohamiann_mod.get_default_network(3)
    x = torch.zeros(4, 3)
    out = model(x)
    assert out.shape == (4, 2)


def test_init_sets_configuration():
    bnn = Bohamiann()
    assert bnn.do_normalize_input is True
    assert bnn.do_normalize_output is True
    assert bnn.sampling_method == "adaptive_sghmc"
    assert bnn.use_double_precision is True
    assert bnn.is_trained is False
    assert bnn.sampled_weights == []


def test_network_weights_get_and_set_roundtrip(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    bnn.model = SimpleBohamiannNet(2).float()

    weights = bnn.network_weights
    new_weights = [np.ones_like(w) * 3.5 for w in weights]
    bnn.network_weights = new_weights

    got = bnn.network_weights
    for g in got:
        assert np.allclose(g, 3.5)


def test_train_sets_is_trained_and_samples_weights(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=True, normalize_output=True)
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.linspace(0.0, 1.0, 6).astype(np.float32)

    bnn.train(x, y, num_steps=5, num_burn_in_steps=0, keep_every=1, batch_size=2, verbose=False)

    assert bnn.is_trained is True
    assert len(bnn.sampled_weights) == 4
    assert bnn.sampler.step_calls == 5


def test_train_warns_and_adjusts_batch_size(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    with pytest.warns(None) as record:
        bnn.train(x, y, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=10, verbose=False)

    assert any("Not enough datapoints" in str(w.message) for w in record)
    assert len(bnn.sampled_weights) == 0
    assert bnn.is_trained is True


def test_train_rejects_invalid_batch_size(monkeypatch):
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros(2, dtype=np.float32)

    with pytest.raises(AssertionError, match="Invalid batch size"):
        bnn.train(x, y, batch_size=0, num_steps=1, num_burn_in_steps=0, verbose=False)


def test_train_rejects_invalid_target_shape(monkeypatch):
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((2, 2, 1), dtype=np.float32)

    with pytest.raises(AssertionError, match="Targets need to be in vector format"):
        bnn.train(x, y, batch_size=1, num_steps=1, num_burn_in_steps=0, verbose=False)


def test_continue_training_keeps_previous_samples(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    bnn.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=2, verbose=False)
    first_len = len(bnn.sampled_weights)
    bnn.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2,
              verbose=False, continue_training=True)

    assert len(bnn.sampled_weights) == first_len + 2


def test_train_and_evaluate_returns_curves(monkeypatch):
    np.random.seed(0)
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    x_train = np.arange(12, dtype=np.float32).reshape(6, 2)
    y_train = np.linspace(0.0, 1.0, 6).astype(np.float32)
    x_valid = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    y_valid = np.array([0.2, 0.8], dtype=np.float32)

    n_steps, ll, mse = bnn.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=2, validate_every_n_steps=1, keep_every=1,
        num_burn_in_steps=0, batch_size=2, verbose=False
    )

    assert n_steps == [1, 2]
    assert len(ll) == 2
    assert len(mse) == 2


def test_predict_returns_expected_shapes(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    bnn.model = SimpleBohamiannNet(2).float()
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mu, var = bnn.predict(x)

    assert mu.shape == (2,)
    assert var.shape == (2,)
    assert np.all(var >= 0.0)


def test_predict_return_individual_predictions(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    bnn.model = SimpleBohamiannNet(2).float()
    w = bnn.network_weights
    bnn.sampled_weights = [w, w]

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    mu, var, individual = bnn.predict(x, return_individual_predictions=True)

    assert individual.shape == (2, 1)
    assert mu.shape == (1,)
    assert var.shape == (1,)


def test_predict_single_matches_predict_for_one_sample(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    bnn.model = SimpleBohamiannNet(2).float()
    w = bnn.network_weights
    bnn.sampled_weights = [w]

    x = np.array([1.0, 2.0], dtype=np.float32)
    single = bnn.predict_single(x, sample_index=0)
    mu, var = bnn.predict(x[None, :])

    assert single.shape == (1, 2)
    assert np.allclose(single[0, 0], mu[0])
    assert np.all(var >= 0.0)


def test_f_gradient_for_linear_network(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    net = SimpleBohamiannNet(2).float()
    with torch.no_grad():
        net.linear.weight[:] = torch.tensor([[2.0, -3.0]])
        net.linear.bias[:] = torch.tensor([0.5])
        net.log_var[:] = torch.tensor([0.0])
    bnn.model = net
    bnn.sampled_weights = [bnn.network_weights]

    g = bnn.f_gradient(np.array([1.0, 2.0], dtype=np.float32), weights=bnn.sampled_weights[0])
    assert g.shape == (2,)
    assert np.allclose(g, np.array([2.0, -3.0], dtype=np.float32), atol=1e-6)


def test_predictive_mean_gradient_equals_single_sample_gradient(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    net = SimpleBohamiannNet(2).float()
    with torch.no_grad():
        net.linear.weight[:] = torch.tensor([[1.5, 4.0]])
        net.linear.bias[:] = torch.tensor([0.0])
        net.log_var[:] = torch.tensor([0.0])
    bnn.model = net
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    x = np.array([0.3, -0.7], dtype=np.float32)
    g = bnn.predictive_mean_gradient(x)

    assert np.allclose(g, np.array([1.5, 4.0], dtype=np.float32), atol=1e-6)


def test_predictive_variance_gradient_zero_for_identical_samples(monkeypatch):
    torch.manual_seed(0)
    bnn = make_model(monkeypatch, use_double_precision=False, normalize_input=False, normalize_output=False)
    net = SimpleBohamiannNet(2).float()
    with torch.no_grad():
        net.linear.weight[:] = torch.tensor([[1.0, 0.0]])
        net.linear.bias[:] = torch.tensor([0.0])
        net.log_var[:] = torch.tensor([0.0])
    bnn.model = net
    w = bnn.network_weights
    bnn.sampled_weights = [w, w]

    g = bnn.predictive_variance_gradient(np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(g, np.zeros(2, dtype=np.float32), atol=1e-6)
