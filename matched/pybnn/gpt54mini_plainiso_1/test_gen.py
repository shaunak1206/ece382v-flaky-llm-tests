import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


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


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.linear(x)


def tiny_network(input_dimensionality):
    net = TinyNet(input_dimensionality)
    with torch.no_grad():
        net.linear.weight.zero_()
        net.linear.bias.zero_()
    return net


@pytest.fixture(autouse=True)
def patch_bohamiann(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: loader)


def test_nll_matches_manual_computation():
    inp = torch.tensor([[0.0, 0.0], [1.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([0.0, 2.0], dtype=torch.float64)
    out = boh.nll(inp, target)

    pred_mean = np.array([0.0, 1.0])
    pred_log_var = np.array([0.0, np.log(2.0)])
    pred_var_inv = 1.0 / (np.exp(pred_log_var) + 1e-16)
    mse = (np.array([0.0, 2.0]) - pred_mean) ** 2
    ll = np.sum(-mse * (0.5 * pred_var_inv) - 0.5 * pred_log_var) / 2.0
    expected = -ll
    assert np.isclose(out.item(), expected)


def test_default_network_output_shape():
    net = boh.get_default_network(3)
    x = torch.zeros(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_property_roundtrip():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False)
    model.model = tiny_network(2).double()
    weights = model.network_weights
    assert len(weights) == 2  # weight and bias
    new_weights = [np.ones_like(w) * 3.5 for w in weights]
    model.network_weights = new_weights
    roundtrip = model.network_weights
    assert np.allclose(roundtrip[0], 3.5)
    assert np.allclose(roundtrip[1], 3.5)


def test_train_rejects_invalid_batch_size():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False)
    x = np.zeros((2, 1), dtype=np.float64)
    y = np.zeros((2,), dtype=np.float64)
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=0)


def test_train_rejects_invalid_target_shape():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False)
    x = np.zeros((2, 1), dtype=np.float64)
    y = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=1)


def test_train_initializes_model_sampler_and_marks_trained():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    model.train(x, y, num_steps=2, keep_every=1, num_burn_in_steps=0, batch_size=2)
    assert model.is_trained is True
    assert isinstance(model.sampler, DummySampler)
    assert model.model is not None


def test_train_collects_sampled_weights_after_burn_in():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    model.train(x, y, num_steps=5, keep_every=1, num_burn_in_steps=1, batch_size=2)
    assert len(model.sampled_weights) == 4


def test_train_continue_training_keeps_previous_samples():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    model.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=2)
    first_len = len(model.sampled_weights)
    model.train(x, y, num_steps=2, keep_every=1, num_burn_in_steps=0, batch_size=2, continue_training=True)
    assert len(model.sampled_weights) >= first_len


def test_predict_without_normalization_uses_sampled_weights():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    model.sampled_weights = [model.network_weights, model.network_weights]
    mu, var = model.predict(np.array([[2.0]], dtype=np.float64))
    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert np.allclose(mu, [2.0])
    assert np.allclose(var, [1.0])


def test_predict_return_individual_predictions():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    w1 = model.network_weights
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[3.0], [0.0]], dtype=torch.float64)
    w2 = model.network_weights
    model.sampled_weights = [w1, w2]
    mu, var, funcs = model.predict(np.array([[2.0]], dtype=np.float64), return_individual_predictions=True)
    assert funcs.shape == (2, 1)
    assert np.allclose(funcs[:, 0], [2.0, 6.0])
    assert np.allclose(mu, [4.0])
    assert var.shape == (1,)


def test_predict_single_returns_single_sample_prediction():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    model.sampled_weights = [model.network_weights]
    out = model.predict_single(np.array([3.0], dtype=np.float64), 0)
    assert out.shape == (1, 2)
    assert np.allclose(out[0, 0], 7.0)


def test_f_gradient_computes_input_gradient():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([1.0, 0.0], dtype=torch.float64)
    g = model.f_gradient(np.array([3.0], dtype=np.float64), model.network_weights)
    assert g.shape == (1,)
    assert np.allclose(g, [2.0])


def test_predictive_mean_gradient_averages_sample_gradients():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    w1 = model.network_weights
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[3.0], [0.0]], dtype=torch.float64)
    w2 = model.network_weights
    model.sampled_weights = [w1, w2]
    g = model.predictive_mean_gradient(np.array([2.0], dtype=np.float64))
    assert g.shape == (1,)
    assert np.allclose(g, [2.0])


def test_predictive_variance_gradient_has_expected_shape():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=False, normalize_output=False, use_double_precision=True)
    model.model = tiny_network(1).double()
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
        model.model.linear.bias[:] = torch.tensor([0.0, 0.0], dtype=torch.float64)
    w1 = model.network_weights
    with torch.no_grad():
        model.model.linear.weight[:] = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
    w2 = model.network_weights
    model.sampled_weights = [w1, w2]
    g = model.predictive_variance_gradient(np.array([2.0], dtype=np.float64))
    assert g.shape == (1,)
    assert np.isfinite(g).all()


def test_normalize_input_and_output_roundtrip():
    model = boh.Bohamiann(get_network=tiny_network, normalize_input=True, normalize_output=True)
    x = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    xn, m, s = model.normalize_input(x)
    x_back = boh.zero_mean_unit_var_denormalization(xn, m, s)
    assert np.allclose(x_back, x)

    y = np.array([2.0, 6.0], dtype=np.float64)
    yn, my, sy = model.normalize_output(y)
    y_back = boh.zero_mean_unit_var_denormalization(yn, my, sy)
    assert np.allclose(y_back, y)
