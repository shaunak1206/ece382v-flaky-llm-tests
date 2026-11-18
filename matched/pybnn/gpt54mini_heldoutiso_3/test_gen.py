```python
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

    def step(self):
        self.step_calls += 1


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


def make_bohamiann(monkeypatch, **kwargs):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: iter(loader))
    return boh.Bohamiann(get_network=TinyNet, **kwargs)


def test_nll_zero_when_predictions_match_target():
    inp = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)
    loss = boh.nll(inp, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_nll_increases_with_error():
    inp = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([2.0], dtype=torch.float64)
    loss = boh.nll(inp, target)
    assert loss.item() > 0.0


def test_get_default_network_output_shape():
    net = boh.get_default_network(3)
    x = torch.zeros(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_getter_returns_numpy_arrays(monkeypatch):
    model = TinyNet(2).double()
    bnn = make_bohamiann(monkeypatch)
    bnn.model = model
    weights = bnn.network_weights
    assert isinstance(weights, tuple)
    assert len(weights) == 1
    assert isinstance(weights[0], np.ndarray)
    assert weights[0].shape == tuple(model.linear.weight.shape)


def test_network_weights_setter_assigns_values(monkeypatch):
    model = TinyNet(2).double()
    bnn = make_bohamiann(monkeypatch)
    bnn.model = model
    new_w = np.array([[1.5, -2.0], [3.25, 4.75]], dtype=np.float64)
    bnn.network_weights = [new_w]
    assert np.allclose(model.linear.weight.detach().numpy(), new_w)


def test_train_rejects_invalid_batch_size(monkeypatch):
    bnn = make_bohamiann(monkeypatch)
    x = np.zeros((3, 2), dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)
    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=1, batch_size=0)


def test_train_accepts_column_vector_targets(monkeypatch):
    bnn = make_bohamiann(monkeypatch)
    x = np.zeros((3, 2), dtype=np.float64)
    y = np.zeros((3, 1), dtype=np.float64)
    bnn.train(x, y, num_steps=0, batch_size=1, verbose=False)
    assert bnn.is_trained is True


def test_train_initializes_sampler_and_model(monkeypatch):
    bnn = make_bohamiann(monkeypatch, sampling_method="sgld", use_double_precision=False, normalize_input=False, normalize_output=False)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    bnn.train(x, y, num_steps=1, batch_size=1, continue_training=False)
    assert isinstance(bnn.model, TinyNet)
    assert isinstance(bnn.sampler, DummySampler)
    assert bnn.sampler.kwargs["lr"] == np.float32(1e-2)
    assert bnn.sampler.kwargs["scale_grad"] == 2
    assert bnn.is_trained is True


def test_train_collects_sampled_weights_after_burn_in(monkeypatch):
    bnn = make_bohamiann(monkeypatch, sampling_method="adaptive_sghmc", normalize_input=False, normalize_output=False)
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    bnn.train(x, y, num_steps=4, num_burn_in_steps=1, keep_every=1, batch_size=1)
    assert len(bnn.sampled_weights) >= 1
    assert isinstance(bnn.sampled_weights[0], tuple)


def test_continue_training_keeps_existing_sampled_weights(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False)
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)
    bnn.train(x, y, num_steps=1, batch_size=1)
    old_model = bnn.model
    old_samples = list(bnn.sampled_weights)
    bnn.train(x, y, num_steps=1, batch_size=1, continue_training=True)
    assert bnn.model is old_model
    assert len(bnn.sampled_weights) >= len(old_samples)


def test_predict_returns_mean_and_variance(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=True)
    model = TinyNet(2).double()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
    bnn.model = model
    bnn.sampled_weights = [tuple(p.detach().cpu().numpy() for p in bnn.model.parameters())]
    mu, var = bnn.predict(np.array([[2.0, 3.0]], dtype=np.float64))
    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert mu[0] == pytest.approx(2.0, abs=1e-12)
    assert var[0] == pytest.approx(1.0, abs=1e-12)


def test_predict_return_individual_predictions(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=True)
    model = TinyNet(2).double()
    bnn.model = model
    w1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    w2 = np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    bnn.sampled_weights = [(w1,), (w2,)]
    mu, var, indiv = bnn.predict(np.array([[2.0, 0.0]], dtype=np.float64), return_individual_predictions=True)
    assert indiv.shape == (2, 1)
    assert np.allclose(indiv[:, 0], np.array([2.0, 6.0]))
    assert mu[0] == pytest.approx(4.0, abs=1e-12)
    assert var[0] == pytest.approx(5.0, abs=1e-12)


def test_predict_single_uses_selected_sample(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=True)
    model = TinyNet(2).double()
    bnn.model = model
    w1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    w2 = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    bnn.sampled_weights = [(w1,), (w2,)]
    out = bnn.predict_single(np.array([2.0, 0.0], dtype=np.float64), sample_index=1)
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(10.0, abs=1e-12)


def test_f_gradient_for_linear_network(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=True)
    model = TinyNet(2).double()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[2.0, -1.0], [0.0, 0.0]], dtype=torch.float64))
    bnn.model = model
    weights = tuple(p.detach().cpu().numpy() for p in bnn.model.parameters())
    g = bnn.f_gradient(np.array([3.0, 4.0], dtype=np.float64), weights=weights)
    assert np.allclose(g, np.array([2.0, -1.0]))


def test_predictive_mean_gradient_matches_average(monkeypatch):
    bnn = make_bohamiann(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=True)
    model = TinyNet(2).double()
    bnn.model = model
    w1 = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    w2 = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)
    bnn.sampled_weights = [(w1,), (w2,)]
    g = bnn.predictive_mean_gradient(np.array([1.0, 1.0], dtype=np.float64))
    assert np.allclose(g, np.array([2.0, 3.0]))
