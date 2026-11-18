import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.linear(x)


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


def make_bnn(monkeypatch, normalize_input=True, normalize_output=True, use_double_precision=False):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    return boh.Bohamiann(
        get_network=lambda input_dimensionality: TinyNet(input_dimensionality),
        normalize_input=normalize_input,
        normalize_output=normalize_output,
        use_double_precision=use_double_precision,
        print_every_n_steps=1_000_000,
    )


def test_nll_zero_for_perfect_prediction():
    inp = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1.0, 2.0], dtype=torch.float32)
    loss = boh.nll(inp, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_nll_increases_with_error():
    perfect = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    bad = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([1.0], dtype=torch.float32)
    assert boh.nll(bad, target) > boh.nll(perfect, target)


def test_get_default_network_output_shape():
    net = boh.get_default_network(3)
    x = torch.zeros(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_getter_returns_tuple_of_ndarrays(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.model = TinyNet(2)
    weights = bnn.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)
    assert len(weights) == len(list(bnn.model.parameters()))


def test_network_weights_setter_updates_parameters(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.model = TinyNet(2)

    new_weights = []
    for p in bnn.model.parameters():
        arr = np.ones_like(p.detach().numpy()) * 3.14
        new_weights.append(arr)

    bnn.network_weights = new_weights

    for p in bnn.model.parameters():
        assert np.allclose(p.detach().numpy(), 3.14)


def test_normalize_input_and_output_roundtrip(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([10.0, 20.0])

    xn, xm, xs = bnn.normalize_input(x)
    x_back = boh.zero_mean_unit_var_denormalization(xn, xm, xs)
    yn, ym, ys = bnn.normalize_output(y)
    y_back = boh.zero_mean_unit_var_denormalization(yn, ym, ys)

    assert np.allclose(x_back, x)
    assert np.allclose(y_back, y)


def test_train_sets_is_trained_and_records_sampled_weights(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.random.RandomState(0).randn(6, 2)
    y = np.random.RandomState(1).randn(6)

    bnn.train(x, y, num_steps=5, keep_every=1, num_burn_in_steps=0, batch_size=2, verbose=False)

    assert bnn.is_trained is True
    assert bnn.sampler.step_calls == 5
    assert len(bnn.sampled_weights) == 5


def test_train_warns_and_reduces_batch_size_when_too_large(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.random.RandomState(0).randn(3, 2)
    y = np.random.RandomState(1).randn(3)

    bnn.train(x, y, num_steps=1, keep_every=1, num_burn_in_steps=0, batch_size=10, verbose=False)

    assert bnn.sampler.kwargs["scale_grad"] == 3.0


def test_train_rejects_invalid_batch_size(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.random.RandomState(0).randn(3, 2)
    y = np.random.RandomState(1).randn(3)

    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=1, batch_size=0)


def test_train_rejects_invalid_target_shape(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.random.RandomState(0).randn(3, 2)
    y = np.random.RandomState(1).randn(3, 2, 1)

    with pytest.raises(AssertionError):
        bnn.train(x, y, num_steps=1, batch_size=1)


def test_train_and_evaluate_returns_curve_lengths(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x_train = np.random.RandomState(0).randn(6, 2)
    y_train = np.random.RandomState(1).randn(6)
    x_valid = np.random.RandomState(2).randn(4, 2)
    y_valid = np.random.RandomState(3).randn(4)

    steps, ll, mse = bnn.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=0,
        batch_size=2,
        verbose=False,
    )

    assert steps == [2, 4]
    assert len(ll) == len(mse) == 2


def test_predict_returns_mean_variance_and_individual_predictions(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2)

    # deterministic weights: output = [sum(x), -sum(x)]
    with torch.no_grad():
        bnn.model.linear.weight.zero_()
        bnn.model.linear.bias[:] = torch.tensor([1.0, np.log(0.5)], dtype=bnn.model.linear.bias.dtype)

    bnn.sampled_weights = [tuple(p.detach().numpy().copy() for p in bnn.model.parameters())]
    x = np.array([[0.0, 0.0], [1.0, 2.0]])

    mean, var, individual = bnn.predict(x, return_individual_predictions=True)

    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert individual.shape == (1, 2)
    assert np.allclose(mean, [1.0, 1.0])


def test_predict_single_matches_predict_for_single_sample(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2)
    with torch.no_grad():
        bnn.model.linear.weight.zero_()
        bnn.model.linear.bias[:] = torch.tensor([2.0, np.log(1.0)], dtype=bnn.model.linear.bias.dtype)
    weights = tuple(p.detach().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [weights]

    pred_single = bnn.predict_single(np.array([0.2, -0.3]), 0)
    pred_all = bnn.predict(np.array([[0.2, -0.3]]), return_individual_predictions=True)[2]

    assert pred_single.shape == (1, 2)
    assert np.allclose(pred_single, pred_all[0:1])


def test_f_gradient_returns_correct_linear_gradient(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2)
    with torch.no_grad():
        bnn.model.linear.weight[:] = torch.tensor([[2.0, -1.0], [0.0, 0.0]], dtype=bnn.model.linear.weight.dtype)
        bnn.model.linear.bias.zero_()

    weights = tuple(p.detach().numpy().copy() for p in bnn.model.parameters())
    g = bnn.f_gradient(np.array([3.0, 4.0]), weights)

    assert np.allclose(g, np.array([2.0, -1.0]))


def test_predictive_mean_gradient_equals_single_gradient_for_one_sample(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2)
    with torch.no_grad():
        bnn.model.linear.weight[:] = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=bnn.model.linear.weight.dtype)
        bnn.model.linear.bias.zero_()
    weights = tuple(p.detach().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [weights]

    g = bnn.predictive_mean_gradient(np.array([5.0, 6.0]))

    assert np.allclose(g, np.array([1.0, 2.0]))


def test_predictive_variance_gradient_is_zero_for_single_sample(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2)
    with torch.no_grad():
        bnn.model.linear.weight[:] = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=bnn.model.linear.weight.dtype)
        bnn.model.linear.bias.zero_()
    weights = tuple(p.detach().numpy().copy() for p in bnn.model.parameters())
    bnn.sampled_weights = [weights]

    g = bnn.predictive_variance_gradient(np.array([5.0, 6.0]))

    assert np.allclose(g, np.zeros(2))
