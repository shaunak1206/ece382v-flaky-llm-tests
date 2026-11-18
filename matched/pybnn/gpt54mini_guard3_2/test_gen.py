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


class LinearTwoOutputNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 1, bias=True)
        self.log_var_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mean = self.linear(x)
        log_var = torch.zeros_like(mean) + self.log_var_bias
        return torch.cat([mean, log_var], dim=1)


class ConstantTwoOutputNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        batch = x.shape[0]
        mean = torch.zeros(batch, 1, dtype=x.dtype, device=x.device)
        log_var = torch.zeros(batch, 1, dtype=x.dtype, device=x.device)
        return torch.cat([mean, log_var], dim=1)


def infinite_cycle(loader):
    while True:
        for batch in loader:
            yield batch


def make_model(monkeypatch, get_network=LinearTwoOutputNet, **kwargs):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", infinite_cycle)

    return boh.Bohamiann(
        get_network=get_network,
        normalize_input=False,
        normalize_output=False,
        use_double_precision=False,
        print_every_n_steps=10,
        **kwargs,
    )


def test_get_default_network_output_shape():
    torch.manual_seed(0)
    net = boh.get_default_network(3)
    x = torch.zeros(5, 3)
    y = net(x)
    assert y.shape == (5, 2)


def test_nll_matches_manual_computation():
    torch.manual_seed(0)
    inp = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float32)
    target = torch.tensor([1.0, 4.0], dtype=torch.float32)

    got = boh.nll(inp, target).item()

    mean = inp[:, 0:1]
    log_var = inp[:, 1:2]
    var_inv = 1.0 / (torch.exp(log_var) + 1e-16)
    mse = (target.view(-1, 1) - mean) ** 2
    ll = torch.sum(torch.sum(-mse * (0.5 * var_inv) - 0.5 * log_var, dim=1)) / inp.size(0)
    expected = (-ll).item()

    assert got == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_network_weights_roundtrip(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch)
    model.model = LinearTwoOutputNet(2).float()

    original = model.network_weights
    modified = [w + 1.0 for w in original]
    model.network_weights = modified

    for param, expected in zip(model.model.parameters(), modified):
        assert np.allclose(param.detach().numpy(), expected)
    assert len(model.network_weights) == len(modified)


def test_train_sets_is_trained_and_collects_samples(monkeypatch):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)

    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    model.train(x, y, num_steps=5, num_burn_in_steps=1, keep_every=1, batch_size=2, lr=1e-2, verbose=False)

    assert model.is_trained is True
    assert len(model.sampled_weights) == 4
    assert model.sampler.step_calls == 5


def test_train_clears_previous_samples_when_not_continuing(monkeypatch):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(monkeypatch)

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)

    model.sampled_weights = [("old",)]
    model.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2, verbose=False)

    assert len(model.sampled_weights) == 2
    assert model.sampled_weights[0] != ("old",)


def test_continue_training_appends_samples(monkeypatch):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(monkeypatch)

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)

    model.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2, verbose=False)
    first_len = len(model.sampled_weights)
    model.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2,
                verbose=False, continue_training=True)

    assert len(model.sampled_weights) == first_len + 2


def test_predict_returns_expected_mean_and_variance(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()
    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.model.linear.bias.copy_(torch.tensor([0.5]))
        model.model.log_var_bias.copy_(torch.tensor(np.log(0.25), dtype=torch.float32))
    model.sampled_weights = [model.network_weights]

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean, var = model.predict(x)

    expected_mean = np.array([0.5, 2.5], dtype=np.float32)
    expected_var = np.array([0.25, 0.25], dtype=np.float32)

    assert np.allclose(mean, expected_mean)
    assert np.allclose(var, expected_var)


def test_predict_return_individual_predictions(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.model.linear.bias.copy_(torch.tensor([0.0]))
        model.model.log_var_bias.copy_(torch.tensor(0.0))
    w1 = model.network_weights

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[2.0, 0.0]]))
        model.model.linear.bias.copy_(torch.tensor([1.0]))
        model.model.log_var_bias.copy_(torch.tensor(0.0))
    w2 = model.network_weights

    model.sampled_weights = [w1, w2]
    x = np.array([[3.0, 4.0]], dtype=np.float32)

    mean, var, indiv = model.predict(x, return_individual_predictions=True)

    assert mean.shape == (1,)
    assert var.shape == (1,)
    assert indiv.shape == (2, 1)
    assert np.allclose(indiv[:, 0], np.array([3.0, 7.0], dtype=np.float32))
    assert np.allclose(mean, np.array([5.0], dtype=np.float32))


def test_predict_single_uses_selected_sample(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.model.linear.bias.copy_(torch.tensor([0.0]))
        model.model.log_var_bias.copy_(torch.tensor(0.0))
    w = model.network_weights
    model.sampled_weights = [w]

    out = model.predict_single(np.array([3.0, 4.0], dtype=np.float32), 0)
    assert out.shape == (1, 2)
    assert np.allclose(out[0, 0], 11.0)
    assert np.allclose(out[0, 1], 0.0)


def test_f_gradient_matches_linear_weights(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[2.0, -3.0]]))
        model.model.linear.bias.copy_(torch.tensor([0.0]))
        model.model.log_var_bias.copy_(torch.tensor(0.0))
    w = model.network_weights

    g = model.f_gradient(np.array([1.0, 2.0], dtype=np.float32), w)
    assert np.allclose(g, np.array([2.0, -3.0], dtype=np.float32), atol=1e-6)


def test_predictive_mean_gradient_with_one_sample_equals_f_gradient(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[1.5, -0.5]]))
        model.model.linear.bias.copy_(torch.tensor([0.0]))
        model.model.log_var_bias.copy_(torch.tensor(0.0))
    w = model.network_weights
    model.sampled_weights = [w]

    x = np.array([2.0, 5.0], dtype=np.float32)
    g1 = model.f_gradient(x, w)
    g2 = model.predictive_mean_gradient(x)
    assert np.allclose(g1, g2, atol=1e-6)


def test_predictive_variance_gradient_is_zero_for_identical_samples(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)
    model.model = LinearTwoOutputNet(2).float()

    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.model.linear.bias.copy_(torch.tensor([0.0]))
        model.model.log_var_bias.copy_(torch.tensor(np.log(0.5), dtype=torch.float32))
    w = model.network_weights
    model.sampled_weights = [w, w]

    g = model.predictive_variance_gradient(np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(g, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_train_and_evaluate_returns_learning_curves(monkeypatch):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)

    x_train = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    x_valid = np.array([[1.5, 0.5], [2.5, 0.5]], dtype=np.float32)
    y_valid = np.array([1.5, 2.5], dtype=np.float32)

    steps, ll_curve, mse_curve = model.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=4, validate_every_n_steps=2, keep_every=1,
        num_burn_in_steps=1, batch_size=2, verbose=False
    )

    assert steps == [3, 5]
    assert len(ll_curve) == 2
    assert len(mse_curve) == 2
    assert all(np.isfinite(v) for v in ll_curve)
    assert all(np.isfinite(v) for v in mse_curve)


def test_train_warns_and_reduces_batch_size_when_needed(monkeypatch):
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_model(monkeypatch, get_network=LinearTwoOutputNet)

    x = np.array([[0.0, 1.0]], dtype=np.float32)
    y = np.array([0.0], dtype=np.float32)

    with pytest.warns(None) as record:
        model.train(x, y, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=5, verbose=False)

    assert len(record) == 0
    assert model.is_trained is True
    assert len(model.sampled_weights) == 1


def test_constant_network_predicts_zero_mean_and_unit_variance(monkeypatch):
    torch.manual_seed(0)
    model = make_model(monkeypatch, get_network=ConstantTwoOutputNet)
    model.model = ConstantTwoOutputNet(2).float()
    model.sampled_weights = [model.network_weights]

    mean, var = model.predict(np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.allclose(mean, np.array([0.0], dtype=np.float32))
    assert np.allclose(var, np.array([1.0], dtype=np.float32))
