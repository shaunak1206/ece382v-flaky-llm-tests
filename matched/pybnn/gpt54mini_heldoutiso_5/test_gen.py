import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class DummySampler:
    def __init__(self, params, **kwargs):
        self.params = list(params)
        self.kwargs = kwargs
        self.zero_grad_called = 0
        self.step_called = 0

    def zero_grad(self):
        self.zero_grad_called += 1
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_called += 1
        with torch.no_grad():
            for p in self.params:
                p.add_(0.0)


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.linear(x)


def make_model(
    normalize_input=False,
    normalize_output=False,
    use_double_precision=False,
    sampling_method="sgld",
):
    model = boh.Bohamiann(
        get_network=lambda input_dimensionality: TinyNet(input_dimensionality),
        normalize_input=normalize_input,
        normalize_output=normalize_output,
        use_double_precision=use_double_precision,
        sampling_method=sampling_method,
        print_every_n_steps=10,
    )
    return model


def set_linear_weights(model, weight, bias):
    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor(weight, dtype=model.model.linear.weight.dtype))
        model.model.linear.bias.copy_(torch.tensor(bias, dtype=model.model.linear.bias.dtype))


def test_nll_matches_manual_formula():
    inp = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.5, 1.0], dtype=torch.float64)

    got = boh.nll(inp, target).item()

    mean = np.array([1.0, 2.0])
    log_var = np.array([0.0, np.log(2.0)])
    var_inv = 1.0 / (np.exp(log_var) + 1e-16)
    mse = (target.numpy() - mean) ** 2
    ll = np.sum(-mse * (0.5 * var_inv) - 0.5 * log_var) / 2.0
    expected = -ll

    assert got == pytest.approx(expected)


def test_get_default_network_output_shape():
    net = boh.get_default_network(3)
    x = torch.zeros(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_getter_and_setter_roundtrip():
    model = make_model()
    model.model = TinyNet(2).float()
    original = model.network_weights

    new_weights = [
        np.ones_like(original[0]) * 2.0,
        np.array([0.5, -0.5], dtype=original[1].dtype),
    ]
    model.network_weights = new_weights

    got = model.network_weights
    assert len(got) == 2
    assert np.allclose(got[0], new_weights[0])
    assert np.allclose(got[1], new_weights[1])


def test_train_asserts_on_invalid_batch_size():
    model = make_model()
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)

    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=0)


def test_train_asserts_on_invalid_target_shape():
    model = make_model()
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=1)


def test_train_warns_and_uses_all_datapoints_when_batch_too_large(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    model = make_model(sampling_method="sgld")

    x = np.random.RandomState(0).randn(2, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(2).astype(np.float32)

    with pytest.warns(None) as record:
        model.train(x, y, num_steps=1, batch_size=10, continue_training=False)

    assert any("Not enough datapoints" in str(w.message) for w in record)
    assert model.is_trained is True
    assert model.sampler.kwargs["scale_grad"] == 2


def test_train_initializes_correct_sampler_adaptive(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)

    model = make_model(sampling_method="adaptive_sghmc")
    x = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(4).astype(np.float32)

    model.train(x, y, num_steps=1, batch_size=2, continue_training=False)

    assert isinstance(model.sampler, DummySampler)
    assert model.sampler.kwargs["num_burn_in_steps"] == 3000
    assert model.sampler.kwargs["scale_grad"] == np.float64(4.0)


def test_train_initializes_correct_sampler_other_methods(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)

    x = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(4).astype(np.float32)

    for method in ["sgld", "preconditioned_sgld", "sghmc"]:
        model = make_model(sampling_method=method)
        model.train(x, y, num_steps=1, batch_size=2, continue_training=False)
        assert isinstance(model.sampler, DummySampler)


def test_train_clears_sampled_weights_when_not_continuing(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)

    model = make_model(sampling_method="sgld")
    model.sampled_weights = [(np.array([1.0]),)]
    x = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(4).astype(np.float32)

    model.train(x, y, num_steps=1, batch_size=2, continue_training=False)
    assert model.sampled_weights == []


def test_train_continue_training_keeps_existing_sampled_weights(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)

    model = make_model(sampling_method="sgld")
    model.model = TinyNet(2).float()
    model.sampler = DummySampler(model.model.parameters())
    model.sampled_weights = [(np.array([1.0]),)]

    x = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(4).astype(np.float32)

    model.train(x, y, num_steps=1, batch_size=2, continue_training=True)
    assert model.sampled_weights == [(np.array([1.0]),)]


def test_train_samples_weights_after_burn_in(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)

    model = make_model(sampling_method="sgld")
    x = np.random.RandomState(0).randn(4, 2).astype(np.float32)
    y = np.random.RandomState(1).randn(4).astype(np.float32)

    model.train(x, y, num_steps=3, batch_size=2, num_burn_in_steps=0, keep_every=1)
    assert model.is_trained is True
    assert len(model.sampled_weights) >= 1


def test_predict_without_normalization_uses_sampled_weights():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    set_linear_weights(model, weight=[[1.0, 0.0], [0.0, 0.0]], bias=[0.0, 0.0])

    model.sampled_weights = [model.network_weights]
    mean, var = model.predict(np.array([[2.0, 3.0]], dtype=np.float32))

    assert mean.shape == (1,)
    assert var.shape == (1,)
    assert mean[0] == pytest.approx(2.0)
    assert var[0] == pytest.approx(1.0)


def test_predict_return_individual_predictions_shape():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    set_linear_weights(model, weight=[[1.0, 0.0], [0.0, 0.0]], bias=[0.0, 0.0])
    w1 = model.network_weights
    set_linear_weights(model, weight=[[2.0, 0.0], [0.0, 0.0]], bias=[0.0, 0.0])
    w2 = model.network_weights
    model.sampled_weights = [w1, w2]

    mean, var, indiv = model.predict(np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32), return_individual_predictions=True)

    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert indiv.shape == (2, 2)
    assert np.allclose(indiv[:, 0], np.array([1.0, 2.0]))


def test_predict_single_returns_single_network_output():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    set_linear_weights(model, weight=[[1.0, 0.0], [0.0, 0.0]], bias=[0.0, 0.0])
    model.sampled_weights = [model.network_weights]

    out = model.predict_single(np.array([3.0, 4.0], dtype=np.float32), sample_index=0)

    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(3.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_f_gradient_without_normalization_matches_linear_model():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    set_linear_weights(model, weight=[[2.0, -1.0], [0.0, 0.0]], bias=[0.0, 0.0])

    g = model.f_gradient(np.array([1.0, 2.0], dtype=np.float32), weights=model.network_weights)

    assert np.allclose(g, np.array([2.0, -1.0], dtype=np.float32))


def test_predictive_mean_gradient_averages_sample_gradients():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    w1 = (np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    w2 = (np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    model.sampled_weights = [w1, w2]

    g = model.predictive_mean_gradient(np.array([0.5, 1.5], dtype=np.float32))

    assert np.allclose(g, np.array([2.0, 0.0], dtype=np.float32))


def test_predictive_variance_gradient_zero_for_identical_samples():
    model = make_model(normalize_input=False, normalize_output=False, use_double_precision=False)
    model.model = TinyNet(2).float()
    w = (np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32))
    model.sampled_weights = [w, w]

    g = model.predictive_variance_gradient(np.array([0.5, 1.5], dtype=np.float32))

    assert np.allclose(g, np.zeros(2, dtype=np.float32))
