import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class DummySampler:
    def __init__(self, *args, **kwargs):
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


class LinearOutNet(nn.Module):
    """Small deterministic network with a single linear layer and a 2-dim output."""
    def __init__(self, input_dimensionality):
        super().__init__()
        self.fc = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


class ScalarNet(nn.Module):
    """Network used for gradient tests: output equals x0 + 2*x1 and log_var=0."""
    def __init__(self, input_dimensionality):
        super().__init__()
        self.fc = nn.Linear(input_dimensionality, 2, bias=False)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[0, 0] = 1.0
            self.fc.weight[0, 1] = 2.0
            self.fc.weight[1, :] = 0.0

    def forward(self, x):
        return self.fc(x)


def make_bnn(monkeypatch, get_network=LinearOutNet, **kwargs):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    model = boh.Bohamiann(get_network=get_network, **kwargs)
    return model


def set_simple_linear_weights(model, weight_2x2, bias_2):
    with torch.no_grad():
        model.model.fc.weight.copy_(torch.tensor(weight_2x2, dtype=model.model.fc.weight.dtype))
        model.model.fc.bias.copy_(torch.tensor(bias_2, dtype=model.model.fc.bias.dtype))


def test_nll_matches_manual_computation():
    inp = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    tgt = torch.tensor([1.0, 3.0], dtype=torch.float64)
    got = boh.nll(inp, tgt).item()

    # Manual average negative log-likelihood
    mse1 = (1.0 - 1.0) ** 2
    ll1 = -0.5 * mse1 * np.exp(-0.0) - 0.5 * 0.0
    mse2 = (3.0 - 2.0) ** 2
    ll2 = -0.5 * mse2 * np.exp(-np.log(2.0)) - 0.5 * np.log(2.0)
    expected = -(ll1 + ll2) / 2.0

    assert np.isclose(got, expected)


def test_network_weights_getter_and_setter_roundtrip(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.model = LinearOutNet(2).double()

    weights = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([5.0, 6.0], dtype=np.float64),
    )
    bnn.network_weights = weights
    got = bnn.network_weights

    assert len(got) == 2
    np.testing.assert_allclose(got[0], weights[0])
    np.testing.assert_allclose(got[1], weights[1])


def test_train_rejects_invalid_batch_size(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.zeros((3, 2), dtype=np.float64)
    y = np.zeros(3, dtype=np.float64)

    with pytest.raises(AssertionError, match="Invalid batch size"):
        bnn.train(x, y, batch_size=0, num_steps=1, num_burn_in_steps=0)


def test_train_rejects_invalid_target_shape(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.zeros((3, 2), dtype=np.float64)
    y = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(AssertionError, match="Targets need to be in vector format"):
        bnn.train(x, y, batch_size=1, num_steps=1, num_burn_in_steps=0)


def test_train_uses_all_points_when_batch_size_too_large(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    captured = {}

    def fake_dataloader(loader):
        captured["batch_size"] = loader.batch_size
        return iter([(next(iter(loader))[0], next(iter(loader))[1])])

    monkeypatch.setattr(boh, "infinite_dataloader", fake_dataloader)

    bnn.train(x, y, batch_size=5, num_steps=1, num_burn_in_steps=0, verbose=False)

    assert captured["batch_size"] == 2
    assert bnn.is_trained is True


@pytest.mark.parametrize(
    "sampling_method",
    ["adaptive_sghmc", "sgld", "sghmc", "preconditioned_sgld"],
)
def test_train_initializes_selected_sampler(monkeypatch, sampling_method):
    bnn = make_bnn(monkeypatch, sampling_method=sampling_method)
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    bnn.train(x, y, batch_size=2, num_steps=1, num_burn_in_steps=0, verbose=False)

    assert isinstance(bnn.sampler, DummySampler)
    assert bnn.model is not None


def test_train_collects_sampled_weights_after_burn_in(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    bnn.train(x, y, batch_size=2, num_steps=4, keep_every=1, num_burn_in_steps=0, verbose=False)

    assert len(bnn.sampled_weights) == 3
    assert all(isinstance(w, tuple) for w in bnn.sampled_weights)


def test_continue_training_keeps_existing_samples(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    bnn.train(x, y, batch_size=2, num_steps=2, keep_every=1, num_burn_in_steps=0, verbose=False)
    first_model = bnn.model
    first_samples = list(bnn.sampled_weights)

    bnn.train(
        x, y, batch_size=2, num_steps=1, keep_every=1, num_burn_in_steps=0,
        verbose=False, continue_training=True
    )

    assert bnn.model is first_model
    assert len(bnn.sampled_weights) >= len(first_samples)


def test_predict_returns_mean_variance_and_individual_predictions(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=False)
    bnn.model = LinearOutNet(1).float()
    set_simple_linear_weights(bnn, [[1.0], [0.0]], [0.0, 0.0])

    w1 = (
        np.array([[1.0], [0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    w2 = (
        np.array([[3.0], [0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    bnn.sampled_weights = [w1, w2]

    mu, var, indiv = bnn.predict(np.array([[2.0]], dtype=np.float32), return_individual_predictions=True)

    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert indiv.shape == (2, 1)
    np.testing.assert_allclose(mu, np.array([4.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(indiv[:, 0], np.array([2.0, 6.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(var, np.array([4.0], dtype=np.float32), atol=1e-6)


def test_predict_applies_normalization_and_denormalization(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=True, normalize_output=True, use_double_precision=False)
    bnn.model = LinearOutNet(1).float()
    set_simple_linear_weights(bnn, [[1.0], [np.log(1.0)]], [0.0, 0.0])

    bnn.x_mean = np.array([1.0], dtype=np.float32)
    bnn.x_std = np.array([2.0], dtype=np.float32)
    bnn.y_mean = 10.0
    bnn.y_std = 3.0

    bnn.sampled_weights = [(
        np.array([[1.0], [0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )]

    mu, var = bnn.predict(np.array([[3.0]], dtype=np.float32))
    np.testing.assert_allclose(mu, np.array([13.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(var, np.array([9.0], dtype=np.float32), atol=1e-6)


def test_predict_single_returns_single_sample_prediction(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=False)
    bnn.model = LinearOutNet(1).float()
    bnn.sampled_weights = [(
        np.array([[2.0], [0.0]], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )]

    out = bnn.predict_single(np.array([3.0], dtype=np.float32), sample_index=0)
    np.testing.assert_allclose(out, np.array([6.0, 0.0], dtype=np.float32), atol=1e-6)


def test_f_gradient_for_linear_model(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False, use_double_precision=False)
    bnn.model = ScalarNet(2).float()
    weights = tuple(p.detach().cpu().numpy() for p in bnn.model.parameters())

    grad = bnn.f_gradient(np.array([5.0, 7.0], dtype=np.float32), weights=weights)
    np.testing.assert_allclose(grad, np.array([1.0, 2.0], dtype=np.float32), atol=1e-6)


def test_predictive_mean_gradient_averages_sample_gradients(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.sampled_weights = ["a", "b", "c"]

    def fake_f_gradient(x_test, weights):
        mapping = {
            "a": np.array([1.0, 0.0], dtype=np.float64),
            "b": np.array([0.0, 2.0], dtype=np.float64),
            "c": np.array([3.0, 4.0], dtype=np.float64),
        }
        return mapping[weights]

    monkeypatch.setattr(bnn, "f_gradient", fake_f_gradient)

    grad = bnn.predictive_mean_gradient(np.array([0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(grad, np.array([4.0 / 3.0, 2.0], dtype=np.float64))


def test_predictive_variance_gradient_uses_mean_and_sample_gradients(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.sampled_weights = ["a", "b"]

    monkeypatch.setattr(bnn, "predict", lambda x, return_individual_predictions=False: (
        np.array([1.0], dtype=np.float64),
        np.array([4.0], dtype=np.float64),
        np.array([[0.0], [2.0]], dtype=np.float64),
    ))
    monkeypatch.setattr(bnn, "predictive_mean_gradient", lambda x: np.array([1.0, 1.0], dtype=np.float64))
    monkeypatch.setattr(
        bnn,
        "f_gradient",
        lambda x, weights: np.array([2.0, 0.0], dtype=np.float64) if weights == "a" else np.array([0.0, 2.0], dtype=np.float64),
    )

    grad = bnn.predictive_variance_gradient(np.array([0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(grad, np.array([1.0, 1.0], dtype=np.float64))


def test_train_and_evaluate_returns_learning_curves(monkeypatch):
    bnn = make_bnn(monkeypatch)
    x_train = np.array([[0.0], [1.0]], dtype=np.float64)
    y_train = np.array([0.0, 1.0], dtype=np.float64)
    x_valid = np.array([[2.0]], dtype=np.float64)
    y_valid = np.array([2.0], dtype=np.float64)

    def fake_train(*args, **kwargs):
        return None

    calls = {"predict": 0}

    def fake_predict(x):
        calls["predict"] += 1
        return np.array([2.0], dtype=np.float64), np.array([1.0], dtype=np.float64)

    monkeypatch.setattr(bnn, "train", fake_train)
    monkeypatch.setattr(bnn, "predict", fake_predict)

    steps, ll, mse = bnn.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=4, validate_every_n_steps=2, keep_every=1, num_burn_in_steps=1,
        batch_size=1, verbose=False
    )

    assert steps == [3, 5]
    assert len(ll) == 2
    assert len(mse) == 2
    assert calls["predict"] == 2
