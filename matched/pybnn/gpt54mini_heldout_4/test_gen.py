import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class DummyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.linear(x)


class TrackingSampler:
    instances = []

    def __init__(self, params, **kwargs):
        self.params = list(params)
        self.kwargs = kwargs
        self.zero_grad_called = 0
        self.step_called = 0
        TrackingSampler.instances.append(self)

    def zero_grad(self):
        self.zero_grad_called += 1
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.step_called += 1


def make_model(
    normalize_input=True,
    normalize_output=True,
    use_double_precision=True,
    get_network=None,
    sampling_method="adaptive_sghmc",
):
    if get_network is None:
        get_network = lambda input_dimensionality: DummyNet(input_dimensionality)
    model = boh.Bohamiann(
        get_network=get_network,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
        use_double_precision=use_double_precision,
        sampling_method=sampling_method,
        print_every_n_steps=1,
    )
    return model


def test_nll_matches_manual_formula():
    inp = torch.tensor([[0.0, 0.0], [1.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([0.0, 2.0], dtype=torch.float64)

    got = boh.nll(inp, target).item()

    mse = np.array([(0.0 - 0.0) ** 2, (2.0 - 1.0) ** 2], dtype=np.float64)
    log_var = np.array([0.0, np.log(2.0)], dtype=np.float64)
    inv_var = 1.0 / (np.exp(log_var) + 1e-16)
    ll = np.sum(-mse * (0.5 * inv_var) - 0.5 * log_var) / 2.0
    expected = -ll

    assert got == pytest.approx(expected)


def test_get_default_network_output_shape():
    net = boh.get_default_network(3)
    x = torch.randn(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_roundtrip():
    model = make_model()
    model.model = DummyNet(2).double()

    original = model.network_weights
    modified = [w + 1.0 for w in original]
    model.network_weights = modified
    roundtrip = model.network_weights

    assert len(original) == len(roundtrip)
    for a, b in zip(modified, roundtrip):
        assert np.allclose(a, b)


def test_train_rejects_invalid_batch_size():
    model = make_model()
    x = np.zeros((2, 1), dtype=np.float64)
    y = np.zeros((2,), dtype=np.float64)

    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=0)


def test_train_rejects_invalid_target_shape():
    model = make_model()
    x = np.zeros((2, 1), dtype=np.float64)
    y = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(AssertionError):
        model.train(x, y, num_steps=1, batch_size=1)


def test_train_uses_all_datapoints_when_batch_too_large(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", TrackingSampler)
    TrackingSampler.instances.clear()

    model = make_model()
    x = np.array([[0.0], [1.0]], dtype=np.float64)
    y = np.array([0.0, 1.0], dtype=np.float64)

    model.train(x, y, num_steps=1, batch_size=10, num_burn_in_steps=0, keep_every=1)

    assert len(TrackingSampler.instances) == 1
    # DataLoader batch_size should be clamped to number of points, reflected by one step
    assert TrackingSampler.instances[0].step_called == 1
    assert model.is_trained is True


def test_train_initializes_sampler_with_expected_kwargs(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", TrackingSampler)
    TrackingSampler.instances.clear()

    model = make_model(sampling_method="adaptive_sghmc")
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    model.train(x, y, num_steps=1, batch_size=2, num_burn_in_steps=5, lr=0.01, mdecay=0.07, epsilon=1e-9)

    sampler = TrackingSampler.instances[0]
    assert sampler.kwargs["scale_grad"] == np.float64(3.0)
    assert sampler.kwargs["num_burn_in_steps"] == 5
    assert sampler.kwargs["lr"] == np.float64(0.01)
    assert sampler.kwargs["mdecay"] == np.float64(0.07)
    assert sampler.kwargs["epsilon"] == np.float64(1e-9)


def test_train_collects_sampled_weights(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", TrackingSampler)
    TrackingSampler.instances.clear()

    model = make_model(normalize_input=False, normalize_output=False)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    model.train(x, y, num_steps=3, batch_size=2, num_burn_in_steps=0, keep_every=1)

    assert len(model.sampled_weights) == 3
    assert model.is_trained is True


def test_train_continue_training_preserves_sampled_weights(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", TrackingSampler)
    TrackingSampler.instances.clear()

    model = make_model(normalize_input=False, normalize_output=False)
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    model.train(x, y, num_steps=2, batch_size=2, num_burn_in_steps=0, keep_every=1)
    first_len = len(model.sampled_weights)
    model.train(x, y, num_steps=1, batch_size=2, num_burn_in_steps=0, keep_every=1, continue_training=True)

    assert len(model.sampled_weights) >= first_len


def test_predict_returns_mean_and_variance_without_normalization():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(1).double()

    w1 = (
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([0.0, np.log(1.0)], dtype=np.float64),
    )
    w2 = (
        np.array([[3.0], [0.0]], dtype=np.float64),
        np.array([0.0, np.log(1.0)], dtype=np.float64),
    )
    model.sampled_weights = [w1, w2]

    mu, var = model.predict(np.array([[2.0]], dtype=np.float64))

    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert mu[0] == pytest.approx(4.0)
    assert var[0] == pytest.approx(5.0)


def test_predict_return_individual_predictions():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(1).double()

    w1 = (
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([0.0, np.log(1.0)], dtype=np.float64),
    )
    w2 = (
        np.array([[2.0], [0.0]], dtype=np.float64),
        np.array([0.0, np.log(1.0)], dtype=np.float64),
    )
    model.sampled_weights = [w1, w2]

    mu, var, indiv = model.predict(np.array([[1.0]], dtype=np.float64), return_individual_predictions=True)

    assert indiv.shape == (2, 1)
    assert mu.shape == (1,)
    assert var.shape == (1,)
    assert np.allclose(indiv[:, 0], np.array([1.0, 2.0]))


def test_predict_single_returns_one_sample_prediction():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(1).double()
    weights = (
        np.array([[2.0], [0.0]], dtype=np.float64),
        np.array([0.0, np.log(1.0)], dtype=np.float64),
    )
    model.sampled_weights = [weights]

    out = model.predict_single(np.array([3.0], dtype=np.float64), sample_index=0)

    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(6.0)
    assert out[0, 1] == pytest.approx(0.0)


def test_f_gradient_matches_linear_model_gradient():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(2).double()
    with torch.no_grad():
        model.model.linear.weight.copy_(torch.tensor([[2.0, -3.0], [0.0, 0.0]], dtype=torch.float64))
        model.model.linear.bias.zero_()

    weights = model.network_weights
    g = model.f_gradient(np.array([4.0, 5.0], dtype=np.float64), weights=weights)

    assert g.shape == (2,)
    assert np.allclose(g, np.array([2.0, -3.0]), atol=1e-8)


def test_predictive_mean_gradient_averages_sample_gradients():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(1).double()

    w1 = (
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )
    w2 = (
        np.array([[3.0], [0.0]], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
    )
    model.sampled_weights = [w1, w2]

    g = model.predictive_mean_gradient(np.array([2.0], dtype=np.float64))

    assert g.shape == (1,)
    assert g[0] == pytest.approx(2.0)


def test_predictive_variance_gradient_zero_for_constant_offset_models():
    model = make_model(normalize_input=False, normalize_output=False)
    model.model = DummyNet(1).double()

    # Two models with identical slope; only bias differs, so predictive variance does not depend on x.
    w1 = (
        np.array([[2.0], [0.0]], dtype=np.float64),
        np.array([1.0, np.log(1.0)], dtype=np.float64),
    )
    w2 = (
        np.array([[2.0], [0.0]], dtype=np.float64),
        np.array([5.0, np.log(1.0)], dtype=np.float64),
    )
    model.sampled_weights = [w1, w2]

    g = model.predictive_variance_gradient(np.array([1.5], dtype=np.float64))

    assert g.shape == (1,)
    assert np.allclose(g, np.array([0.0]), atol=1e-8)


def test_train_and_evaluate_returns_expected_length(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", TrackingSampler)
    TrackingSampler.instances.clear()

    model = make_model(normalize_input=False, normalize_output=False)
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    x_valid = np.array([[4.0], [5.0]], dtype=np.float64)
    y_valid = np.array([4.0, 5.0], dtype=np.float64)

    n_steps, ll_curve, mse_curve = model.train_and_evaluate(
        x_train,
        y_train,
        x_valid,
        y_valid,
        num_steps=4,
        validate_every_n_steps=2,
        keep_every=1,
        num_burn_in_steps=0,
        batch_size=2,
        verbose=False,
    )

    assert n_steps == [2, 4]
    assert len(ll_curve) == 2
    assert len(mse_curve) == 2
    assert all(np.isfinite(v) for v in ll_curve)
    assert all(np.isfinite(v) for v in mse_curve)
