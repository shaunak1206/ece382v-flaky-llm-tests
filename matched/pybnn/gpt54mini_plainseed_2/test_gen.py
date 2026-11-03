import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class TinyNet(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.fc = nn.Linear(input_dimensionality, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


class DummySampler:
    instances = []

    def __init__(self, params, **kwargs):
        self.params = list(params)
        self.kwargs = kwargs
        self.zero_grad_called = 0
        self.step_called = 0
        DummySampler.instances.append(self)

    def zero_grad(self):
        self.zero_grad_called += 1

    def step(self):
        self.step_called += 1


def make_bnn(monkeypatch, *, normalize_input=False, normalize_output=False, use_double_precision=False,
             sampling_method="adaptive_sghmc", get_network=TinyNet):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", DummySampler)
    monkeypatch.setattr(boh, "SGLD", DummySampler)
    monkeypatch.setattr(boh, "SGHMC", DummySampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", DummySampler)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: loader)
    bnn = boh.Bohamiann(
        get_network=get_network,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
        sampling_method=sampling_method,
        use_double_precision=use_double_precision,
        print_every_n_steps=10,
    )
    return bnn


def test_nll_matches_manual_formula():
    pred = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.5, 1.0], dtype=torch.float64)
    got = boh.nll(pred, target).item()

    mean = pred[:, 0:1]
    log_var = pred[:, 1:2]
    inv_var = 1.0 / (torch.exp(log_var) + 1e-16)
    mse = (target.view(-1, 1) - mean) ** 2
    manual = -torch.sum(torch.sum(-mse * (0.5 * inv_var) - 0.5 * log_var, dim=1)) / pred.size(0)
    assert got == pytest.approx(manual.item())


def test_get_default_network_forward_shape():
    net = boh.get_default_network(3)
    x = torch.randn(4, 3)
    y = net(x)
    assert y.shape == (4, 2)


def test_network_weights_getter_and_setter_roundtrip(monkeypatch):
    bnn = make_bnn(monkeypatch)
    bnn.model = TinyNet(2).float()

    original = bnn.network_weights
    new_weights = [np.ones_like(w) * 3.0 for w in original]
    bnn.network_weights = new_weights
    roundtrip = bnn.network_weights

    assert len(original) == len(new_weights) == len(roundtrip)
    for arr in roundtrip:
        assert np.allclose(arr, 3.0)


def test_train_creates_expected_sampler_adaptive(monkeypatch):
    DummySampler.instances.clear()
    bnn = make_bnn(monkeypatch, sampling_method="adaptive_sghmc")
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)

    bnn.train(x, y, num_steps=1, keep_every=1, num_burn_in_steps=0, batch_size=2)

    assert isinstance(bnn.sampler, DummySampler)
    assert "scale_grad" in bnn.sampler.kwargs
    assert bnn.is_trained is True


@pytest.mark.parametrize(
    "sampling_method, expected_key",
    [
        ("sgld", "lr"),
        ("sghmc", "mdecay"),
        ("preconditioned_sgld", "num_train_points"),
    ],
)
def test_train_creates_expected_sampler_variants(monkeypatch, sampling_method, expected_key):
    DummySampler.instances.clear()
    bnn = make_bnn(monkeypatch, sampling_method=sampling_method)
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)

    bnn.train(x, y, num_steps=1, keep_every=1, num_burn_in_steps=0, batch_size=2)

    assert isinstance(bnn.sampler, DummySampler)
    assert expected_key in bnn.sampler.kwargs


def test_train_collects_sampled_weights(monkeypatch):
    DummySampler.instances.clear()
    bnn = make_bnn(monkeypatch, sampling_method="adaptive_sghmc")
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)

    bnn.train(x, y, num_steps=3, keep_every=1, num_burn_in_steps=0, batch_size=1)

    assert len(bnn.sampled_weights) == 2
    assert all(isinstance(w, tuple) for w in bnn.sampled_weights)


def test_train_respects_continue_training_and_keeps_samples(monkeypatch):
    DummySampler.instances.clear()
    bnn = make_bnn(monkeypatch, sampling_method="adaptive_sghmc")
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)

    bnn.train(x, y, num_steps=2, keep_every=1, num_burn_in_steps=0, batch_size=1)
    first_len = len(bnn.sampled_weights)
    old_model = bnn.model
    bnn.train(x, y, num_steps=1, keep_every=1, num_burn_in_steps=0, batch_size=1, continue_training=True)

    assert bnn.model is old_model
    assert len(bnn.sampled_weights) >= first_len


def test_predict_returns_expected_shapes_and_individual_outputs(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    x = np.zeros((5, 2), dtype=np.float32)
    mean, var, indiv = bnn.predict(x, return_individual_predictions=True)

    assert mean.shape == (5,)
    assert var.shape == (5,)
    assert indiv.shape == (2, 5)


def test_predict_applies_output_denormalization(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=True)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights]

    bnn.y_mean = 10.0
    bnn.y_std = 2.0
    x = np.zeros((2, 2), dtype=np.float32)

    mean, var = bnn.predict(x)
    assert np.allclose(var, np.exp(bnn.model(torch.from_numpy(x).float()).detach().numpy()[:, 1]) * 4.0)
    assert mean.shape == (2,)


def test_predict_single_uses_one_sample(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    x = np.array([0.0, 0.0], dtype=np.float32)
    out = bnn.predict_single(x, sample_index=1)

    assert out.shape == (1, 2)


def test_f_gradient_returns_input_gradient(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights]
    x = np.array([1.0, -2.0], dtype=np.float32)

    g = bnn.f_gradient(x, bnn.sampled_weights[0])

    assert g.shape == (2,)
    assert np.isfinite(g).all()


def test_predictive_mean_gradient_averages_over_samples(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    def fake_grad(x_test, weights):
        return np.array([1.0, 3.0])

    monkeypatch.setattr(bnn, "f_gradient", fake_grad)
    g = bnn.predictive_mean_gradient(np.array([0.0, 0.0], dtype=np.float32))

    assert np.allclose(g, np.array([1.0, 3.0]))


def test_predictive_variance_gradient_zero_for_constant_predictions(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights, bnn.network_weights]

    monkeypatch.setattr(bnn, "predict", lambda x, return_individual_predictions=False: (
        np.array([1.0]), np.array([0.0]), np.array([[1.0], [1.0]])
    ))
    monkeypatch.setattr(bnn, "f_gradient", lambda x_test, weights: np.array([2.0, -1.0]))
    monkeypatch.setattr(bnn, "predictive_mean_gradient", lambda x_test: np.array([2.0, -1.0]))

    g = bnn.predictive_variance_gradient(np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(g, np.zeros(2))


def test_train_and_evaluate_returns_learning_curves(monkeypatch):
    bnn = make_bnn(monkeypatch, normalize_input=False, normalize_output=False)
    bnn.model = TinyNet(2).float()
    bnn.sampled_weights = [bnn.network_weights]

    def fake_train(*args, **kwargs):
        bnn.sampled_weights = [bnn.network_weights]

    monkeypatch.setattr(bnn, "train", fake_train)
    monkeypatch.setattr(bnn, "predict", lambda x: (np.zeros(x.shape[0]), np.ones(x.shape[0])))

    xtr = np.zeros((4, 2), dtype=np.float32)
    ytr = np.zeros(4, dtype=np.float32)
    xva = np.zeros((2, 2), dtype=np.float32)
    yva = np.zeros(2, dtype=np.float32)

    steps, ll, mse = bnn.train_and_evaluate(
        xtr, ytr, xva, yva,
        num_steps=2,
        validate_every_n_steps=1,
        keep_every=1,
        num_burn_in_steps=0,
        batch_size=1,
    )

    assert steps == [1, 2]
    assert len(ll) == len(mse) == 2
