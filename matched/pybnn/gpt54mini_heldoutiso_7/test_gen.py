import numpy as np
import pytest
import torch
import torch.nn as nn

import pybnn.bohamiann as boh


class TinyNetwork(nn.Module):
    def __init__(self, input_dimensionality):
        super().__init__()
        self.linear = nn.Linear(input_dimensionality, 1, bias=True)
        self.log_var = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))

    def forward(self, x):
        mu = self.linear(x)
        log_var = self.log_var.expand_as(mu)
        return torch.cat([mu, log_var], dim=1)


class FakeSampler:
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
                p.add_(0.01)


def _cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _make_data(n=6, d=2):
    x = np.arange(n * d, dtype=np.float64).reshape(n, d) / 10.0
    y = x.sum(axis=1) - 0.5
    return x, y


@pytest.fixture
def patch_bohamiann(monkeypatch):
    monkeypatch.setattr(boh, "AdaptiveSGHMC", FakeSampler)
    monkeypatch.setattr(boh, "SGLD", FakeSampler)
    monkeypatch.setattr(boh, "SGHMC", FakeSampler)
    monkeypatch.setattr(boh, "PreconditionedSGLD", FakeSampler)
    monkeypatch.setattr(boh, "weight_prior", lambda params, dtype=None: torch.tensor(0.0, dtype=torch.float64))
    monkeypatch.setattr(boh, "log_variance_prior", lambda x: torch.tensor(0.0, dtype=torch.float64))
    monkeypatch.setattr(boh, "infinite_dataloader", lambda loader: _cycle_loader(loader))
    return monkeypatch


def test_nll_matches_manual_computation():
    inp = torch.tensor([[1.0, 0.0], [2.0, np.log(2.0)]], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)

    got = boh.nll(inp, target).item()

    # sample 1: mse=0, logvar=0 => contribution 0
    # sample 2: mse=4, invvar=1/2 => -4*(0.5*0.5) - 0.5*log(2) = -1 - 0.34657359
    expected = -((-0.0 + (-1.0 - 0.5 * np.log(2.0))) / 2.0)
    assert got == pytest.approx(expected)


def test_default_network_output_shape():
    model = boh.get_default_network(3)
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 2)


def test_network_weights_getter_returns_tuple_of_numpy_arrays(patch_bohamiann):
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.model = TinyNetwork(2).double()

    weights = b.network_weights
    assert isinstance(weights, tuple)
    assert all(isinstance(w, np.ndarray) for w in weights)
    assert len(weights) == len(list(b.model.parameters()))


def test_network_weights_setter_changes_parameters(patch_bohamiann):
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.model = TinyNetwork(2).double()

    new_weights = [np.ones_like(p.detach().numpy()) for p in b.model.parameters()]
    b.network_weights = new_weights

    for param in b.model.parameters():
        assert np.allclose(param.detach().numpy(), 1.0)


def test_normalize_input_and_output_delegate_to_util():
    b = boh.Bohamiann(get_network=TinyNetwork)
    x = np.array([1.0, 2.0, 3.0])
    normed, m, s = b.normalize_input(x)
    assert np.allclose(normed.mean(), 0.0)
    assert np.allclose(normed.std(), 1.0)
    denorm = boh.zero_mean_unit_var_denormalization(normed, m, s)
    assert np.allclose(denorm, x)


def test_train_sets_is_trained_and_samples_weights(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork, print_every_n_steps=1)
    b.train(x, y, num_steps=4, num_burn_in_steps=0, keep_every=1, batch_size=2, verbose=False)

    assert b.is_trained is True
    assert len(b.sampled_weights) == 4
    assert isinstance(b.sampler, FakeSampler)


def test_train_uses_small_batch_size_when_data_smaller_than_batch(patch_bohamiann):
    x, y = _make_data(n=3, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=10, verbose=False)
    assert b.is_trained is True
    assert len(b.sampled_weights) == 1


def test_train_rejects_invalid_batch_size(patch_bohamiann):
    x, y = _make_data(n=4, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    with pytest.raises(AssertionError):
        b.train(x, y, num_steps=1, batch_size=0)


def test_train_rejects_invalid_target_shape(patch_bohamiann):
    x, y = _make_data(n=4, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    with pytest.raises(AssertionError):
        b.train(x, y.reshape(-1, 1, 1), num_steps=1)


def test_train_initializes_sampler_for_each_sampling_method(patch_bohamiann):
    x, y = _make_data(n=5, d=2)
    methods = {
        "adaptive_sghmc": {"scale_grad": np.float64(len(x)), "num_burn_in_steps": 0},
        "sgld": {"scale_grad": len(x)},
        "preconditioned_sgld": {"num_train_points": len(x)},
        "sghmc": {"scale_grad": np.float64(len(x))},
    }
    for method in methods:
        b = boh.Bohamiann(get_network=TinyNetwork, sampling_method=method)
        b.train(x, y, num_steps=1, num_burn_in_steps=0, keep_every=1, batch_size=2)
        assert isinstance(b.sampler, FakeSampler)


def test_predict_returns_mean_variance_and_individual_predictions(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=2)

    mu, var, indiv = b.predict(x[:2], return_individual_predictions=True)
    assert mu.shape == (2,)
    assert var.shape == (2,)
    assert indiv.shape == (len(b.sampled_weights), 2)


def test_predict_single_matches_predict_for_one_sample(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2)

    single = b.predict_single(x[:1], sample_index=0)
    _, _, indiv = b.predict(x[:1], return_individual_predictions=True)
    assert single.shape == (1, 2)
    assert np.allclose(single[:, 0], indiv[0], atol=1e-6)


def test_predict_without_training_raises_or_fails_cleanly():
    b = boh.Bohamiann(get_network=TinyNetwork)
    with pytest.raises(Exception):
        b.predict(np.zeros((1, 2)))


def test_f_gradient_returns_expected_shape(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=2, num_burn_in_steps=0, keep_every=1, batch_size=2)

    g = b.f_gradient(x[0], b.sampled_weights[0])
    assert g.shape == (2,)


def test_predictive_mean_gradient_has_input_shape(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=2)

    g = b.predictive_mean_gradient(x[0])
    assert g.shape == (2,)


def test_predictive_variance_gradient_has_input_shape(patch_bohamiann):
    x, y = _make_data(n=6, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)
    b.train(x, y, num_steps=3, num_burn_in_steps=0, keep_every=1, batch_size=2)

    g = b.predictive_variance_gradient(x[0])
    assert g.shape == (2,)


def test_train_and_evaluate_returns_learning_curves(patch_bohamiann):
    x_train, y_train = _make_data(n=8, d=2)
    x_valid, y_valid = _make_data(n=4, d=2)
    b = boh.Bohamiann(get_network=TinyNetwork)

    steps, ll, mse = b.train_and_evaluate(
        x_train, y_train, x_valid, y_valid,
        num_steps=4, validate_every_n_steps=2, keep_every=1,
        num_burn_in_steps=0, batch_size=2, verbose=False
    )

    assert steps == [2, 4]
    assert len(ll) == len(mse) == 2
    assert all(np.isfinite(v) for v in ll)
    assert all(np.isfinite(v) for v in mse)
